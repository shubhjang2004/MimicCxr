import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchxrayvision as xrv
import pandas as pd
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Paths
    train_csv = "train.csv"
    val_csv = "val.csv"
    save_dir = "checkpoints"
    
    # Model
    text_model = "microsoft/biogpt"
    
    # Training
    batch_size = 4
    num_epochs = 10
    learning_rate = 2e-5
    max_length = 128
    gradient_accumulation_steps = 2
    
    # Image
    image_size = 512
    num_visual_tokens = 256
    resnet_dim = 2048
    biogpt_dim = 1024
    resume_checkpoint = "checkpoints/best_model.pt"
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mixed precision
    use_amp = True

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# ==============================================================================
# DATASET
# ==============================================================================

def load_checkpoint(model, optimizer, checkpoint_path, config):
    """Load checkpoint and return starting epoch"""
    if os.path.exists(checkpoint_path):
        print("\n" + "="*70)
        print(f" LOADING CHECKPOINT: {checkpoint_path}")
        print("="*70)
        
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        print(f" Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   Previous train loss: {checkpoint['train_loss']:.4f}")
        print(f"   Previous val loss: {checkpoint['val_loss']:.4f}")
        print(f" Resuming training from epoch {start_epoch}")

class MIMICDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.mimic_df = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
             # Don't normalize - let TorchXRayVision handle it naturally
        ])
        print(f" Loaded {len(self.mimic_df)} samples from {csv_file}")

    def __len__(self):
        return len(self.mimic_df)

    def __getitem__(self, idx):
        row = self.mimic_df.iloc[idx]
        report = row["report_text"]
        image_path = row["image_path"].replace('\\', '/')
        
        # Load image
        img = Image.open(image_path).convert("L")  # Grayscale
        img = self.transform(img)  # [1, 512, 512]

        return {
            "image": img,
            "report": report
        }

def make_collate_fn(tokenizer, max_len=128):
    def collate_fn(batch):
        texts = [ex["report"] for ex in batch]
        images = torch.stack([ex["image"] for ex in batch])

        # Tokenize reports
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            "images": images,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        }
    return collate_fn

# ==============================================================================
# IMAGE ENCODER
# ==============================================================================

class ImageEncoder(nn.Module):
    def __init__(self, xrv_model, resnet_dim=2048, biogpt_dim=1024):
        super().__init__()
        # Feature extractor (gives [B, 2048, 16, 16])
        self.feature_extractor = nn.Sequential(
            *list(xrv_model.model.children())[:-2]
        )
        
        # Project 2048 → 1024
        self.projection = nn.Sequential(
            nn.Linear(resnet_dim, biogpt_dim),
            nn.LayerNorm(biogpt_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, images):
        # Extract features
        features = self.feature_extractor(images)  # [B, 2048, 16, 16]
        
        # Reshape to tokens
        B, C, H, W = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # [B, 256, 2048]
        
        # Project
        tokens = self.projection(tokens)  # [B, 256, 1024]
        
        return tokens

# ==============================================================================
# COMPLETE MODEL
# ==============================================================================

class MedicalReportGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load image encoder
        print("Loading ResNet50 from TorchXRayVision...")
        xrv_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
        self.image_encoder = ImageEncoder(
            xrv_resnet, 
            resnet_dim=config.resnet_dim,
            biogpt_dim=config.biogpt_dim
        )
        
        # Load text decoder
        print("Loading BioGPT...")
        self.text_decoder = AutoModelForCausalLM.from_pretrained(config.text_model)
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        print(f"Image Encoder:   ResNet50-res512-all (medical pretrained)")
        print(f"Visual Tokens:   {config.num_visual_tokens}")
        print(f"Token Dimension: {config.resnet_dim} → {config.biogpt_dim}")
        print(f"Text Decoder:    BioGPT")
        print("="*70 + "\n")
    
    def forward(self, images, input_ids, attention_mask):
        """
        Training forward pass using two-pass method with caching
    
        Args:
            images: [B, 1, 512, 512]
            input_ids: [B, T] - tokenized reports
            attention_mask: [B, T] - attention mask for reports
    
        Returns:
            loss: scalar
        """
        B = images.shape[0]
        V = self.config.num_visual_tokens  # 256
    
        # ============================================================
        # PASS 1: Encode images and cache
        # ============================================================
    
        # Get image embeddings
        image_embeds = self.image_encoder(images)  # [B, 256, 1024]
    
        # Create prefix attention mask (all ones - all image tokens visible)
        prefix_mask = torch.ones(B, V, dtype=torch.long, device=images.device)
    
        # Forward through decoder to cache image context
        # Use no_grad to save memory (we only want the cache)
        with torch.no_grad():
            out_prefix = self.text_decoder(
                inputs_embeds=image_embeds,
                attention_mask=prefix_mask,
                use_cache=True,
                output_hidden_states=False
            )
        past_kv = out_prefix.past_key_values
    
        # ============================================================
        # PASS 2: Generate text using cached image context
        # ============================================================
    
        # Combined attention mask: image prefix + text
        # This tells the model: "text tokens can see image tokens + previous text"
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, V+T]
    
        # Labels: just the input_ids (model will shift internally for next-token prediction)
        labels = input_ids.clone()  # [B, T]
    
        # Forward through text decoder with cached image context
        outputs = self.text_decoder(
            input_ids=input_ids,           # [B, T] - text tokens to process
            attention_mask=combined_mask,  # [B, V+T] - can attend to image + text
            past_key_values=past_kv,       # Cached K/V for image tokens
            labels=labels,                 # [B, T] - targets for next-token prediction
            use_cache=True
        )
    
        return outputs.loss
    
    @torch.no_grad()
    def generate(self, images, max_length=128, num_beams=4):
        """
        Generate reports from images
        
        Args:
            images: [B, 1, 512, 512]
            max_length: maximum tokens to generate
            num_beams: beam search width
        
        Returns:
            generated_ids: [B, max_length]
        """
        self.eval()
        
        # Encode images
        image_embeds = self.image_encoder(images)  # [B, 256, 1024]
        
        # Generate from image embeddings
        generated_ids = self.text_decoder.generate(
            inputs_embeds=image_embeds,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=self.text_decoder.config.pad_token_id,
            eos_token_id=self.text_decoder.config.eos_token_id
        )
        
        return generated_ids

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_epoch(model, dataloader, optimizer, scaler, config, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress):
        # Move to device
        images = batch["images"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        
        # Forward pass with mixed precision
        if config.use_amp:
            with autocast():
                loss = model(images, input_ids, attention_mask)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss = model(images, input_ids, attention_mask)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * config.gradient_accumulation_steps
        progress.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def validate(model, dataloader, config):
    model.eval()
    total_loss = 0
    
    progress = tqdm(dataloader, desc="Validation")
    for batch in progress:
        images = batch["images"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        
        if config.use_amp:
            with autocast():
                loss = model(images, input_ids, attention_mask)
        else:
            loss = model(images, input_ids, attention_mask)
        
        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    print("="*70)
    print("MEDICAL REPORT GENERATION TRAINING")
    print("="*70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MIMICDataset(config.train_csv)
    val_dataset = MIMICDataset(config.val_csv)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=make_collate_fn(tokenizer, config.max_length),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=make_collate_fn(tokenizer, config.max_length),
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = MedicalReportGenerator(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, config, epoch)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, config)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f" Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print("-"*70 + "\n")
    
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")

# ==============================================================================
# INFERENCE EXAMPLE
# ==============================================================================

def inference_example():
    """Example of how to generate reports from images"""
    
    # Load model
    model = MedicalReportGenerator(config).to(config.device)
    checkpoint = torch.load(os.path.join(config.save_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    
    # Load a sample image
    val_dataset = MIMICDataset(config.val_csv)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, config.max_length)
    )
    
    # Generate
    batch = next(iter(val_loader))
    images = batch["images"].to(config.device)
    
    print("Generating report...")
    generated_ids = model.generate(images, max_length=config.max_length, num_beams=4)
    
    # Decode
    generated_report = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    ground_truth = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
    
    print("\n" + "="*70)
    print("GENERATED REPORT:")
    print("="*70)
    print(generated_report)
    print("\n" + "="*70)
    print("GROUND TRUTH:")
    print("="*70)
    print(ground_truth)

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    # Train
    main()
    
# Upload checkpoint
from google.colab import files
uploaded = files.upload()  # Upload best_model.pt

# Load and continue
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f" Resuming from epoch {checkpoint['epoch'] + 1}")
print(f"   Previous best val loss: {checkpoint['val_loss']:.4f}")

# Continue training from epoch 5
for epoch in range(4, 8):  # Epochs 5-8
    train_epoch(...)    