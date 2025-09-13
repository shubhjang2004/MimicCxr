import os
import time
import math
import torch
import torch
import pickle

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
from contextlib import nullcontext
from torch.utils.data import DataLoader,DistributedSampler
from data import train_dataset,val_dataset

from model import MimicCxr,MimicCxrArgs

#------------------config------------------------------------------
out_dir="out"
os.makedirs(out_dir,exist_ok=True)

#logging 
wandb_log=False
wandb_project="mimic_cxr"
wandb_run_name="medical"

per_device_batch_size=64
gradient_accumulation_steps = 5 * 8 
init_from="pretrained" # resume or hf_pretrained or scratch

#DDP settings
backend="nccl"
device="cuda"
dtype="bfloat32" if torch.cuda.is_availabel() and torch.cuda.is_bf16_supported() else "float16"


# optimizer / schedule
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

compile_model=True

#--------------------------------DDP init--------------------

ddp=int(os.environ.get("RANK",-1)!=-1)
if ddp:
    init_process_group(backend=backend)
    ddp_rank=int(os.environ["RANK"])
    ddp_local_rank=int(os.environ["LOCAL_RANK"])
    ddp_world_size=int(os.environ["WORLD_SIZE"])

    master_process=ddp_rank==0
    device=f"cuda:{ddp_local_rank}"
    seed_offset=ddp_local_rank

    assert gradient_accumulation_steps%ddp_world_size==0
    gradient_accumulation_steps//=ddp_world_size

else:
    master_process=True
    ddp_world_size=1
    seed_offset=0


# ----------------- Seeds & backend flags -----------------
torch.cuda.manual_seed(1472+seed_offset)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudann.allow_tf32=True


device_type="cuda" if "cuda" in device else "cpu"
ptdtype={"float32":torch.float32,"bfloat32":torch.bfloat32,"float16":torch.float16}[dtype]


# -------------------------------Defining model ----------------------------------

# attempt to derive vocab_size from the dataset
metapath=os.path.join("data_dir","meta.pkl")
meta_vocab_size=None

if os.path.exists(metapath):
    with open(metapath,"rb") as f:
        meta=pickle.load(f)

    meta_vocab_size=meta["vocab_size"]

    print(f"vocab_size derived from meta")   

# model init
"""
if init_from=="scratch":
    #init a new model from scratch
    print(f"starting a new model from scratch")
    if meta_vocab_size is None:
        MimicCxrArgs["vocab_size"]=40245

    MimicCxrArgs['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model=MimicCxr(MimicCxrArgs)
"""
if init_from=="resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path=os.path.join(out_dir,"ckpt.pt")
    checkpoint=torch.load(ckpt_path,map_location=device)
    checkpoint_model_args=checkpoint["model_args"]
    # force these MimicCxr attributes to be equal otherwise we can't even resume training
    for k,v in MimicCxrArgs.items():
        MimicCxrArgs[k]=checkpoint_model_args[k]

    model=MimicCxr(MimicCxrArgs)
    state_dict=checkpoint["model"]
    model.load_state_dict(state_dict)  

elif init_from=="hf_pretrained":
    print("training from pretrained models")
    print("no need to load the weights just pass the arguments")

    model=MimicCxr(MimicCxrArgs) 


model=model.to(device)

ctx=nullcontext() if device_type=="cpu" else torch.autocast(device_type=device_type,ptdtype=ptdtype)
scaler=torch.cuda.amp.GradScaler(enabled=(dtype=="float16"))     

# ----------------compiling the model -----------------
if compile:
    print("compiling model ...may take few minutes")
    unoptimized_model=model
    model=torch.compile(model)

#-----------LR schedule-------------------------------------------------
def get_lr(it):
    if not decay_lr:
        return learning_rate
    
    if it>lr_decay_iters:
        return min_lr
    
    if it<warmup_iters:
        return (learning_rate*((it+1)/(warmup_iters+1)))
    
    else:
        decay_ratio=(it-warmup_iters)/(lr_decay_iters-warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr +coeff* (learning_rate - min_lr)



# ----------------- Data --------------------------------------------------
        
if ddp:
    train_sampler=DistributedSampler(train_dataset,num_replicas=ddp_world_size,shuffle=True)
    val_sampler=DistributedSampler(val_dataset,num_replicas=ddp_world_size,)


else:
    train_sampler=None
    val_sampler=None

train_loader=DataLoader(
    train_dataset,
    batch_size=per_device_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)    

val_loader=DataLoader(
    val_dataset,
    batch_size=per_device_batch_size,
    sampler=val_sampler,
    shuffle=False,
    num_workers=4
    pin_memory=True,
    drop_last=True,
)
    

#-------------------define how long to train here 5 epochs(sweeps) over train_loader-----------
epochs=5
steps_per_epoch=len(train_loader)
max_iters=epochs*steps_per_epoch
eval_iters=max(1,len(val_loader))

log_interval=max(1,steps_per_epoch//100)
save_interval=max(1,steps_per_epoch//5)

#--------------custom function for eval over val_loader----------------------
@torch.no_grad()
def evaluate(eval_iters=eval_iters):
    model.eval()
    total = 0.0
    n = 0
    val_iter = iter(val_loader)
    for _ in range(eval_iters):
        batch = next(val_iter, None)
        if batch is None:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        X, Y = batch
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        with ctx:
            _, loss = model(X, Y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


# ----------------- Training loop ------------------------------------
raw_model = model.module if ddp else model # unwrap DDP container if needed
iter_num=0
for epoch in range(epochs):
    # important for DistributedSampler shuffling
    if ddp and isinstance(train_sampler,DistributedSampler):
        train_sampler.set_epoch(epoch)

    train_iter=iter(train_loader)

    while True:
        if iter_num>(epoch+1)*steps_per_epoch:
            break

        # set LR
        lr_now = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now  

        t0 =time.time()
        # gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(gradient_accumulation_steps):
            batch=next(train_iter,None)
            
            if batch is None:
                # restart loader if exhausted (shouldn't happen inside epoch with drop_last=True)
                train_iter=iter(train_loader)
                batch = next(train_iter)

            X,Y=batch
            X=X.to(device,non_blocking=True)
            Y=Y.to(device,non_blocking=True)

             # avoid gradient sync on non-final micro steps in DDP
            maybe_no_sync = model.no_sync if (ddp and micro_step < gradient_accumulation_steps - 1) else nullcontext
            with maybe_no_sync():
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()        

            # clip, step, update scaler
        if grad_clip>0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()    

        t1=time.time()
        if master_process and (iter_num%log_interval==0):
            # quick on-the-fly train loss (last microbatch scaled back)
            train_loss_display = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num} | epoch {epoch+1}/{epochs} | lr {lr_now:.3e} | loss {train_loss_display:.4f} | {1000*(t1-t0):.0f} ms")
            if wandb_log:
                import wandb
                wandb.log({"train/loss": train_loss_display, "lr": lr_now, "iter": iter_num, "epoch": epoch})


         # evaluate + save at epoch boundary (or any interval you like)
         # evaluate + save at epoch boundary (or any interval you like)
        end_of_epoch = ((iter_num + 1) % steps_per_epoch == 0)
        if end_of_epoch and master_process:
            val_loss = evaluate(eval_iters=eval_iters)
            print(f"[eval] epoch {epoch+1}: val/loss {val_loss:.4f}")
            if wandb_log:
                wandb.log({"val/loss": val_loss, "epoch": epoch})

            # save checkpoint once per epoch
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": MimicCxrArgs,
                "iter_num": iter_num,
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_epoch{epoch+1}.pt"))
            # also keep a rolling "latest"
            torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

        iter_num += 1


# ----------------- Cleanup --------------------------------
if ddp:
    destroy_process_group()




    








