import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
#biogpt = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
tokens=torch.tensor(tokenizer.encode("hello shubham how is "),dtype=torch.long).unsqueeze(0)

"""
BioGptForCausalLM(
  (biogpt): BioGptModel(
    (embed_tokens): BioGptScaledWordEmbedding(42384, 1024, padding_idx=1)
    (embed_positions): BioGptLearnedPositionalEmbedding(1026, 1024)
    (layers): ModuleList(
      (0-23): 24 x BioGptDecoderLayer(
        (self_attn): BioGptAttention(
          (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=1024, out_features=4096, bias=True)
        (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (output_projection): Linear(in_features=1024, out_features=42384, bias=False)
)

"""
@dataclass
class Modelargs:
    dim:int=1024
    seqlen:int=1024
    vocab_size:int=42384
    n_heads:int=16
    n_layers:int=24
    

class BioGptForCausalLM(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.biogpt=BioGptModel(args)
        self.output_projection=nn.Linear(args.dim,args.vocab_size,bias=False)

        self.output_projection.weight = self.biogpt.embed_tokens.embed.weight
        self.apply(self.init_weights)
 
    def forward(self,x,y=None):
        x=self.biogpt(x)
        logits=self.output_projection(x)
        if y is not None:
            loss=nn.CrossEntropyLoss(logits.view(-1,logits.size(-1)),y.view(-1),ignore_index=-100,reduction="mean")

            return logits,loss

        else:
            return logits  
    
    def get_params(self,non_embedding=True):
      n_params = sum(p.numel() for p in self.parameters())
      if non_embedding :
         n_params  -= self.transformer.wte.weight.numel()
      return n_params

    def init_weights(self,Module):
      if isinstance(Module,nn.Linear):
         torch.nn.init.normal(Module.weight,mean=0.0,std=0.02)
         if Module.bias is not None:
             torch.nn.init.zeros_(Module.bias)
      elif isinstance(Module,nn.Embedding):
          torch.nn.init.normal_(Module.weight,mean=0.0,std=0.2)
    




class BioGptModel(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.embed_tokens=BioGptScaledWordEmbedding(args.vocab_size,args.dim)
        self.embed_positions=BioGpTLearnedPositionalEmbeddig(args.seqlen,args.dim)
        self.layers=nn.ModuleList([BioGptDecoderLayer(args) for _ in range(args.n_layers)])
        self.layer_norm=nn.LayerNorm(args.dim)

    def forward(self,x):
        B,T=x.size()
        positions=torch.arange(0,T,dtype=torch.long,device=x.device)
        embed=self.embed_tokens(x)
        posn_embed=self.embed_positions(positions)

        x=embed+posn_embed

        for layer in self.layers:
            x=layer(x)

        x=self.layer_norm(x)
        return x        



class BioGptDecoderLayer(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.self_attn=BioGptAttention(args)
        self.activation_fn=nn.GELU()

        self.self_attn_layer_norm=nn.LayerNorm(args.dim)
        self.fc1=nn.Linear(args.dim,4*args.dim)
        self.fc2=nn.Linear(4*args.dim,args.dim)
        self.final_layer_norm=nn.LayerNorm(args.dim)

    def forward(self,x):

        x=x+self.self_attn_layer_norm(self.self_attn(x))
        x=x+self.final_layer_norm(self.fc2(self.activation_fn(self.fc1(x))))

        return x

class BioGptAttention(nn.Module):

    def __init__(self,args:Modelargs):
        super().__init__()

        self.k_proj=nn.Linear(args.dim,args.dim)
        self.v_proj=nn.Linear(args.dim,args.dim)
        self.q_proj=nn.Linear(args.dim,args.dim)
        self.out_proj=nn.Linear(args.dim,args.dim)

        self.n_heads=args.n_heads
        self.head_dim=args.dim//args.n_heads     

        self.register_buffer("bias",torch.tril(torch.ones(args.seqlen,args.seqlen)).view(1,1,args.seqlen,args.seqlen))
        self.softmax_scale = self.head_dim**-0.5

    def forward(self,x):
        B,T,C=x.size()

        assert C%self.n_heads==0, f"dimension mismatch"

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        query=q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        key=k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        value=v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)

        attn=query@key.transpose(-2,-1)*self.softmax_scale

        attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
        attn = torch.softmax(attn, dim=-1)   # <--- missing
        output=attn@value
      
        
        output=output.transpose(1,2).contiguous().view(B,T,C)

        return self.out_proj(output)
        



class BioGptScaledWordEmbedding(nn.Module):
    def __init__(self,vocab_size:int,dim:int):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,dim)

    def forward(self,x):
        return self.embed(x)

class BioGpTLearnedPositionalEmbeddig(nn.Module):
    def __init__(self,seqlen:int,dim:int):
        super().__init__()
        self.embed_positions=nn.Embedding(seqlen+2,dim)

    def forward(self,x):
        return self.embed_positions(x)  


args=Modelargs()
biomodel=BioGptForCausalLM(args)




