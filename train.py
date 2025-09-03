import os
import pickle
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group

from model import model,Modelargs
from data import train_loader,val_loader

"""
Configuration block → sets defaults.
DDP setup → checks if we are distributed, initializes processes, adjusts gradient accumulation.
Data loading → simple np.memmap reader with get_batch().
Model setup → scratch / resume / pretrained.
Optimizer + AMP scaler.
Compile + wrap in DDP.
Training loop →
get lr
eval + checkpoint every eval_interval
forward + backward with AMP and grad accumulation
optimizer step + log
"""
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = "mimic cxr" # 'run' + str(time.time())
# data
dataset = 'mimic cxr'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024


# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


init_from = 'pretrained' # 'scratch' or 'resume' or 'pretrained'
# DDP settings
backend="nccl"
#system
device="cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype="bfolat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
compile=True

ddp=int(os.environ.get("RANK",-1))!= -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank=int(os.environ("RANK"))
    ddp_local_rank=int(os.environ("LOCAL_RANK"))
    ddp_world_size=int(os.environ("WORLD_SIZE"))
    device=f"cuda:{ddp_local_rank}"

    master_process=ddp_rank==0  # this process will do logging, checkpointing etc.
    
    seed_offset=ddp_rank
    assert gradient_accumaltion_steps%ddp_world_size==0
    gradient_accumation_steps//=ddp_world_size

else:
    # if not ddp, we are running on a single gpu, and one process
    master_process=True
    seed_offset=0
    ddp_world_size=1

tokens_per_iter=gradient_accumation_steps*ddp_world_size*batch_size*seqlen
print(f"tokens per iteration will be: {tokens_per_iter:,}")


if master_process:
    os.makedirs(out_dir,exist_ok=True)

torch.manual_seed(1337+seed_offset)
torch.backends.cuda.matmul.allow_tf32=True #allow matmul in fp32
torch.backends.cudann.allow_tf=True  # allow tf32 on cudnn

device_type="cuda" if "cuda" in device else "cpu" # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

ptdtype={"float32":torch.float32,"bfloat16":torch.bfloat16,"float16":torch.float16}[dtype]

ctx=nullcontext() if device_type== "cpu" else torch.amp.autocast(device_type=device_type,ptdtype=ptdtype)



model.to(device)
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler=torch.cuda.amp.GradScaler(enabled=(dtype=="float16"))
# optimizer
if compile:
    print("compiling the model ... takes a few minutes")
    unoptimized_model=model
    model=torch.compile(model)

# wrap model into DDP container
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank])

@torch.no_grad()

def estimate_loss():
    out={}
    model.eval()


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
        # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)





#logging 
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project,name=wandb_run_name,config=config)


# training loop
for X, Y  in train_loader:# fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed



    # determine and set the learning rate for this iteration
    lr=get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:


    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log({
                "iter":iter_num,
                "train_loss":losses["train"],
                "val_loss":losses["val"],
                "lr":lr,
                "mfu":running_mfu*100, #convert to percentage
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:  
                checkpoint={
                    "model": raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,

                }

                torch.save(checkpoint,os.path.join(put,dir,"ckpt.pt"))   


 
    if iter_num==0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16

    for micro_step in range(gradient_accumation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable

            model.require_backward_grad_sync=(micro_step=gradient_accumalation_steps-1)

        with ctx:
            logits,loss=model(X,Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        
        # backward pass, with gradient scaling if training in fp16    
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break


