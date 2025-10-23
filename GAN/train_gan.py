import argparse
import math
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from MYGAN import Discriminator,Generator,ADAugment 
from data import loader

# -----------------------------
# Utilities
# -----------------------------
def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lerp(a, b, t):
    return a + (b - a) * t


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()
    return real_loss + fake_loss


def r1_penalty(real_pred, real_imgs):
    grads = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_imgs, create_graph=True)[0]
    penalty = grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()
    return penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def path_length_regularization(fake_imgs, latents, pl_mean, decay=0.01):
    # approximate path length regularization as in StyleGAN2
    # fake_imgs: [N, C, H, W]
    noise = torch.randn_like(fake_imgs) / math.sqrt(fake_imgs.shape[2] * fake_imgs.shape[3])
    grad = torch.autograd.grad(outputs=(fake_imgs * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(1).mean(1) + 1e-8) if grad.dim() > 1 else torch.sqrt((grad.pow(2).sum(1) + 1e-8))
    # update mean
    pl_mean_new = lerp(pl_mean, path_lengths.mean().item(), decay)
    pl_penalty = (path_lengths - pl_mean_new).pow(2).mean()
    return pl_penalty, pl_mean_new


# -----------------------------
# Training loop
# -----------------------------

def save_checkpoint(state, outdir, step):
    os.makedirs(outdir, exist_ok=True)  # Fixed: os.makedirs instead of os.make_dir
    ckpt = os.path.join(outdir, f"checkpoint_{step}.pt")
    torch.save(state, ckpt)
    print(f"Saved checkpoint {ckpt}")


def sample_and_save(G_ema, z_samples, outdir, step, device, grayscale=True):
    os.makedirs(outdir, exist_ok=True)
    G_ema.eval()
    with torch.no_grad():
        imgs = G_ema(z_samples.to(device))
        imgs = (imgs.clamp(-1, 1) + 1) * 0.5  # [0,1]
        
        # Since generator now outputs 1 channel, no conversion needed
        # Just ensure it's properly formatted
        if imgs.shape[1] != 1:
            imgs = imgs.mean(dim=1, keepdim=True)  # Fallback if needed
            
        grid = utils.make_grid(imgs.cpu(), nrow=int(math.sqrt(imgs.shape[0])), normalize=False)
        out_path = os.path.join(outdir, f"samples_{step:06d}.png")
        utils.save_image(grid, out_path)
        print(f"Saved sample grid to {out_path}")
        

from torch.cuda.amp import autocast, GradScaler

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
    G = Generator(
    z_dim=512, 
    w_dim=512, 
    max_channels=256,  # Reduced from 512
    resolution=args.resolution
        ).to(device)


    D = Discriminator(
    resolution=args.resolution, 
    max_channels=256  # Reduced from 512
        ).to(device)
    G_ema = Generator(z_dim=args.z_dim, w_dim=args.w_dim, max_channels=args.max_channels, resolution=args.resolution).to(device)
    G_ema.load_state_dict(G.state_dict())

    # Optimizers
    g_optim = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.0, 0.99))
    d_optim = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.99))

    # AMP scaler
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    ada = ADAugment(device=device, target=args.ada_target, initial_p=0.0, ada_speed=args.ada_speed, grayscale=args.grayscale)

    pl_mean = torch.zeros(1).to(device)
    global_step = 0
    sample_z = torch.randn(args.n_sample, args.z_dim)

    # Resume checkpoint
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        G_ema.load_state_dict(checkpoint.get('G_ema', checkpoint['G']))
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from {args.resume} starting at step {start_step}")

    print("Starting training")
    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for real in pbar:
            real = real.to(device)
            bs = real.size(0)
            global_step += 1

            # -------------------------
            # Discriminator step
            # -------------------------
            z = torch.randn(bs, args.z_dim, device=device)
            with autocast():  # FP16 forward
                fake = G(z).detach()
                real_for_d = ada.augment(real)
                d_real = D(real_for_d)
                d_fake = D(fake)
                loss_d = d_logistic_loss(d_real, d_fake)

            D.zero_grad()
            scaler_d.scale(loss_d).backward()
            scaler_d.step(d_optim)
            scaler_d.update()

            # R1 regularization
            if global_step % args.r1_interval == 0:
                real.requires_grad = True
                real_for_d2 = ada.augment(real)
                with autocast():
                    d_real2 = D(real_for_d2)
                    r1 = r1_penalty(d_real2, real)
                D.zero_grad()
                scaler_d.scale(args.r1_gamma * 0.5 * r1).backward()
                scaler_d.step(d_optim)
                scaler_d.update()

            # ADA update
            with torch.no_grad():
                signs = (d_real > 0).float().sum().item()
                ada.update(signs, bs)

            # -------------------------
            # Generator step
            # -------------------------
            z = torch.randn(bs, args.z_dim, device=device)
            with autocast():
                fake = G(z)
                d_fake_for_g = D(fake)
                loss_g = g_nonsaturating_loss(d_fake_for_g)

            G.zero_grad()
            scaler_g.scale(loss_g).backward()
            scaler_g.step(g_optim)
            scaler_g.update()

            # Path-length regularization
            if global_step % args.pl_interval == 0:
                pl_z = torch.randn(args.pl_batch, args.z_dim, device=device)
                pl_z.requires_grad = True
                with autocast():
                    pl_fake = G(pl_z)
                    pl_penalty, pl_mean_new = path_length_regularization(pl_fake, pl_z, pl_mean, decay=args.pl_decay)
                    pl_mean = torch.tensor(pl_mean_new, device=device)
                G.zero_grad()
                scaler_g.scale(args.pl_weight * pl_penalty).backward()
                scaler_g.step(g_optim)
                scaler_g.update()

            # EMA
            ema_decay = 0.5 ** (bs / (args.ema_kimg * 1000.0))
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.data = ema_decay * p_ema.data + (1 - ema_decay) * p.data

            # Logging
            if global_step % args.print_every == 0:
                pbar.set_postfix({
                    "loss_d": f"{loss_d.item():.4f}",
                    "loss_g": f"{loss_g.item():.4f}",
                    "ada_p": f"{ada.p:.4f}"
                })

            if global_step % args.sample_every == 0:
                sample_and_save(G_ema, sample_z, args.outdir, global_step, device, grayscale=args.grayscale)

            if global_step % args.save_every == 0:
                state = {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'g_optim': g_optim.state_dict(),
                    'd_optim': d_optim.state_dict(),
                    'G_ema': G_ema.state_dict(),
                    'step': global_step
                }
                save_checkpoint(state, args.outdir, global_step)

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    # final save
    state = {
        'G': G.state_dict(),
        'D': D.state_dict(),
        'g_optim': g_optim.state_dict(),
        'd_optim': d_optim.state_dict(),
        'G_ema': G_ema.state_dict(),
        'step': global_step
    }
    save_checkpoint(state, args.outdir, global_step)
    print("Training complete")
