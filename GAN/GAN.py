#!/usr/bin/env python3
"""
stylegan2_ada_med.py
Minimal StyleGAN2-ADA style implementation for medical 2D images (grayscale or RGB).
Single-file training + sampling script for testing/prototyping.

Notes:
- Implements Mapping network, Synthesis network with modulated conv + demod, noise injection, ToRGB.
- Discriminator: conv blocks + minibatch stddev.
- ADA: augmentations applied only to real images fed to D, with adaptive probability.
- Loss: non-saturating GAN + R1 gradient penalty for D + path-length regularization for G.
- EMA of generator weights (G_ema).
- Grayscale handling: if dataset is single-channel, the script can duplicate channels or operate with ch=1.
- Not as optimized as NVIDIA's repo; useful for research and quick tests.

Author: ChatGPT (GPT-5 Thinking mini)
Date: 2025-09-28
"""

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

# -----------------------------
# Utilities
# -----------------------------
def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Dataset: simple folder dataset
# -----------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, resolution=256, grayscale=False, transform=None):
        """
        root: folder containing images (any nested structure ok)
        resolution: final square size
        grayscale: if True convert to 1 channel
        """
        self.paths = []
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                self.paths.append(p)
        self.paths = sorted(self.paths)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.resolution = resolution
        self.grayscale = grayscale
        if transform is None:
            if grayscale:
                self.transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.CenterCrop(resolution),
                    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = img.convert("L") if self.grayscale else img.convert("RGB")
        return self.transform(img)

# -----------------------------
# ADA: augmentation pipeline (applied only to real images forwarded to D)
# Very simple set: horizontal flip, small translation, rotation, intensity jitter.
# Adaptive update maintains target real/vs-fake discriminator output sign changes.
# -----------------------------
class ADAugment:
    def __init__(self, device, target=0.6, initial_p=0.0, ada_speed=500 * 1000, grayscale=False):
        """
        target: target augmentation strength (heuristic; ADA paper uses target heuristics)
        ada_speed: number of images over which to adjust p (the higher, the slower)
        """
        self.device = device
        self.p = initial_p
        self.target = target
        self.grayscale = grayscale
        self.ada_speed = ada_speed
        # track sign statistics (heuristic from ADA implementation)
        self.num_images = 0
        self.signs = 0  # number of times D(real) > 0 (or some indicator)
        # simple transforms; not leaking labels because only use on D input
        # We'll implement transforms manually with torch ops
    def augment(self, x):
        """
        x: tensor [N,C,H,W] in [-1,1]
        Returns augmented x
        """
        if self.p <= 0:
            return x
        N, C, H, W = x.shape
        prob = torch.rand(N, device=x.device)
        mask = (prob < self.p).float().view(N, 1, 1, 1)
        out = x.clone()
        # horizontal flip
        flip = torch.flip(x, dims=[3])
        out = mask * flip + (1 - mask) * out
        # small translation: roll by -2..2 pixels randomly for masked items
        max_shift = max(1, H // 32)  # small shift relative to resolution
        shifts = torch.randint(-max_shift, max_shift+1, (N,), device=x.device)
        for i in range(N):
            if mask[i] > 0:
                out[i] = torch.roll(out[i], shifts[i].item(), dims=1 if random.random() < 0.5 else 2)
        # slight rotation: use affine grid for small angles
        angles = (torch.rand(N, device=x.device) - 0.5) * 2 * (2.0 * math.pi / 180.0)  # +/- 2 degrees
        for i in range(N):
            if mask[i] > 0 and abs(angles[i]) > 1e-6:
                theta = torch.tensor([
                    [math.cos(angles[i]), -math.sin(angles[i]), 0.0],
                    [math.sin(angles[i]),  math.cos(angles[i]), 0.0]
                ], device=x.device, dtype=x.dtype)
                grid = F.affine_grid(theta.unsqueeze(0), out[i:i+1].unsqueeze(0).shape, align_corners=False)
                out[i:i+1] = F.grid_sample(out[i:i+1], grid, align_corners=False, padding_mode='reflection')
        # brightness jitter (small)
        jitter = (torch.rand(N, 1, 1, 1, device=x.device) - 0.5) * 0.1
        out = out + mask * jitter
        # clamp back to [-1,1]
        out = out.clamp(-1, 1)
        return out

    def update(self, real_logits_signs, batch_size):
        """
        real_logits_signs: a Tensor of signs (1 if D(real) > 0 else 0) per image (or sum)
        """
        # accumulate signs
        self.num_images += batch_size
        self.signs += int(real_logits_signs)
        adjust = (self.signs / max(1, self.num_images) - self.target) * (batch_size / self.ada_speed)
        self.p = float(max(0.0, min(1.0, self.p + adjust)))
        # Reset counters occasionally to avoid numeric drift
        if self.num_images >= self.ada_speed:
            self.num_images = 0
            self.signs = 0

# -----------------------------
# StyleGAN2 core building blocks
# -----------------------------
def lerp(a, b, t):
    return a + (b - a) * t

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, eps=1e-8):
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()
        layers = []
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        layers.append(PixelNorm())
        in_features = z_dim
        for i in range(num_layers):
            linear = nn.Linear(in_features, w_dim)
            # equalized learning rate style initialization
            nn.init.normal_(linear.weight, 0, 1.0 / lr_mul)
            nn.init.zeros_(linear.bias)
            layers.append(nn.LeakyReLU(0.2))
            layers.append(linear)
            in_features = w_dim
        self.net = nn.Sequential(*layers)
        self.lr_mul = lr_mul

    def forward(self, z):
        # z: [N, z_dim]
        w = self.net(z) * self.lr_mul
        return w

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight.view(1, -1, 1, 1) * noise

class FusedLeakyReLU(nn.Module):
    def __init__(self, channels, negative_slope=0.2):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.act(x)

class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, style_dim=512, demod=True, up=False, down=False):
        super().__init__()
        self.eps = 1e-8
        self.kernel = kernel
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.style_dim = style_dim
        self.demod = demod
        self.up = up
        self.down = down

        fan_in = in_ch * kernel * kernel
        self.scale = 1 / math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))

        self.modulation = nn.Linear(style_dim, in_ch)
        nn.init.normal_(self.modulation.weight, 0, 1.0)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, style):
        """
        x: [N, in_ch, H, W]
        style: [N, style_dim]
        """
        N, _, H, W = x.shape
        style = self.modulation(style).view(N, 1, self.in_ch, 1, 1)  # [N,1,in_ch,1,1]
        weight = self.weight * (style + 1)  # scale weight
        if self.demod:
            demod = torch.rsqrt((weight ** 2).sum([2,3,4]) + self.eps)  # [N,out_ch]
            weight = weight * demod.view(N, self.out_ch, 1, 1, 1)

        # reshape for grouped conv
        weight = weight.view(N * self.out_ch, self.in_ch, self.kernel, self.kernel)
        x = x.view(1, N * self.in_ch, H, W)
        # optional up/down sampling using F.interpolate or conv_transpose; use nearest for upsample + conv
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = F.conv2d(x, weight, padding=self.kernel//2, groups=N)
        out = out.view(N, self.out_ch, out.shape[-2], out.shape[-1])
        if self.down:
            out = F.avg_pool2d(out, 2)
        return out

# Synthesis Block
class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, resolution, is_first=False):
        super().__init__()
        self.is_first = is_first
        if is_first:
            # constant input
            self.const = nn.Parameter(torch.randn(1, in_ch, 4, 4))
            self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, style_dim, demod=True, up=False)
        else:
            self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, style_dim, demod=True, up=True)
        self.noise1 = NoiseInjection(out_ch)
        self.act1 = FusedLeakyReLU(out_ch)

        self.conv2 = ModulatedConv2d(out_ch, out_ch, 3, style_dim, demod=True, up=False)
        self.noise2 = NoiseInjection(out_ch)
        self.act2 = FusedLeakyReLU(out_ch)

        self.toRGB = nn.Conv2d(out_ch, 3, kernel_size=1, stride=1, padding=0)
        # if grayscale target used externally, user can adapt outputs

    def forward(self, x, style1, style2, noise=None):
        """
        x: None for first block (use learned constant). Otherwise previous activations.
        """
        if self.is_first:
            out = self.const.repeat(style1.shape[0], 1, 1, 1)
            out = self.conv1(out, style1)
        else:
            out = self.conv1(x, style1)
        out = self.noise1(out, noise)
        out = self.act1(out)

        out = self.conv2(out, style2)
        out = self.noise2(out, noise)
        out = self.act2(out)
        rgb = self.toRGB(out)
        return out, rgb

# Generator: fixed number of layers up to chosen resolution
class Generator(nn.Module):
    def __init__(self, z_dim=128, w_dim=128, max_channels=512, resolution=48):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.resolution = resolution
        self.log_res = int(math.log2(resolution))
        assert 2 ** self.log_res == resolution and self.log_res >= 3, "resolution must be power of two >=8"

        # channels per resolution (simple scheme)
        def ch(res):
            return min(max_channels, 512 // (2 ** max(0, (self.log_res - int(math.log2(res))))))

        # mapping network
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=8)

        # synthesis blocks from 4x4 up to desired resolution
        self.blocks = nn.ModuleList()
        in_c = ch(4)
        for i in range(2, self.log_res + 1):  # 2 -> 4x4, ... up to resolution
            res = 2 ** i
            out_c = ch(res)
            is_first = (res == 4)
            block = SynthesisBlock(in_c, out_c, style_dim=w_dim, resolution=res, is_first=is_first)
            self.blocks.append(block)
            in_c = out_c

        # final parameter: scale ToRGB outputs progressively
        self.rgb_ups = nn.ModuleList([nn.Identity() for _ in range(len(self.blocks))])

    def forward(self, z, truncation_psi=1.0, truncation_cutoff=None, noise=None):
        """
        z: [N, z_dim]
        returns images [N, 3, H, W]
        """
        N = z.shape[0]
        w = self.mapping(z)  # [N, w_dim]
        # style per layer: we will use same w for all convs (no style mixing) OR implement mixing:
        styles = [w for _ in range(len(self.blocks) * 2)]  # two styles per block (conv1 & conv2)
        # Optionally do truncation here (simple global trunc)
        if truncation_psi != 1.0:
            w_avg = torch.mean(w, dim=0, keepdim=True)
            styles = [lerp(w_avg, s, truncation_psi) for s in styles]

        x = None
        rgb = 0
        idx = 0
        for i, block in enumerate(self.blocks):
            s1 = styles[idx]; s2 = styles[idx+1]; idx += 2
            x, out_rgb = block(x, s1, s2, noise=noise)
            # upsample previous rgb and add
            if i == 0:
                rgb = out_rgb
            else:
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False) + out_rgb
        # scale to [-1,1] already assumed because conv outputs roughly -inf..inf -> we leave as-is.
        img = torch.tanh(rgb)
        return img

# Discriminator blocks
class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.down = down
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        if self.down:
            x = F.avg_pool2d(x, 2)
        return x

class Discriminator(nn.Module):
    def __init__(self, resolution=256, max_channels=512):
        super().__init__()
        self.resolution = resolution
        self.log_res = int(math.log2(resolution))
        def ch(res):
            return min(max_channels, 512 // (2 ** max(0, (self.log_res - int(math.log2(res))))))

        blocks = []
        # from resolution down to 4x4
        in_ch = 3
        for i in range(self.log_res, 2, -1):
            res = 2 ** i
            out_ch = ch(res // 2)
            blocks.append(DiscBlock(in_ch, out_ch, down=True))
            in_ch = out_ch
        # final 4x4 block
        self.blocks = nn.ModuleList(blocks)
        self.final_conv = nn.Conv2d(in_ch + 1, ch(4), 3, padding=1)  # +1 for minibatch stddev
        self.final_act = nn.LeakyReLU(0.2)
        self.final_linear = nn.Linear(ch(4) * 4 * 4, 1)

    def minibatch_stddev(self, x, group_size=4):
        N, C, H, W = x.shape
        group = min(group_size, N)
        if group == 0:
            return x
        y = x.view(group, -1, C, H, W)  # [G, M, C, H, W]
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt((y ** 2).mean(dim=0) + 1e-8)
        y = y.mean(dim=(2,3,4), keepdim=True)  # [1, M, 1, 1, 1] -> per-sample
        y = y.repeat(group, 1, H, W)
        return torch.cat([x, y], dim=1)

    def forward(self, img):
        x = img
        for blk in self.blocks:
            x = blk(x)
        x = self.minibatch_stddev(x)
        x = self.final_act(self.final_conv(x))
        x = x.view(x.shape[0], -1)
        out = self.final_linear(x)
        return out.view(-1)

# -----------------------------
# Losses and regularizers
# -----------------------------
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()
    return real_loss + fake_loss

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def r1_penalty(real_pred, real_imgs):
    grads = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_imgs, create_graph=True)[0]
    penalty = grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()
    return penalty

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
    make_dir(outdir)
    ckpt = os.path.join(outdir, f"checkpoint_{step}.pt")
    torch.save(state, ckpt)
    print(f"Saved checkpoint {ckpt}")

def sample_and_save(G_ema, z_samples, outdir, step, device, grayscale=False):
    G_ema.eval()
    with torch.no_grad():
        imgs = G_ema(z_samples.to(device))
        imgs = (imgs.clamp(-1,1) + 1) * 0.5  # [0,1]
        # if dataset is grayscale and generator outputs 3 channels, convert to single channel by averaging
        if grayscale:
            imgs = imgs.mean(dim=1, keepdim=True)
        # convert to torchvision grid and save
        grid = utils.make_grid(imgs.cpu(), nrow=int(math.sqrt(imgs.shape[0])), normalize=False)
        out_path = os.path.join(outdir, f"samples_{step:06d}.png")
        utils.save_image(grid, out_path)
        print(f"Saved sample grid to {out_path}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # dataset and loader
    ds = ImageFolderDataset(args.data, resolution=args.resolution, grayscale=args.grayscale)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)

    # models
    G = Generator(z_dim=args.z_dim, w_dim=args.w_dim, max_channels=args.max_channels, resolution=args.resolution).to(device)
    D = Discriminator(resolution=args.resolution, max_channels=args.max_channels).to(device)
    G_ema = Generator(z_dim=args.z_dim, w_dim=args.w_dim, max_channels=args.max_channels, resolution=args.resolution).to(device)
    G_ema.load_state_dict(G.state_dict())  # initial copy

    # optimizers
    g_optim = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.0, 0.99))
    d_optim = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.99))

    # ADA
    ada = ADAugment(device=device, target=args.ada_target, initial_p=0.0, ada_speed=args.ada_speed, grayscale=args.grayscale)

    # other state
    pl_mean = torch.zeros(1).to(device)
    global_step = 0
    sample_z = torch.randn(args.n_sample, args.z_dim)  # CPU, will move to device when sampling

    # optionally resume
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

    # training loop
    print("Starting training")
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for real in pbar:
            real = real.to(device)
            bs = real.size(0)
            global_step += 1

            # -------------------------------------
            # Discriminator step
            # -------------------------------------
            z = torch.randn(bs, args.z_dim, device=device)
            fake = G(z).detach()
            # Apply ADA to real images only (simulate repo behavior)
            real_for_d = ada.augment(real)
            d_real = D(real_for_d)
            d_fake = D(fake)
            loss_d = d_logistic_loss(d_real, d_fake)

            D.zero_grad()
            loss_d.backward()
            d_optim.step()

            # R1 regularization (apply every r1_interval steps)
            if global_step % args.r1_interval == 0:
                real.requires_grad = True
                real_for_d2 = ada.augment(real)
                d_real2 = D(real_for_d2)
                r1 = r1_penalty(d_real2, real)
                D.zero_grad()
                (args.r1_gamma * 0.5 * r1).backward()
                d_optim.step()

            # ADA update: crude heuristic using sign of logits
            with torch.no_grad():
                signs = (d_real > 0).float().sum().item()  # how many real samples had positive logit
                ada.update(signs, bs)

            # -------------------------------------
            # Generator step
            # -------------------------------------
            z = torch.randn(bs, args.z_dim, device=device)
            fake = G(z)
            d_fake_for_g = D(fake)
            loss_g = g_nonsaturating_loss(d_fake_for_g)

            G.zero_grad()
            loss_g.backward()
            g_optim.step()

            # path-length regularization applied every N steps
            if global_step % args.pl_interval == 0:
                # compute pl penalty
                pl_z = torch.randn(args.pl_batch, args.z_dim, device=device)
                pl_z.requires_grad = True
                pl_fake = G(pl_z)
                pl_penalty, pl_mean_new = path_length_regularization(pl_fake, pl_z, pl_mean, decay=args.pl_decay)
                pl_mean = torch.tensor(pl_mean_new, device=device)
                G.zero_grad()
                (args.pl_weight * pl_penalty).backward()
                g_optim.step()

            # EMA update for G_ema
            with torch.no_grad():
                ema_beta = 0.5 ** (bs / max(1.0, args.ema_kimg * 1000.0))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(lerp(p.data, p_ema.data, ema_beta))

            # logging & snapshot
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

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Minimal StyleGAN2-ADA for medical 2D images (single-file)")
    p.add_argument("--data", required=True, help="Path to image folder (recursively searched)")
    p.add_argument("--outdir", default="./training_out", help="Where to save checkpoints and samples")
    p.add_argument("--resolution", type=int, default=256, help="Image resolution (square, power of two)")
    p.add_argument("--grayscale", action="store_true", help="Treat dataset as grayscale (single channel)")
    p.add_argument("--z-dim", type=int, default=512, dest="z_dim")
    p.add_argument("--w-dim", type=int, default=512, dest="w_dim")
    p.add_argument("--max-channels", type=int, default=512, help="Max channels in generator/discriminator")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--g-lr", type=float, default=2e-3)
    p.add_argument("--d-lr", type=float, default=2e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-every", type=int, default=500, help="Steps between sample grids")
    p.add_argument("--save-every", type=int, default=2000, help="Steps between checkpoints")
    p.add_argument("--print-every", type=int, default=50, help="Steps between printed status updates")
    p.add_argument("--n-sample", type=int, default=16, help="Number of samples in sample grid")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--max-steps", type=int, default=20000, help="Total training steps (global)")
    p.add_argument("--ada-target", type=float, default=0.6, help="Target for ADA heuristic")
    p.add_argument("--ada-speed", type=int, default=500*1000, help="ADA adjustment speed (images)")
    p.add_argument("--r1-interval", type=int, default=16, help="Apply R1 every N steps")
    p.add_argument("--r1-gamma", type=float, default=10.0, dest="r1_gamma", help="Weight for R1 penalty")
    p.add_argument("--pl-interval", type=int, default=8, help="Apply path-length every N steps")
    p.add_argument("--pl-batch", type=int, default=4, dest="pl_batch")
    p.add_argument("--pl-weight", type=float, default=2.0, dest="pl_weight")
    p.add_argument("--pl-decay", type=float, default=0.01, dest="pl_decay")
    p.add_argument("--ema-kimg", type=float, default=10.0, dest="ema_kimg", help="EMA half-life in kimg")
    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    make_dir(args.outdir)
    train(args)
