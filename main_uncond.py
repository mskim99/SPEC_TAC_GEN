import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import SpectrogramDataset   # ✅ dataset.py 불러오기


# =====================
# UNet 모델
# =====================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x, t=None):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out_conv(d1)


# =====================
# Diffusion Utilities
# =====================
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


class Diffusion:
    def __init__(self, timesteps=300, device="cuda"):
        self.device = device
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise


# =====================
# 학습 루프
# =====================
def train_diffusion(data_dir, is_phase=False, epochs=10, batch_size=16, lr=1e-4,
                    timesteps=300, base_ch=32, device="cuda",
                    ckpt_dir="checkpoints", save_interval=5):

    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = SpectrogramDataset(data_dir, is_phase=is_phase)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_ch = 2 if is_phase else 1
    out_ch = 2 if is_phase else 1
    model = UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    diffusion = Diffusion(timesteps=timesteps, device=device)

    for epoch in range(epochs):
        for step, x in enumerate(dataloader):
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()

            noisy_x, noise = diffusion.add_noise(x, t)
            noise_pred = model(noisy_x, t)

            loss = mse(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    return model, diffusion


# =====================
# 샘플 생성
# =====================
@torch.no_grad()
def sample(model, diffusion, shape=(1, 1, 129, 376), device="cuda",
           min_db=-80, max_db=0, is_phase=False):
    img = torch.randn(shape, device=device)

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        noise_pred = model(img, t_tensor)
        alpha = diffusion.alphas[t]
        alpha_hat = diffusion.alpha_hat[t]
        beta = diffusion.betas[t]

        noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
        img = 1/torch.sqrt(alpha) * (img - ((1-alpha)/torch.sqrt(1-alpha_hat))*noise_pred) + torch.sqrt(beta)*noise

    img = img.squeeze().cpu().numpy()

    if is_phase:
        sin_out, cos_out = img[0], img[1]
        phase_recon = np.arctan2(sin_out, cos_out)
        return phase_recon
    else:
        spec_db = (img + 1) / 2 * (max_db - min_db) + min_db
        spec_mag = 10 ** (spec_db / 20)
        return spec_mag


# =====================
# 실행 예시
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_phase", type=bool, default=False)
    parser.add_argument("--data_dir", type=str, default="data/magnitude/")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/magnitude/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--base_ch", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model, diffusion = train_diffusion(
        data_dir=args.data_dir,
        is_phase=args.is_phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        device=device,
        ckpt_dir=args.ckpt_dir
    )

    # 샘플 생성
    gen = sample(model, diffusion,
                 shape=(1, 2, 129, 376) if args.is_phase else (1, 1, 129, 376),
                 device=device,
                 is_phase=args.is_phase)

    plt.imshow(gen, aspect="auto", origin="lower", cmap="jet")
    plt.colorbar(label="Phase" if args.is_phase else "Magnitude")
    plt.title("Generated Spectrogram")
    plt.savefig("generated_example.png", dpi=300)