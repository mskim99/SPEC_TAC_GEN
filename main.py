import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서도 안전

import sys
import os
import glob
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


def setup_logger(log_path):
    """print 출력을 터미널과 파일에 동시에 저장"""
    class Logger(object):
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout  # 에러도 동일하게 기록


def log_training_params(log_path, args):
    """학습 시작 시 파라미터 기록"""
    with open(log_path, "a") as f:
        f.write(f"\n==== New Run: {datetime.now()} ====\n")
        f.write("Training Parameters:\n")
        f.write(f"  is_phase     : {args.is_phase}\n")
        f.write(f"  data_dir     : {args.data_dir}\n")
        f.write(f"  sample_dir   : {args.sample_dir}\n")
        f.write(f"  ckpt_dir     : {args.ckpt_dir}\n")
        f.write(f"  epochs       : {args.epochs}\n")
        f.write(f"  batch_size   : {args.batch_size}\n")
        f.write(f"  lr           : {args.lr}\n")
        f.write(f"  timesteps    : {args.timesteps}\n")
        f.write(f"  base_ch      : {args.base_ch}\n")
        f.write(f"  save_interval: {args.save_interval}\n")
        f.write(f"  num_samples  : {args.num_samples}\n\n")


def log_loss(epoch, loss_value):
    """학습 중 loss 값 기록 (print와 함께 log.txt에도 저장됨)"""
    print(f"[Epoch {epoch}] - Loss: {loss_value:.6f}")


# =====================
# Dataset 정의
# =====================
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, is_phase=False, min_db=-80, max_db=0):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.is_phase = is_phase
        self.min_db = min_db
        self.max_db = max_db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # magnitude 값 (129,376)

        if self.is_phase:
            spec_norm = spec / math.pi
        else:
            # magnitude → dB
            spec_db = 20*np.log10(spec + 1e-8)

            # (1) dB 범위 클리핑
            spec_db = np.clip(spec_db, self.min_db, self.max_db)

            # (2) [-1,1] normalize
            spec_norm = (spec_db - self.min_db) / (self.max_db - self.min_db) * 2 - 1

        spec_norm = torch.tensor(spec_norm, dtype=torch.float32)

        return spec_norm.unsqueeze(0)  # (1,129,376)


# 기본 Conv Block
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


# UNet 구조
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)       # (B, 32, 129, 376)
        self.pool1 = nn.MaxPool2d(2)                # (B, 32, 64, 188)

        self.enc2 = ConvBlock(base_ch, base_ch*2)   # (B, 64, 64, 188)
        self.pool2 = nn.MaxPool2d(2)                # (B, 64, 32, 94)

        self.enc3 = ConvBlock(base_ch*2, base_ch*4) # (B,128, 32, 94)
        self.pool3 = nn.MaxPool2d(2)                # (B,128, 16, 47)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)  # (B,256,16,47)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)  # skip 연결

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        # 출력
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x, t=None):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=e3.shape[2:], mode="bilinear", align_corners=False)  # 크기 보정
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return out


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
        """
        x0: 원본 데이터
        t: timestep
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise


# =====================
# 학습 루프
# =====================
def train_diffusion(is_phase,
                    data_dir,
                    epochs=10,
                    batch_size=16,
                    lr=1e-4,
                    timesteps=300,
                    base_ch=32,
                    ckpt_dir="checkpoints",
                    device="cuda",
                    save_interval=5):
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = SpectrogramDataset(data_dir, is_phase=is_phase)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(in_ch=1, out_ch=1, base_ch=base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    diffusion = Diffusion(timesteps=timesteps, device=device)

    for epoch in range(epochs):
        loss = 0.
        for step, x in enumerate(dataloader):
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()

            noisy_x, noise = diffusion.add_noise(x, t)
            noise_pred = model(noisy_x, t)

            loss = mse(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # === loss 기록 ===
        log_loss(epoch+1, loss.item())

        # print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

        # Save checkpoints
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
# 실행 예시
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_phase", type=bool, default=False)
    parser.add_argument("--data_dir", type=str, default="data/magnitude/",
                        help="데이터셋 npy 폴더 경로")
    parser.add_argument("--sample_dir", type=str, default="samples/magnitude/",
                        help="생성 샘플 저장 폴더")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/magnitude/",
                        help="checkpoint 저장 폴더")
    parser.add_argument("--epochs", type=int, default=50, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--timesteps", type=int, default=300, help="Diffusion 타임스텝 수")
    parser.add_argument("--base_ch", type=int, default=32, help="UNet base channel 수")
    parser.add_argument("--save_interval", type=int, default=10, help="체크포인트 저장 주기")
    parser.add_argument("--num_samples", type=int, default=16, help="생성할 샘플 개수")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # === 로그 저장 세팅 ===
    log_path = os.path.join(args.sample_dir, "log.txt")
    os.makedirs(args.sample_dir, exist_ok=True)
    from datetime import datetime
    with open(log_path, "a") as f:
        f.write(f"\n==== New Run: {datetime.now()} ====\n")

    # === 로그 저장 세팅 ===
    log_path = os.path.join(args.sample_dir, "log.txt")
    from datetime import datetime

    # 파라미터 기록
    log_training_params(log_path, args)

    # print -> log.txt 동시 저장
    setup_logger(log_path)

    model, diffusion = train_diffusion(
        is_phase=args.is_phase,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        save_interval=args.save_interval,
        device=device
    )