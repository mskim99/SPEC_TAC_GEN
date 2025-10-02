import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서도 안전

import sys
import os
import glob
import argparse
import math
import numpy as np
import json
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

        # 파일명에서 condition 추출
        self.conditions = [os.path.basename(f).split("_")[0] for f in self.files]
        unique_conditions = sorted(set(self.conditions))
        self.cond2idx = {c: i for i, c in enumerate(unique_conditions)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # (129,376)

        if self.is_phase:
            spec_norm = spec / math.pi
        else:
            spec_db = 20*np.log10(spec + 1e-8)
            spec_db = np.clip(spec_db, self.min_db, self.max_db)
            spec_norm = (spec_db - self.min_db) / (self.max_db - self.min_db) * 2 - 1

        spec_norm = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
        cond_idx = self.cond2idx[self.conditions[idx]]  # int label

        return spec_norm, cond_idx


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


def sinusoidal_embedding(timesteps, dim=128):
    """
    Sinusoidal timestep embedding (Transformer/StableDiffusion 스타일)
    timesteps: (B,) int tensor
    return: (B, dim) tensor
    """
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # (B, dim)


# UNet 구조
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, num_classes=10, emb_dim=128):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

        # === Embeddings ===
        self.cond_emb = nn.Embedding(num_classes, emb_dim)   # condition embedding
        self.fc = nn.Linear(emb_dim*2, base_ch*8)            # timestep+condition → bottleneck 채널 매핑

    def forward(self, x, t=None, cond=None):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        # === timestep + condition embedding 결합 ===
        if t is not None and cond is not None:
            t_emb = sinusoidal_embedding(t, dim=self.cond_emb.embedding_dim)   # (B, emb_dim)
            c_emb = self.cond_emb(cond)                                        # (B, emb_dim)
            joint_emb = torch.cat([t_emb, c_emb], dim=1)                       # (B, emb_dim*2)
            joint_vec = self.fc(joint_emb)                                     # (B, base_ch*8)
            joint_vec = joint_vec[:, :, None, None]                            # (B, C, 1, 1)
            b = b + joint_vec

        # Decoder
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

    num_classes = len(dataset.cond2idx)

    # === cond2idx 매핑 저장 ===
    cond_map_path = os.path.join(ckpt_dir, "cond2idx.json")
    with open(cond_map_path, "w") as f:
        json.dump(dataset.cond2idx, f, indent=4)
    print(f"✅ Saved cond2idx mapping to {cond_map_path}")

    model = UNet(in_ch=1, out_ch=1, base_ch=base_ch, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    diffusion = Diffusion(timesteps=timesteps, device=device)

    for epoch in range(epochs):
        for step, (x, cond) in enumerate(dataloader):
            x, cond = x.to(device), cond.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()

            noisy_x, noise = diffusion.add_noise(x, t)
            noise_pred = model(noisy_x, t, cond)  # cond 추가

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