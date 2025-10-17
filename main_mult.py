import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import UNet
import matplotlib
matplotlib.use("Agg")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ===============================
# Dataset
# ===============================
class ComplexWaveletDataset(Dataset):
    """
    Real + Imag 2채널 Wavelet coefficient 학습용
    (입력 clamp + tanh 정규화)
    """
    def __init__(self, real_dir, imag_dir, norm_type="tanh"):
        self.real_files = sorted(glob.glob(os.path.join(real_dir, "*.npy")))
        self.imag_files = sorted(glob.glob(os.path.join(imag_dir, "*.npy")))
        self.norm_type = norm_type
        assert len(self.real_files) == len(self.imag_files), "real/imag 파일 개수가 일치해야 함"

    def __len__(self):
        return len(self.real_files)

    def _normalize(self, arr):
        arr = np.clip(arr, -1.0, 1.0)  # ✅ Clamp to [-1,1]
        if self.norm_type == "zscore":
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-12)
        elif self.norm_type == "tanh":
            return np.tanh(arr)
        elif self.norm_type == "0to1":
            arr_min, arr_max = arr.min(), arr.max()
            return (arr - arr_min) / (arr_max - arr_min + 1e-12)
        else:
            return arr

    def __getitem__(self, idx):
        real = np.load(self.real_files[idx]).astype(np.float32)
        imag = np.load(self.imag_files[idx]).astype(np.float32)
        real = self._normalize(real)
        imag = self._normalize(imag)
        spec = np.stack([real, imag], axis=0)  # (2, H, W)
        return torch.tensor(spec, dtype=torch.float32)


# ===============================
# Cosine Beta Schedule (Improved)
# ===============================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)


class Diffusion:
    def __init__(self, timesteps=500, device="cuda"):
        self.device = device
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        sa = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        so = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        return sa * x0 + so * noise, noise


# ===============================
# Frequency-aware Loss (강화)
# ===============================
class FrequencyAwareLoss(nn.Module):
    def __init__(self, weight_low=1.0, weight_high=3.0):
        super().__init__()
        self.weight_low = weight_low
        self.weight_high = weight_high

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        freqs = torch.linspace(0, 1, H, device=pred.device).view(1, 1, H, 1)
        freq_weight = self.weight_low + (self.weight_high - self.weight_low) * freqs
        return torch.mean(freq_weight * (pred - target) ** 2)


# ===============================
# Phase Consistency Loss (개선)
# ===============================
def phase_consistency_loss(pred, target):
    pred_real, pred_imag = pred[:, 0], pred[:, 1]
    gt_real, gt_imag = target[:, 0], target[:, 1]

    # 벡터 내적 기반 cosine similarity
    dot = (pred_real * gt_real + pred_imag * gt_imag)
    norm_pred = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-12)
    norm_gt = torch.sqrt(gt_real**2 + gt_imag**2 + 1e-12)
    cos_sim = dot / (norm_pred * norm_gt + 1e-12)

    return 1 - torch.mean(cos_sim)  # ✅ 1 - mean(cosine similarity)


# ===============================
# Train Function
# ===============================
def train_diffusion_complex(real_dir, imag_dir, epochs=200, batch_size=8, lr=1e-4,
                            timesteps=500, base_ch=64, ckpt_dir="checkpoints_complex_v2",
                            device="cuda", save_interval=20, norm_type="tanh",
                            lambda_freq=1.0, lambda_phase=1.0,
                            resume_ckpt=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    dataset = ComplexWaveletDataset(real_dir, imag_dir, norm_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    in_ch = out_ch = 2
    model = UNet(in_ch, out_ch, base_ch, [1, 2, 4, 8]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    mse = nn.MSELoss()
    freq_loss_fn = FrequencyAwareLoss()
    diffusion = Diffusion(timesteps, device)

    start_epoch = 0

    # ✅ Resume 기능
    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"✅ Checkpoint loaded from {resume_ckpt} (epoch {start_epoch})")

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        model.train()

        for batch in dataloader:
            x = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            noisy, noise = diffusion.add_noise(x, t)
            pred = model(noisy, t)

            # 손실 계산
            loss_mse = mse(pred, noise)
            loss_freq = freq_loss_fn(pred, noise)
            loss_phase = phase_consistency_loss(pred, noise)
            loss = loss_mse + lambda_freq * loss_freq + lambda_phase * loss_phase

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ✅ Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"[{epoch+1}/{epochs}] Loss={avg_loss:.6f} | LR={scheduler.get_last_lr()[0]:.6e}")

        # ✅ Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "lambda_freq": lambda_freq,
                "lambda_phase": lambda_phase,
                "norm_type": norm_type,
                "type": "complex_wavelet_v2"
            }, ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")

    print("🎯 Training Complete!")
    return model, diffusion


# ===============================
# Main Entry
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--imag_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_complex_v2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--norm_type", type=str, default="tanh", choices=["tanh", "zscore", "0to1"])
    parser.add_argument("--lambda_freq", type=float, default=1.0)
    parser.add_argument("--lambda_phase", type=float, default=1.0)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_diffusion_complex(
        real_dir=args.real_dir,
        imag_dir=args.imag_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        ckpt_dir=args.ckpt_dir,
        device=device,
        save_interval=args.save_interval,
        norm_type=args.norm_type,
        lambda_freq=args.lambda_freq,
        lambda_phase=args.lambda_phase,
        resume_ckpt=args.resume_ckpt
    )