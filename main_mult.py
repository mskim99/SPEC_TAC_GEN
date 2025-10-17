import os
import glob
import json
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
    """
    def __init__(self, real_dir, imag_dir, norm_type="tanh"):
        self.real_files = sorted(glob.glob(os.path.join(real_dir, "*.npy")))
        self.imag_files = sorted(glob.glob(os.path.join(imag_dir, "*.npy")))
        self.norm_type = norm_type
        assert len(self.real_files) == len(self.imag_files), "real/imag 파일 개수가 일치해야 함"

    def __len__(self):
        return len(self.real_files)

    def _normalize(self, arr):
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
# Frequency-aware Loss
# ===============================
class FrequencyAwareLoss(nn.Module):
    def __init__(self, weight_low=1.0, weight_high=2.0):
        super().__init__()
        self.weight_low = weight_low
        self.weight_high = weight_high

    def forward(self, pred, target):
        # 주파수 축: scale (height)
        B, C, H, W = pred.shape
        freqs = torch.linspace(0, 1, H, device=pred.device).view(1, 1, H, 1)
        freq_weight = self.weight_low + (self.weight_high - self.weight_low) * freqs
        return torch.mean(freq_weight * (pred - target) ** 2)


# ===============================
# Phase Consistency Loss
# ===============================
def phase_consistency_loss(pred, target):
    """
    Real–Imag 위상 일관성 유지용
    """
    pred_real, pred_imag = pred[:, 0], pred[:, 1]
    gt_real, gt_imag = target[:, 0], target[:, 1]
    dot = pred_real * gt_real + pred_imag * gt_imag
    norm_p = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-12)
    norm_t = torch.sqrt(gt_real**2 + gt_imag**2 + 1e-12)
    cos_sim = dot / (norm_p * norm_t)
    return 1 - torch.mean(cos_sim)


# ===============================
# Diffusion
# ===============================
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
        sa = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        so = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        return sa * x0 + so * noise, noise


def train_diffusion_complex(real_dir, imag_dir, epochs=50, batch_size=4, lr=1e-4,
                            timesteps=300, base_ch=32, ckpt_dir="checkpoints_complex",
                            device="cuda", save_interval=10, norm_type="tanh",
                            lambda_freq=0.5, lambda_phase=0.3,
                            resume_ckpt=None):
    """
    resume_ckpt: checkpoint 파일 경로 (있을 경우 이어서 학습)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    dataset = ComplexWaveletDataset(real_dir, imag_dir, norm_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    in_ch = out_ch = 2
    model = UNet(in_ch, out_ch, base_ch, [1, 2, 4, 8]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    freq_loss_fn = FrequencyAwareLoss()
    diffusion = Diffusion(timesteps, device)

    start_epoch = 0

    # ✅ Checkpoint resume 기능
    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"✅ Checkpoint loaded from {resume_ckpt} (start from epoch {start_epoch})")

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
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
            optimizer.step()
            total_loss += loss.item()

        print(f"[{epoch+1}/{epochs}] Total Loss: {total_loss/len(dataloader):.6f}")

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": total_loss/len(dataloader),
                "lambda_freq": lambda_freq,
                "lambda_phase": lambda_phase,
                "norm_type": norm_type,
                "type": "complex_wavelet"
            }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt"))

    return model, diffusion


# ===============================
# Main (with checkpoint index)
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="Real coefficient npy 폴더")
    parser.add_argument("--imag_dir", type=str, required=True, help="Imag coefficient npy 폴더")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint 폴더")
    parser.add_argument("--ckpt_index", type=int, default=None, help="불러올 checkpoint index (예: 100 → ckpt_epoch_0100.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--norm_type", type=str, default="tanh", choices=["tanh", "zscore", "0to1"])
    parser.add_argument("--lambda_freq", type=float, default=0.5)
    parser.add_argument("--lambda_phase", type=float, default=0.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ checkpoint index 기반 로드 기능 추가
    ckpt_path = None
    if args.ckpt_index is not None:
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_epoch_{args.ckpt_index:04d}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ 지정된 checkpoint 파일이 존재하지 않습니다: {ckpt_path}")
        print(f"✅ Checkpoint {ckpt_path} 로드 중...")

        # checkpoint 로드 후 이어서 학습 (선택적으로)
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"✅ 로드 완료 (epoch={ckpt.get('epoch', 'unknown')}, loss={ckpt.get('loss', 0):.6f})")

    # 학습 실행 (checkpoint가 있으면 이어서 학습 가능)
    model, diffusion = train_diffusion_complex(
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
        resume_ckpt=ckpt_path
    )