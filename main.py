import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
from model import UNet

matplotlib.use("Agg")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use

# ===============================
# Utility
# ===============================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected boolean value.")


def compute_db_stats(files, min_db, max_db):
    """전체 magnitude 데이터의 평균, 표준편차를 계산"""
    vals = []
    for f in files:
        x = np.load(f)
        db = 20 * np.log10(x + 1e-8)
        db = np.clip(db, min_db, max_db)
        vals.append(db)
    vals = np.stack(vals, axis=0)
    mu = float(np.mean(vals))
    sigma = float(np.std(vals) + 1e-12)
    return mu, sigma


# ===============================
# Dataset
# ===============================
class SpectrogramDataset(Dataset):
    """
    is_phase=True인 경우 STFT의 'phase residual'을 학습
    r_t(k) = princ(Δφ_t(k) - 2π·hop·k/n_fft)
    """

    def __init__(self, data_dir, type="None",
                 min_db=-80, max_db=0, conditional=False,
                 norm_type="zscore", n_fft=256, hop_length=128,
                 db_mu=None, db_sigma=None):

        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.type = type
        self.min_db, self.max_db = min_db, max_db
        self.conditional = conditional
        self.norm_type = norm_type
        self.n_fft = n_fft
        self.hop = hop_length

        if self.conditional:
            self.conditions = [os.path.basename(f).split("_")[0] for f in self.files]
            uniq = sorted(set(self.conditions))
            self.cond2idx = {c: i for i, c in enumerate(uniq)}

        # ---------- 정규화 설정 ----------
        if self.norm_type == "zscore":
            if db_mu is None or db_sigma is None:
                self.db_mu, self.db_sigma = compute_db_stats(self.files, self.min_db, self.max_db)
            else:
                self.db_mu, self.db_sigma = db_mu, db_sigma

        elif self.norm_type in ["0to1", "-1to1"]:
            self.db_mu, self.db_sigma = self.min_db, self.max_db

        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])

        # ---------- Phase residual ----------
        if self.type == "phase":
            phase = arr  # (F, T)
            F, T = phase.shape
            k = np.arange(F)
            expected = 2 * np.pi * self.hop * k / float(self.n_fft)
            dphi = phase[:, 1:] - phase[:, :-1]
            residual = np.angle(np.exp(1j * (dphi - expected[:, None])))  # wrap [-π, π]
            spec = np.stack([np.sin(residual), np.cos(residual)], axis=0)  # (2, F, T-1)

        # ---------- Magnitude ----------
        elif self.type == "magnitude":
            db = 20 * np.log10(arr + 1e-8)
            db = np.clip(db, self.min_db, self.max_db)
            spec = self._apply_normalization(db)
            spec = np.expand_dims(spec, axis=0)

        # ---------- None (raw signal) ----------
        elif self.type in ["None", "none", None]:
            # Raw 입력값에도 동일한 norm_type 로직 적용
            raw = arr.astype(np.float32)
            if self.norm_type == "zscore":
                mean = np.mean(raw)
                std = np.std(raw) + 1e-12
                spec = (raw - mean) / std
            elif self.norm_type == "0to1":
                min_v, max_v = raw.min(), raw.max()
                spec = (raw - min_v) / (max_v - min_v + 1e-12)
            elif self.norm_type == "-1to1":
                min_v, max_v = raw.min(), raw.max()
                spec = (raw - min_v) / (max_v - min_v + 1e-12)
                spec = spec * 2 - 1
            else:
                raise ValueError(f"Unsupported normalization type: {self.norm_type}")

            spec = np.expand_dims(spec, axis=0)

        else:
            raise ValueError(f"Unsupported dataset type: {self.type}")

        spec = torch.tensor(spec, dtype=torch.float32)

        if self.conditional:
            cond_idx = self.cond2idx[self.conditions[idx]]
            return spec, cond_idx
        return spec

    # ---------- 내부 정규화 함수 ----------
    def _apply_normalization(self, db):
        """Magnitude 및 기타 입력에서 동일하게 사용"""
        if self.norm_type == "zscore":
            return (db - self.db_mu) / self.db_sigma
        elif self.norm_type == "0to1":
            return (db - self.min_db) / (self.max_db - self.min_db + 1e-12)
        elif self.norm_type == "-1to1":
            spec = (db - self.min_db) / (self.max_db - self.min_db + 1e-12)
            return spec * 2 - 1
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")


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


# ===============================
# Train
# ===============================
def train_diffusion(type, conditional, data_dir,
                    epochs=10, batch_size=16, lr=1e-4,
                    timesteps=300, base_ch=32,
                    ckpt_dir="checkpoints", device="cuda",
                    save_interval=5, norm_type="zscore",
                    min_db=-80, max_db=0,
                    n_fft=256, hop_length=128):

    os.makedirs(ckpt_dir, exist_ok=True)
    dataset = SpectrogramDataset(data_dir, type, min_db, max_db,
                                 conditional, norm_type, n_fft, hop_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.cond2idx) if conditional else 0
    if conditional:
        with open(os.path.join(ckpt_dir, "cond2idx.json"), "w") as f:
            json.dump(dataset.cond2idx, f, indent=4)

    in_ch = out_ch = 2 if type == "phase" else 1
    model = UNet(in_ch, out_ch, base_ch, [1, 2, 4, 8], conditional, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(timesteps, device)

    for epoch in range(epochs):
        for batch in dataloader:
            if conditional:
                x, cond = batch
                x = x.to(device)
                cond = cond.to(device).long()
            else:
                x = batch.to(device)
                cond = None

            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            noisy, noise = diffusion.add_noise(x, t)
            pred = model(noisy, t, cond) if conditional else model(noisy, t)
            loss = mse(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[{epoch+1}/{epochs}] Loss = {loss.item():.6f}")

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
                "type": type,
                "phase_mode": "residual",
                "n_fft": n_fft,
                "hop_length": hop_length,
                "min_db": min_db,
                "max_db": max_db,
                "db_mu": dataset.db_mu,
                "db_sigma": dataset.db_sigma
            }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt"))

    return model, diffusion

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="None")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--norm_type", type=str, choices=["zscore", "0to1", "-1to1"], default="zscore")
    parser.add_argument("--min_db", type=float, default=-80.0)
    parser.add_argument("--max_db", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model, diffusion = train_diffusion(
        type=args.type,
        conditional=args.conditional,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        ckpt_dir=args.ckpt_dir,
        device=device,
        save_interval=args.save_interval,
        norm_type=args.norm_type,
        min_db=args.min_db,
        max_db=args.max_db
    )