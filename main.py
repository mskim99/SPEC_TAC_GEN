import os, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
from datetime import datetime

# --------------------
# Utils
# --------------------
def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("Expected boolean value.")

def compute_db_stats(files, min_db, max_db):
    vals = []
    for f in files:
        x = np.load(f)  # magnitude
        db = 20*np.log10(x + 1e-8)
        db = np.clip(db, min_db, max_db)
        vals.append(db)
    vals = np.stack(vals, axis=0)
    mu = float(vals.mean())
    sigma = float(vals.std() + 1e-12)
    return mu, sigma

# --------------------
# Dataset
# --------------------
class SpectrogramDataset(Dataset):
    """
    - is_phase=False: magnitude dB를 z-score(default)로 표준화 → (1, 129, 376)
    - is_phase=True : phase 절대값 φ 대신 Δφ=diff(φ, axis=1)을 sin/cos로 → (2, 129, 375)
    - conditional=True: 파일명 프리픽스(첫 '_' 앞)를 condition으로 사용
    """
    def __init__(self, data_dir, is_phase=False, min_db=-80, max_db=0,
                 conditional=False, norm_type="zscore", db_mu=None, db_sigma=None):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.is_phase = is_phase
        self.min_db, self.max_db = min_db, max_db
        self.conditional = conditional
        self.norm_type = norm_type

        if self.conditional:
            self.conditions = [os.path.basename(f).split("_")[0] for f in self.files]
            uniq = sorted(set(self.conditions))
            self.cond2idx = {c:i for i,c in enumerate(uniq)}

        # magnitude의 zscore용 통계
        if not self.is_phase and self.norm_type == "zscore":
            if db_mu is None or db_sigma is None:
                self.db_mu, self.db_sigma = compute_db_stats(self.files, self.min_db, self.max_db)
            else:
                self.db_mu, self.db_sigma = db_mu, db_sigma
        else:
            self.db_mu, self.db_sigma = None, None

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])

        if self.is_phase:
            # arr: absolute phase φ ∈ [-π, π], shape (129, 376)
            phase = arr
            # Δφ along time axis (width-1)
            phase_diff = np.diff(phase, axis=1)         # (129, 375)
            # wrap to [-π, π]
            phase_diff = np.angle(np.exp(1j * phase_diff))
            spec = np.stack([np.sin(phase_diff), np.cos(phase_diff)], axis=0)  # (2, 129, 375)
        else:
            # magnitude → dB → 정규화
            db = 20*np.log10(arr + 1e-8)
            db = np.clip(db, self.min_db, self.max_db)
            if self.norm_type == "zscore":
                spec = (db - self.db_mu) / self.db_sigma
            else:
                spec01 = (db - self.min_db) / (self.max_db - self.min_db + 1e-12)
                spec = spec01 if self.norm_type == "0to1" else (spec01 * 2 - 1)
            spec = np.expand_dims(spec, axis=0)  # (1, 129, 376)

        spec = torch.tensor(spec, dtype=torch.float32)

        if self.conditional:
            cond_idx = self.cond2idx[self.conditions[idx]]
            return spec, cond_idx
        return spec

# --------------------
# Model
# --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

def sinusoidal_embedding(t, dim=128):
    device = t.device
    half = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, conditional=False, num_classes=0, emb_dim=128):
        super().__init__()
        self.conditional = conditional
        self.enc1 = ConvBlock(in_ch, base_ch);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2); self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2); self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2);   self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)
        if self.conditional:
            self.cond_emb = nn.Embedding(num_classes, emb_dim)
            self.fc = nn.Linear(emb_dim*2, base_ch*8)

    def forward(self, x, t=None, cond=None):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b  = self.bottleneck(p3)

        if self.conditional and t is not None and cond is not None:
            t_emb = sinusoidal_embedding(t, dim=self.cond_emb.embedding_dim)
            c_emb = self.cond_emb(cond)
            joint = torch.cat([t_emb, c_emb], dim=1)
            b = b + self.fc(joint)[:, :, None, None]

        u3 = self.up3(b); u3 = F.interpolate(u3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], 1))
        u2 = self.up2(d3); u2 = F.interpolate(u2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], 1))
        u1 = self.up1(d2); u1 = F.interpolate(u1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], 1))
        return self.out_conv(d1)

# --------------------
# Diffusion
# --------------------
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

# --------------------
# Train
# --------------------
def train_diffusion(is_phase, conditional, data_dir, epochs=10, batch_size=16, lr=1e-4,
                    timesteps=300, base_ch=32, ckpt_dir="checkpoints",
                    device="cuda", save_interval=5, norm_type="zscore",
                    min_db=-80, max_db=0):
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = SpectrogramDataset(data_dir, is_phase, min_db, max_db, conditional, norm_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.cond2idx) if conditional else 0
    if conditional:
        with open(os.path.join(ckpt_dir, "cond2idx.json"), "w") as f:
            json.dump(dataset.cond2idx, f, indent=4)

    # 채널 및 해상도
    if is_phase:
        in_ch = out_ch = 2
    else:
        in_ch = out_ch = 1

    model = UNet(in_ch, out_ch, base_ch, conditional, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(timesteps, device)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if conditional:
                x, cond = batch
                x = x.to(device); cond = cond.to(device).long()
            else:
                x = batch.to(device); cond = None

            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device).long()
            noisy, noise = diffusion.add_noise(x, t)
            pred = model(noisy, t, cond) if conditional else model(noisy, t)
            loss = mse(pred, noise)

            optim.zero_grad(); loss.backward(); optim.step()

        print(f"[{epoch+1}/{epochs}] loss={loss.item():.6f}")

        if (epoch+1) % save_interval == 0 or (epoch+1) == epochs:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": loss.item(),
                "is_phase": is_phase,
                "phase_mode": "diff",             # ✅ Δphase 모드 명시
                "conditional": conditional,
                "norm_type": norm_type,
                "min_db": min_db, "max_db": max_db,
                "db_mu": dataset.db_mu, "db_sigma": dataset.db_sigma
            }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt"))

    return model, diffusion

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--is_phase", type=str2bool, default=False)
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--data_dir", type=str, default="data/magnitude/")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--norm_type", type=str, choices=["zscore","0to1","-1to1"], default="zscore")
    p.add_argument("--min_db", type=float, default=-80.0)
    p.add_argument("--max_db", type=float, default=0.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model, diffusion = train_diffusion(
        is_phase=args.is_phase, conditional=args.conditional,
        data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        timesteps=args.timesteps, base_ch=args.base_ch, ckpt_dir=args.ckpt_dir, device=device,
        save_interval=args.save_interval, norm_type=args.norm_type, min_db=args.min_db, max_db=args.max_db
    )