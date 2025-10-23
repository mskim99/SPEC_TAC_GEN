import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model import UNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# -----------------------------
# Global normalization utils
# -----------------------------
def norm_z_global(x, mu, std):
    mu  = torch.as_tensor(mu,  dtype=x.dtype, device=x.device).view(1,2,1,1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1,2,1,1)
    return (x - mu) / (std + 1e-8)

def denorm_z_global(x, mu, std):
    mu  = torch.as_tensor(mu,  dtype=x.dtype, device=x.device).view(1,2,1,1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1,2,1,1)
    return x * std + mu

# -----------------------------
# NEW: Affine-invariant loss
# -----------------------------
def scale_invariant_mse(pred, gt, eps=1e-8):
    """Affine-invariant (scale-invariant) MSE"""
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        p = pred[b].flatten()
        g = gt[b].flatten()
        alpha = (p @ g) / (p @ p + eps)
        total += torch.mean((alpha * p - g) ** 2)
    return total / B

# -----------------------------
# TV loss
# -----------------------------
def tv_loss(x):  # x:(B,2,H,W)
    return (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean() + (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean()

# -----------------------------
# Diffusion
# -----------------------------
class Diffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.device = torch.device(device)
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = (1.0 - self.betas)
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def _cosine_beta_schedule(self, T, s=0.008, beta_min=1e-4, beta_max=0.02):
        steps = T + 1
        x = torch.linspace(0, T, steps, device=self.device)
        ac = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.clamp(betas, beta_min, beta_max)

    def add_noise(self, x0, t):
        a_hat = self.alpha_hat[t].view(-1,1,1,1)
        mean = torch.sqrt(a_hat) * x0
        std  = torch.sqrt(1.0 - a_hat)
        eps = torch.randn_like(x0)
        return mean + std * eps, eps

# -----------------------------
# Dataset
# -----------------------------
class WaveletComplexDataset(Dataset):
    def __init__(self, real_dir, imag_dir, norm="per_scale_z"):
        self.real_files = sorted(glob.glob(os.path.join(real_dir, "*.npy")))
        self.imag_files = sorted(glob.glob(os.path.join(imag_dir, "*.npy")))
        assert len(self.real_files) == len(self.imag_files)
        self.norm = norm

    def __len__(self): return len(self.real_files)

    def _per_scale_z(self, x):
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mu) / std

    def _normalize_pair(self, real, imag):
        if self.norm == "per_scale_z":
            real = self._per_scale_z(real)
            imag = self._per_scale_z(imag)
        elif self.norm == "none":
            pass
        else:
            m = real.mean(); s = real.std() + 1e-8
            real = (real - m) / s
            m = imag.mean(); s = imag.std() + 1e-8
            imag = (imag - m) / s
        return real, imag

    def __getitem__(self, idx):
        r = np.load(self.real_files[idx]).astype(np.float32)
        g = np.load(self.imag_files[idx]).astype(np.float32)
        r = r.squeeze()
        g = g.squeeze()
        assert r.ndim == 2 and g.ndim == 2, f"Bad shape: real={r.shape}, imag={g.shape}"
        r, g = self._normalize_pair(r, g)
        x = np.stack([r, g], axis=0)
        return torch.from_numpy(x)

# -----------------------------
# Losses
# -----------------------------
def circular_mse_phase(pred_complex, gt_complex, eps=1e-8):
    pr = pred_complex[:,0]; pi = pred_complex[:,1]
    gr = gt_complex[:,0]; gi = gt_complex[:,1]
    pred_ang = torch.atan2(pi, pr + eps)
    gt_ang   = torch.atan2(gi, gr + eps)
    d = torch.atan2(torch.sin(pred_ang-gt_ang), torch.cos(pred_ang-gt_ang))
    return (d**2).mean()

class WaveletPerceptualLoss(nn.Module):
    def __init__(self, scales=(1,2,4), kernel_size=3):
        super().__init__()
        self.scales = scales
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
    def forward(self, pred, gt):
        loss = 0.0
        p, g = pred, gt
        for s in self.scales:
            if s > 1:
                p = self.pool(p)
                g = self.pool(g)
            pmag = torch.sqrt(p[:,0]**2 + p[:,1]**2 + 1e-8)
            gmag = torch.sqrt(g[:,0]**2 + g[:,1]**2 + 1e-8)
            loss = loss + torch.mean(torch.abs(pmag - gmag))
        return loss / len(self.scales)

class LogFreqAwareMSE(nn.Module):
    def forward(self, pred, target):
        B,C,H,W = pred.shape
        f = torch.arange(1, H+1, device=pred.device).float()
        w = torch.log(1.0 + f) / torch.log(torch.tensor(1.0 + H, device=pred.device))
        w = w.view(1,1,H,1)
        return torch.mean(w * (pred - target)**2)

# -----------------------------
# DDPM helpers
# -----------------------------
def predict_x0_from_noise(x_t, noise, a_hat_t):
    return (x_t - torch.sqrt(1.0 - a_hat_t) * noise) / (torch.sqrt(a_hat_t) + 1e-8)

# -----------------------------
# Train
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "tb"))

    ds = WaveletComplexDataset(args.real_dir, args.imag_dir, norm=args.norm_type)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)

    model = UNet(in_ch=2, out_ch=2, base_ch=args.base_ch,
                 ch_mult=tuple(args.ch_mult), conditional=False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    mse = nn.MSELoss()
    l1  = nn.L1Loss()
    freq_loss = LogFreqAwareMSE()
    wave_perc = WaveletPerceptualLoss()
    diffusion = Diffusion(args.timesteps, device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_stats = None
    if args.norm_type == "global_z":
        real_files = ds.real_files
        imag_files = ds.imag_files
        mus, stds = [], []
        for rfile, ifile in zip(real_files, imag_files):
            r = np.load(rfile).astype(np.float32)
            i = np.load(ifile).astype(np.float32)
            mus.append([r.mean(), i.mean()])
            stds.append([r.std(), i.std()])
        mu = np.mean(mus, axis=0).tolist()
        std = np.mean(stds, axis=0).tolist()
        print(f"[INFO] Computed norm_mu={mu}, norm_std={std}")
        global_stats = {"mu": mu, "std": std}

    global_step = 0
    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for step, x0 in enumerate(dl, start=1):
            global_step += 1
            x0 = x0.to(device, non_blocking=True)
            if args.norm_type == "global_z" and global_stats is not None:
                x0 = norm_z_global(x0, global_stats["mu"], global_stats["std"])

            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device).long()
            with torch.cuda.amp.autocast(enabled=args.amp):
                x_t, noise = diffusion.add_noise(x0, t)
                pred_noise = model(x_t, t)
                a_hat_t = diffusion.alpha_hat[t].view(-1,1,1,1)
                x0_pred = predict_x0_from_noise(x_t, pred_noise, a_hat_t)

                loss_noise = mse(pred_noise, noise)
                loss_x0_l1 = l1(x0_pred, x0)
                loss_x0_si = scale_invariant_mse(x0_pred, x0)
                loss_x0    = 0.5 * (loss_x0_l1 + loss_x0_si)
                loss_phase = circular_mse_phase(x0_pred, x0)
                loss_freq  = freq_loss(x0_pred, x0)
                loss_perc  = wave_perc(x0_pred, x0)

                mean_pred, mean_gt = x0_pred.mean(), x0.mean()
                std_pred, std_gt   = x0_pred.std(),  x0.std()
                loss_scale = (mean_pred - mean_gt).abs() + (std_pred - std_gt).abs()

                loss = (args.l_noise * loss_noise
                        + args.l_x0 * loss_x0
                        + args.l_phase * loss_phase
                        + args.l_freq * loss_freq
                        + args.l_perc * loss_perc
                        + 0.05 * loss_scale)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += loss.item()

            print(f"[Epoch {ep:03d} | Step {step:04d}] "
                    f"loss={loss.item():.6f} | noise={loss_noise.item():.6f} | "
                    f"x0={loss_x0.item():.6f} | phase={loss_phase.item():.6f} | "
                    f"freq={loss_freq.item():.6f} | perc={loss_perc.item():.6f}")

            # === TensorBoard write ===
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/loss_noise", loss_noise.item(), global_step)
            writer.add_scalar("train/loss_x0", loss_x0.item(), global_step)
            writer.add_scalar("train/loss_phase", loss_phase.item(), global_step)
            writer.add_scalar("train/loss_freq", loss_freq.item(), global_step)
            writer.add_scalar("train/loss_perc", loss_perc.item(), global_step)

        avg_loss = total_loss / len(dl)
        print(f"[Epoch {ep}] Avg Loss: {avg_loss:.6f}")
        writer.add_scalar("train/epoch_avg_loss", avg_loss, ep)
        sched.step()

        # === Save checkpoint ===
        if ep % args.save_interval == 0 or ep == args.epochs:
            cfg = dict(vars(args))
            if global_stats is not None:
                cfg["norm_mu"] = global_stats["mu"]
                cfg["norm_std"] = global_stats["std"]
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "cfg": cfg
            }, os.path.join(args.ckpt_dir, f"ckpt_ep{ep:04d}.pt"))

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True)
    ap.add_argument("--imag_dir", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="ckpts_v6")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--save_interval", type=int, default=20)
    ap.add_argument("--norm_type", type=str, default="per_scale_z",
                    choices=["per_scale_z","zscore","global_z","none"])
    ap.add_argument("--amp", action="store_true")
    # ap.add_argument("--global_mu", type=float, nargs=2, default=None)
    # ap.add_argument("--global_std", type=float, nargs=2, default=None)
    ap.add_argument("--l_noise", type=float, default=1.0)
    ap.add_argument("--l_x0",    type=float, default=1.0)
    ap.add_argument("--l_phase", type=float, default=0.5)
    ap.add_argument("--l_freq",  type=float, default=0.5)
    ap.add_argument("--l_perc",  type=float, default=0.3)
    args = ap.parse_args()
    train(args)