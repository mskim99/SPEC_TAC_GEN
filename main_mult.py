# -*- coding: utf-8 -*-
"""
Unified training & sampling for complex STFT (Re/Im) with:
 - cosine beta schedule
 - STFT consistency loss
 - TV regularization
 - DDIM sampler (eta=0 default)
The network outputs Complex STFT (Re, Im) jointly (no --is_phase).
Magnitude/phase는 샘플링 시 from (Re, Im)로 복원합니다.
"""

import os, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import istft, stft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === (1) 모델은 기존 main.py의 UNet 재사용 ===
# in_ch=2, out_ch=2 (Re, Im)
try:
    from main import UNet  # 사용자 프로젝트의 UNet
except Exception as e:
    raise ImportError("main.py의 UNet을 import할 수 없습니다. 동일한 폴더에 main.py가 있어야 합니다.") from e


# =========================
# Diffusion (cosine schedule)
# =========================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 1e-8, 0.999), dtype=torch.float32)

class Diffusion:
    def __init__(self, timesteps=300, betas=None, device="cuda"):
        self.device = device
        if betas is None:
            betas = cosine_beta_schedule(timesteps)
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas)
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.timesteps = timesteps

    def add_noise(self, x0, t):
        """
        x0: clean (B,C,F,T)
        t:  (B,)
        """
        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(a_hat) * x0 + torch.sqrt(1 - a_hat) * noise
        return xt, noise


# =========================
# DDIM Sampler (eta=0 default)
# =========================
@torch.no_grad()
def ddim_sample(model, diffusion, shape, device, eta=0.0):
    x = torch.randn(shape, device=device)
    T = diffusion.timesteps
    alphas = diffusion.alphas
    a_hat = diffusion.alpha_hat

    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        alpha_hat_t = a_hat[i]
        x0 = (x - torch.sqrt(1 - alpha_hat_t) * eps) / torch.sqrt(alpha_hat_t + 1e-8)

        if i == 0:
            x = x0
            break

        alpha_hat_prev = a_hat[i - 1]
        sigma_t = eta * torch.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat_t)) * torch.sqrt(1 - alpha_hat_t / alpha_hat_prev)
        dir_xt = torch.sqrt(1 - alpha_hat_prev - sigma_t ** 2 + 1e-8) * eps
        z = torch.randn_like(x) if eta > 0 else 0.0
        x = torch.sqrt(alpha_hat_prev) * x0 + dir_xt + sigma_t * z
    return x


# =========================
# Dataset: load paired mag & phase -> build Re/Im
# =========================
class ComplexSpectrogramDataset(Dataset):
    """
    한 샘플 = (Re, Im) 2채널로 정규화하여 반환.
    GT magnitude/phase .npy가 같은 index를 공유한다고 가정(파일명 정렬).
    """
    def __init__(self, mag_dir, phs_dir, freq_wise_norm=False):
        self.mag_files = sorted(glob.glob(os.path.join(mag_dir, "*.npy")))
        self.phs_files = sorted(glob.glob(os.path.join(phs_dir, "*.npy")))
        assert len(self.mag_files) == len(self.phs_files) and len(self.mag_files) > 0, \
            "mag_dir과 phs_dir의 파일 수가 일치해야 합니다."

        # 통계 계산
        re_list, im_list = [], []
        for mp, pp in zip(self.mag_files, self.phs_files):
            mag = np.load(mp, allow_pickle=True)
            phs = np.load(pp, allow_pickle=True)
            re = mag * np.cos(phs)
            im = mag * np.sin(phs)
            re_list.append(re)
            im_list.append(im)

        re_stack = np.stack(re_list, axis=0)  # (N,F,T)
        im_stack = np.stack(im_list, axis=0)
        if freq_wise_norm:
            # (F,T) 전체에서 F축 기준 μ/σ를 잡으려면 더 복잡하지만
            # 여기서는 global μ/σ를 기본으로 두고 옵션으로만 제공
            self.re_mu = re_stack.mean(axis=(0,2)).mean()
            self.re_std = re_stack.std()
            self.im_mu = im_stack.mean(axis=(0,2)).mean()
            self.im_std = im_stack.std()
        else:
            self.re_mu = re_stack.mean()
            self.re_std = re_stack.std() + 1e-8
            self.im_mu = im_stack.mean()
            self.im_std = im_stack.std() + 1e-8

        # dB 표시용 범위(샘플링 시 시각화에 사용)
        self.min_db = -80.0
        self.max_db = 0.0

    def __len__(self):
        return len(self.mag_files)

    def __getitem__(self, idx):
        mag = np.load(self.mag_files[idx], allow_pickle=True).astype(np.float32)
        phs = np.load(self.phs_files[idx], allow_pickle=True).astype(np.float32)

        re = (mag * np.cos(phs) - self.re_mu) / self.re_std
        im = (mag * np.sin(phs) - self.im_mu) / self.im_std
        spec = np.stack([re, im], axis=0)  # (2, F, T)
        return torch.from_numpy(spec)


# =========================
# Loss helpers
# =========================
def stft_consistency_loss(pred, n_fft, hop):
    """
    pred: (B,2,F,T)  -> (Re,Im)
    1) 복소 STFT -> iSTFT -> STFT
    2) 재투영 STFT와 pred의 L1 차이
    """
    B, C, F, T = pred.shape
    re = pred[:, 0].detach().cpu().numpy()
    im = pred[:, 1].detach().cpu().numpy()
    loss = 0.0
    for b in range(B):
        Z = re[b] + 1j * im[b]
        _, x = istft(Z, nperseg=n_fft, noverlap=n_fft - hop)
        # shape 안정화를 위해 length 맞춤
        _, _, Zc = stft(x, nperseg=n_fft, noverlap=n_fft - hop)
        Zc = Zc[:F, :T]  # 크기 보호
        loss += np.mean(np.abs(Zc.real - re[b][:Zc.shape[0], :Zc.shape[1]]) +
                        np.abs(Zc.imag - im[b][:Zc.shape[0], :Zc.shape[1]]))
    return float(loss / B)


def tv_regularization(pred):
    """
    pred: (B,2,F,T)
    시간/주파수 방향 total variation
    """
    tv_t = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
    tv_f = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
    return tv_t + tv_f


# =========================
# Train
# =========================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    dataset = ComplexSpectrogramDataset(args.mag_dir, args.phs_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = UNet(in_ch=2, out_ch=2, base_ch=args.base_ch,
                 conditional=False, num_classes=0).to(device)

    diffusion = Diffusion(timesteps=args.timesteps, betas=cosine_beta_schedule(args.timesteps), device=device)
    optim_ = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Train samples: {len(dataset)} | timesteps={args.timesteps} | n_fft={args.n_fft}, hop={args.hop}")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            x0 = batch.to(device)  # (B,2,F,T)
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device).long()
            xt, noise = diffusion.add_noise(x0, t)
            eps_hat = model(xt, t)

            loss_main = F.mse_loss(eps_hat, noise)

            # TV regularization
            L_tv = tv_regularization(eps_hat)

            # STFT consistency (NumPy 기반)
            L_cons = stft_consistency_loss(eps_hat, args.n_fft, args.hop)

            loss = loss_main + args.lambda_tv * L_tv + args.lambda_cons * L_cons

            optim_.zero_grad()
            loss.backward()
            optim_.step()

            running += loss.item()

        print(f"[Epoch {epoch+1}/{args.epochs}] loss={running/len(loader):.6f}")

        if ((epoch + 1) % args.save_interval == 0) or (epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": float(running / max(1, len(loader))),
                "phase_mode": "complex",
                "re_mu": float(dataset.re_mu), "re_std": float(dataset.re_std),
                "im_mu": float(dataset.im_mu), "im_std": float(dataset.im_std),
                "min_db": float(dataset.min_db), "max_db": float(dataset.max_db),
                "n_fft": int(args.n_fft), "hop": int(args.hop),
                "timesteps": int(args.timesteps),
            }, ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")


# =========================
# Sample
# =========================
@torch.no_grad()
def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 모델/메타 로드
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet(in_ch=2, out_ch=2, base_ch=args.base_ch, conditional=False, num_classes=0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # timesteps = ckpt.get("timesteps", args.timesteps)
    timesteps = args.timesteps
    diffusion = Diffusion(timesteps=timesteps, betas=cosine_beta_schedule(timesteps), device=device)

    re_mu, re_std = ckpt.get("re_mu", 0.0), ckpt.get("re_std", 1.0)
    im_mu, im_std = ckpt.get("im_mu", 0.0), ckpt.get("im_std", 1.0)
    min_db, max_db = ckpt.get("min_db", -80.0), ckpt.get("max_db", 0.0)
    n_fft, hop = ckpt.get("n_fft", args.n_fft), ckpt.get("hop", args.hop)

    # shape 추정: 사용 중인 학습 데이터의 F,T와 동일해야 함
    F_bins = args.f_bins
    T_frames = args.t_frames
    shape = (args.num_samples, 2, F_bins, T_frames)

    print(f"Sampling {args.num_samples} with DDIM(eta={args.ddim_eta}) | shape={shape}")

    x = ddim_sample(model, diffusion, shape, device, eta=args.ddim_eta)
    x = x.cpu().numpy()  # (N,2,F,T)

    for i in range(args.num_samples):
        re = x[i, 0] * re_std + re_mu
        im = x[i, 1] * im_std + im_mu
        mag = np.sqrt(re**2 + im**2) + 1e-12
        phs = np.arctan2(im, re)

        # 저장
        np.save(os.path.join(args.out_dir, f"sample_{i+1:03d}_re.npy"), re)
        np.save(os.path.join(args.out_dir, f"sample_{i+1:03d}_im.npy"), im)
        np.save(os.path.join(args.out_dir, f"sample_{i+1:03d}_mag.npy"), mag)
        np.save(os.path.join(args.out_dir, f"sample_{i+1:03d}_phs.npy"), phs)

        plt.imsave(os.path.join(args.out_dir, f"sample_{i+1:03d}_mag.png"),
                   20*np.log10(mag) , cmap="jet")
        plt.imsave(os.path.join(args.out_dir, f"sample_{i+1:03d}_phs.png"),
                   phs, cmap="twilight_shifted")

        # iSTFT 복원
        Z = re + 1j * im
        _, wave = istft(Z, nperseg=n_fft, noverlap=n_fft - hop)
        np.save(os.path.join(args.out_dir, f"sample_{i+1:03d}_wave.npy"), wave)

        # wav 저장(선택)
        try:
            import soundfile as sf
            sf.write(os.path.join(args.out_dir, f"sample_{i+1:03d}.wav"), wave, args.sr)
        except Exception:
            pass

    # 메타 저장
    meta = {
        "re_mu": re_mu, "re_std": re_std, "im_mu": im_mu, "im_std": im_std,
        "min_db": min_db, "max_db": max_db, "n_fft": n_fft, "hop": hop,
        "timesteps": timesteps
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Saved samples & meta to: {args.out_dir}")


# =========================
# CLI
# =========================
def main():
    p = argparse.ArgumentParser(description="Unified complex STFT diffusion (train & sample)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train")
    t.add_argument("--mag_dir", type=str, required=True)
    t.add_argument("--phs_dir", type=str, required=True)
    t.add_argument("--ckpt_dir", type=str, default="ckpts_unified")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--base_ch", type=int, default=32)
    t.add_argument("--timesteps", type=int, default=300)
    t.add_argument("--save_interval", type=int, default=10)
    t.add_argument("--n_fft", type=int, default=256)
    t.add_argument("--hop", type=int, default=128)
    t.add_argument("--lambda_tv", type=float, default=0.05)
    t.add_argument("--lambda_cons", type=float, default=0.1)

    # sample
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", type=str, required=True)
    s.add_argument("--out_dir", type=str, default="samples_unified")
    s.add_argument("--num_samples", type=int, default=8)
    s.add_argument("--ddim_eta", type=float, default=0.0)
    s.add_argument("--base_ch", type=int, default=32)
    s.add_argument("--f_bins", type=int, default=129)    # 학습 데이터와 동일해야 함
    s.add_argument("--t_frames", type=int, default=376)  # 학습 데이터와 동일해야 함
    s.add_argument("--n_fft", type=int, default=256)     # fallback (ckpt에 있으면 무시)
    s.add_argument("--hop", type=int, default=128)       # fallback
    s.add_argument("--sr", type=int, default=16000)      # wav 저장 샘플레이트
    s.add_argument("--timesteps", type=int, default=300)

    args = p.parse_args()

    if args.cmd == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()