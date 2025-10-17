import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main_mult import UNet, Diffusion  # 개선된 학습 코드의 클래스 불러오기

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ===============================
# Model Loader
# ===============================
def load_model_complex(ckpt_path, device="cuda", base_ch=32, timesteps=300):
    """complex wavelet checkpoint 로드"""
    in_ch = out_ch = 2
    model = UNet(in_ch, out_ch, base_ch, [1, 2, 4, 8]).to(device)
    diffusion = Diffusion(timesteps, device)
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")
    return model, diffusion, ckpt


# ===============================
# Diffusion Sampling
# ===============================
@torch.no_grad()
def sample_complex(model, diffusion, shape, device="cuda"):
    """
    complex wavelet diffusion 모델에서 샘플링
    (real, imag) 채널 동시 생성
    """
    img = torch.randn(shape, device=device)

    print(f"🎨 Sampling start: shape={shape}, timesteps={diffusion.timesteps}")
    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        pred = model(img, t_tensor)
        a = diffusion.alphas[t]
        ah = diffusion.alpha_hat[t]
        b = diffusion.betas[t]
        noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
        img = 1 / torch.sqrt(a) * (img - ((1 - a) / torch.sqrt(1 - ah)) * pred) + torch.sqrt(b) * noise

    img = img.squeeze().cpu().numpy()  # shape: (2, H, W)
    real, imag = img[0], img[1]
    return real, imag


# ===============================
# Main
# ===============================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Load checkpoint
    model, diffusion, ckpt = load_model_complex(args.ckpt, device, args.base_ch, args.timesteps)
    print(f"✅ Sampling config | norm: {ckpt.get('norm_type', 'tanh')} | lambda_freq: {ckpt.get('lambda_freq', 0.5)} | lambda_phase: {ckpt.get('lambda_phase', 0.3)}")

    H, W = args.shape
    shape = (1, 2, H, W)

    for i in range(args.num_samples):
        print(f"\n🎨 Generating sample {i+1}/{args.num_samples} ...")
        real, imag = sample_complex(model, diffusion, shape, device)

        # Save npy
        np.save(os.path.join(args.out_dir, f"sample_{i}_real.npy"), real)
        np.save(os.path.join(args.out_dir, f"sample_{i}_imag.npy"), imag)

        # Save visualizations
        plt.imsave(os.path.join(args.out_dir, f"sample_{i}_real.png"), real, cmap="gray", vmin=-1, vmax=1)
        plt.imsave(os.path.join(args.out_dir, f"sample_{i}_imag.png"), imag, cmap="gray", vmin=-1, vmax=1)

    print(f"\n✅ Sampling complete — {args.num_samples} samples saved in {args.out_dir}")


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complex Wavelet Diffusion Sampling")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--out_dir", type=str, default="samples_complex", help="Output directory")
    parser.add_argument("--shape", type=int, nargs=2, default=[1024, 1024], help="Output shape as H W")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=300)
    args = parser.parse_args()
    main(args)
