import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
from model import UNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ==== NEW: denorm from saved cfg ====
def denorm_from_cfg(x, cfg):
    if cfg is None:
        return x
    mu  = cfg.get("norm_mu", None)
    std = cfg.get("norm_std", None)
    if (mu is None) or (std is None):
        return x
    mu  = torch.tensor(mu,  dtype=x.dtype, device=x.device).view(1,2,1,1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(1,2,1,1)
    return x * std + mu
# ==== /NEW ====

# =====================================================
# Diffusion (cosine schedule) — similar spirit to main_mult
# =====================================================
class Diffusion(object):
    def __init__(self, timesteps=1000, device="cuda"):
        self.device = torch.device(device)
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = (1.0 - self.betas)
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def _cosine_beta_schedule(self, T, s=0.008, beta_min=1e-4, beta_max=0.02):
        steps = T + 1
        x = torch.linspace(0, T, steps, device=self.device)
        ac = torch.cos(((x / T) + s) / (1.0 + s) * np.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1.0 - (ac[1:] / ac[:-1])
        return torch.clamp(betas, beta_min, beta_max)


# =====================================================
# One reverse-diffusion step
# =====================================================
@torch.no_grad()
def p_sample(model,
             x,
             t_long,
             t_index,
             betas,
             sqrt_one_minus_alpha_hat,
             sqrt_recip_alphas,
             noise_scale=1.0):
    """
    DDPM reverse step with ε-theta prediction:
      x_{t-1} = sqrt(1/α_t) * (x_t - (β_t / sqrt(1-α̂_t)) * ε_θ(x_t,t)) + sqrt(β_t) * z
    """
    pred_noise = model(x, t_long)  # ε_θ(x_t, t)
    beta_t = betas[t_index]
    sqrt_recip_alpha = sqrt_recip_alphas[t_index]
    sqrt_one_minus_alpha = sqrt_one_minus_alpha_hat[t_index]
    mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha * pred_noise)
    if t_index > 0:
        noise = torch.randn_like(x) * noise_scale
    else:
        noise = 0.0
    return mean + torch.sqrt(beta_t) * noise


# =====================================================
# Full sampling loop (with optional intermediate saves & TB logging)
# =====================================================
@torch.no_grad()
def sample_complex(model,
                   diffusion,
                   shape,                     # type: Tuple[int, int, int, int]
                   device="cuda",
                   save_every=0,
                   out_dir=None,
                   tag_prefix="sample",
                   tb_writer=None,            # type: Optional[SummaryWriter]
                   tb_every=0,
                   noise_scale=1.0):
    """
    shape: (B, 2, H, W)  — channel 0: real, 1: imag
    save_every: save intermediate x_t every N steps (0=off)
    tb_every:   log intermediate to TensorBoard every N steps (0=off)
    """
    device = torch.device(device)
    betas = diffusion.betas.to(device)
    alphas = diffusion.alphas.to(device)
    alpha_hat = diffusion.alpha_hat.to(device)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat)

    x = torch.randn(shape, device=device)
    B, C, H, W = x.shape
    print("🎨 Sampling start: shape={} timesteps={} on {}".format(shape, diffusion.timesteps, device))

    def _to_img(tensor_2d):
        # clip to [-1,1] -> [0,1]
        arr = torch.clamp(tensor_2d, -1.0, 1.0)
        arr = (arr + 1.0) * 0.5
        return arr

    for t in reversed(range(diffusion.timesteps)):
        t_long = torch.full((B,), t, device=device, dtype=torch.long)
        x = p_sample(
            model,
            x,
            t_long,
            t,
            betas,
            sqrt_one_minus_alpha_hat,
            sqrt_recip_alphas,
            noise_scale=noise_scale,
        )

        # save intermediates
        '''
        if save_every and out_dir and (t % save_every == 0 or t == diffusion.timesteps - 1):
            xr = x[0, 0].detach().cpu().numpy()
            xi = x[0, 1].detach().cpu().numpy()
            plt.imsave(os.path.join(out_dir, "{}_t{:04d}_real.png".format(tag_prefix, t)), xr, cmap="gray", vmin=-1, vmax=1)
            plt.imsave(os.path.join(out_dir, "{}_t{:04d}_imag.png".format(tag_prefix, t)), xi, cmap="gray", vmin=-1, vmax=1)
        '''

        # tensorboard intermediates
        if (tb_writer is not None) and tb_every and (t % tb_every == 0 or t == diffusion.timesteps - 1):
            # Only first sample (index 0) to keep logs light
            r = _to_img(x[0, 0:1])  # (1,H,W)
            im = _to_img(x[0, 1:2])
            global_step = diffusion.timesteps - 1 - t
            tb_writer.add_image("{}/real_t{:04d}".format(tag_prefix, t), r, global_step=global_step)
            tb_writer.add_image("{}/imag_t{:04d}".format(tag_prefix, t), im, global_step=global_step)

    return x.detach()  # (B,2,H,W)


# =====================================================
# Main
# =====================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})  # saved by main_mult.py

    # resolve model config
    base_ch = args.base_ch if args.base_ch is not None else cfg.get("base_ch", 64)
    ch_mult = args.ch_mult if args.ch_mult is not None else cfg.get("ch_mult", [1, 2, 4, 8])
    if isinstance(ch_mult, tuple):
        ch_mult = list(ch_mult)

    # resolve sampling timesteps
    timesteps = args.timesteps if args.timesteps is not None else cfg.get("timesteps", 1000)

    # Build & load model
    model = UNet(in_ch=2, out_ch=2, base_ch=base_ch, ch_mult=tuple(ch_mult), conditional=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = Diffusion(timesteps=timesteps, device=device)

    print("✅ Loaded model: {}".format(args.ckpt))
    if "cfg" in ckpt:
        print("   → cfg: base_ch={}, ch_mult={}, timesteps={}, norm_type={}".format(
            base_ch, ch_mult, timesteps, cfg.get('norm_type', '-')))
    else:
        print("   → (no cfg found in ckpt) base_ch={}, ch_mult={}, timesteps={}".format(
            base_ch, ch_mult, timesteps))
    print("   → device = {}".format(device))

    H, W = args.shape
    B = args.batch
    shape = (B, 2, H, W)

    # TensorBoard writer (optional)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb")) if args.tensorboard else None

    # Generate samples
    for i in range(args.num_samples):
        tag = "sample{:03d}".format(i)
        print("\n🎨 Generating {} ({}/{}) ...".format(tag, i + 1, args.num_samples))
        x = sample_complex(
            model,
            diffusion,
            shape,
            device=device,
            save_every=args.save_every,
            out_dir=args.out_dir,
            tag_prefix=tag,
            tb_writer=writer,
            tb_every=args.tb_every,
            noise_scale=args.noise_scale,
        )
        x = denorm_from_cfg(x, cfg)
        # save npy & png for the first in batch
        x_np = x[0].cpu().numpy()  # (2,H,W)
        real, imag = x_np[0], x_np[1]
        np.save(os.path.join(args.out_dir, "{}_real.npy".format(tag)), real)
        np.save(os.path.join(args.out_dir, "{}_imag.npy".format(tag)), imag)

        # ==== NEW: optional vmin/vmax for fair visualization ====
        vmin = args.vmin if args.vmin is not None else real.min()
        vmax = args.vmax if args.vmax is not None else real.max()

        plt.imsave(os.path.join(args.out_dir, "{}_real.png".format(tag)), real, cmap="gray", vmin=vmin, vmax=vmax)
        plt.imsave(os.path.join(args.out_dir, "{}_imag.png".format(tag)), imag, cmap="gray", vmin=vmin, vmax=vmax)

        # also TB final image (normalized)
        if writer is not None:
            r = torch.from_numpy(real).unsqueeze(0)  # (1,H,W)
            im = torch.from_numpy(imag).unsqueeze(0)
            r = torch.clamp(r, -1.0, 1.0)
            im = torch.clamp(im, -1.0, 1.0)
            r = (r + 1.0) * 0.5
            im = (im + 1.0) * 0.5
            writer.add_image("{}/final_real".format(tag), r, global_step=i)
            writer.add_image("{}/final_imag".format(tag), im, global_step=i)

    if writer is not None:
        writer.close()

    print("\n✅ Sampling complete — {} samples saved in {}".format(args.num_samples, args.out_dir))


# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Complex Wavelet Diffusion — Sampling (Py3.7)")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    p.add_argument("--out_dir", type=str, default="samples_complex")
    p.add_argument("--shape", type=int, nargs=2, default=[1024, 1024], help="H W")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--batch", type=int, default=1, help="batch size during sampling")
    p.add_argument("--base_ch", type=int, default=None, help="override; else use ckpt cfg")
    p.add_argument("--ch_mult", type=int, nargs="+", default=None, help="override; else use ckpt cfg")
    p.add_argument("--timesteps", type=int, default=None, help="override; else use ckpt cfg")
    p.add_argument("--save_every", type=int, default=0, help="save x_t every N steps (0=off)")
    p.add_argument("--tb_every", type=int, default=0, help="TensorBoard image log every N steps (0=off)")
    p.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise_scale", type=float, default=1.0, help="extra noise temperature in reverse step")
    p.add_argument("--vmin", type=float, default=None, help="PNG 저장시 vmin (GT 최소값 넣으면 공정 비교)")
    p.add_argument("--vmax", type=float, default=None, help="PNG 저장시 vmax (GT 최대값)")
    args = p.parse_args()
    main(args)