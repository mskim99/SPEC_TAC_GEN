import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
from model_complex import UNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
# One reverse-diffusion step (DDIM)
# =====================================================
@torch.no_grad()
def ddim_sample_step(model,
                     x_t,
                     t_current,
                     t_prev,
                     alpha_hat,
                     eta):
    """
    DDIM reverse step.
    x_{t-1} = sqrt(α̂_{t-1}) * pred_x0 + sqrt(1 - α̂_{t-1} - σ_t^2) * ε_θ + σ_t * z
    """
    B = x_t.shape[0]
    device = x_t.device

    # 1. Get parameters for t_current and t_prev
    a_t = alpha_hat[t_current].view(B, 1, 1, 1)
    a_prev = alpha_hat[t_prev].view(B, 1, 1, 1) \
        if t_prev >= 0 else \
        torch.tensor(1.0, device=device).view(B, 1, 1, 1)  # a_hat[-1] = 1.0

    # 2. Get model prediction (ε_θ)
    t_long = torch.full((B,), t_current, device=device, dtype=torch.long)
    pred_noise = model(x_t, t_long)

    # 3. Calculate predicted x_0
    pred_x0 = (x_t - torch.sqrt(1.0 - a_t) * pred_noise) / torch.sqrt(a_t)

    # 4. Handle final step (t_prev = -1) -> return pred_x0
    if t_prev < 0:
        return pred_x0

    # 5. Calculate σ_t (stochasticity)
    sigma_t = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev))

    # 6. Calculate coefficient for pred_noise (ε_θ)
    pred_noise_coeff = torch.sqrt(1.0 - a_prev - sigma_t ** 2)

    # 7. Sample noise z
    noise = 0.0
    if eta > 0:
        noise = torch.randn_like(x_t)

    # 8. Calculate x_{t-1}
    x_prev = torch.sqrt(a_prev) * pred_x0 + \
             pred_noise_coeff * pred_noise + \
             sigma_t * noise

    return x_prev


# =====================================================
# Full sampling loop (with optional intermediate saves & TB logging)
# =====================================================
@torch.no_grad()
def sample_complex(model,
                   diffusion,
                   shape,  # type: Tuple[int, int, int, int]
                   device="cuda",
                   save_every=0,
                   out_dir=None,
                   tag_prefix="sample",
                   # tb_writer=None,  # type: Optional[SummaryWriter]
                   tb_every=0,
                   noise_scale=1.0,
                   # --- [수정된 부분] ---
                   use_ddim=False,  # DDIM 사용할지 여부
                   ddim_steps=50,  # DDIM 스텝 수
                   eta=0.0):  # DDIM eta (0=deterministic)
    # --- [수정 끝] ---
    """
    shape: (B, 2, H, W)  — channel 0: real, 1: imag
    save_every: save intermediate x_t every N steps (0=off)
    tb_every:   log intermediate to TensorBoard every N steps (0=off)
    """
    device = torch.device(device)
    B, C, H, W = shape
    x = torch.randn(shape, device=device)

    # --- [수정된 부분] ---
    # (기존 _to_img 헬퍼 함수 - 이전 답변에서 수정됨)
    def _to_img(tensor_chw):  # (C, H, W)
        min_val = tensor_chw.min()
        max_val = tensor_chw.max()
        arr = (tensor_chw - min_val) / (max_val - min_val + 1e-8)
        return arr

    # DDIM vs DDPM 분기 처리
    if use_ddim:
        # DDIM 샘플링
        print(f"🎨 Sampling start (DDIM): shape={shape} ddim_steps={ddim_steps} eta={eta} on {device}")

        alpha_hat = diffusion.alpha_hat.to(device)

        # [T-1, ..., 0] 에 해당하는 ddim_steps 개의 시퀀스 생성
        times = torch.linspace(-1, diffusion.timesteps - 1, ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for t_current, t_prev in time_pairs:
            x = ddim_sample_step(
                model,
                x,
                t_current,
                t_prev,
                alpha_hat,
                eta
            )

            # tensorboard intermediates (로깅 시점 t = t_current 사용)
            '''
            if (tb_writer is not None) and tb_every and (t_current % tb_every == 0 or t_prev == -1):
                r = _to_img(x[0, 0:1])
                im = _to_img(x[0, 1:2])
                global_step = diffusion.timesteps - 1 - t_current
                tb_writer.add_image("{}/real_t{:04d}".format(tag_prefix, t_current), r, global_step=global_step)
                tb_writer.add_image("{}/imag_t{:04d}".format(tag_prefix, t_current), im, global_step=global_step)
            '''

    else:
        # 기존 DDPM 샘플링
        print("🎨 Sampling start (DDPM): shape={} timesteps={} on {}".format(shape, diffusion.timesteps, device))
        betas = diffusion.betas.to(device)
        alphas = diffusion.alphas.to(device)
        alpha_hat = diffusion.alpha_hat.to(device)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat)

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

            # (기존 DDPM 로깅 - 수정 없음)
            if (tb_writer is not None) and tb_every and (t % tb_every == 0 or t == diffusion.timesteps - 1):
                r = _to_img(x[0, 0:1])
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

    # DDIM 사용 여부 결정
    use_ddim = args.ddim_steps > 0
    if use_ddim:
        print(f"   → 🚀 Using DDIM sampler: steps={args.ddim_steps}, eta={args.eta}")
    else:
        print(f"   → 🐌 Using DDPM sampler: steps={timesteps}")

    H, W = args.shape
    B = args.batch
    shape = (B, 2, H, W)

    # TensorBoard writer (optional)
    # writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb")) if args.tensorboard else None

    # num_samples 만큼 루프를 도는 대신, 필요한 배치 수 만큼 루프를 돌고
    # 각 배치의 모든 샘플을 저장합니다.
    total_samples_needed = args.num_samples
    samples_generated = 0
    num_batches = (total_samples_needed + B - 1) // B  # 필요한 배치의 수 (올림)

    print(f"🎨 총 {total_samples_needed}개 샘플 생성 시작 (배치 크기 {B}, 총 {num_batches} 배치)")

    for i in range(num_batches):
        print(f"\n🎨 배치 {i + 1}/{num_batches} 생성 중...")
        batch_tag = "batch{:03d}".format(i)  # TB용 배치 태그

        x = sample_complex(
            model,
            diffusion,
            shape,  # (B, 2, H, W)
            device=device,
            save_every=args.save_every,
            out_dir=args.out_dir,
            tag_prefix=batch_tag,  # 중간 로그용 태그
            # tb_writer=writer,
            tb_every=args.tb_every,
            noise_scale=args.noise_scale,
            use_ddim=use_ddim,  # DDIM 인자 전달
            ddim_steps=args.ddim_steps,  # DDIM 인자 전달
            eta=args.eta  # DDIM 인자 전달
        )
        x = denorm_from_cfg(x, cfg)  # (B, 2, H, W)

        # 배치 루프: 생성된 B개의 샘플을 순회하며 저장
        for j in range(B):
            if samples_generated >= total_samples_needed:
                break  # 요청한 샘플 수를 모두 채웠으면 중지

            tag = "sample_{:03d}".format(samples_generated)
            print(f"   ... 저장 중: {tag} (배치 {i + 1}, 샘플 {j + 1})")

            # save npy (원본 데이터 저장)
            x_np = x[j].cpu().numpy()  # (2,H,W)
            real, imag = x_np[0], x_np[1]
            np.save(os.path.join(args.out_dir, "{}_real.npy".format(tag)), real)
            np.save(os.path.join(args.out_dir, "{}_imag.npy".format(tag)), imag)

            # ==== PNG 저장 (이 부분은 vmin/vmax 로직이 이미 올바르게 구현됨) ====
            vmin = args.vmin if args.vmin is not None else real.min()
            vmax = args.vmax if args.vmax is not None else real.max()
            # (원본 로직은 real의 min/max를 imag에도 사용하므로 그대로 둡니다)
            plt.imsave(os.path.join(args.out_dir, "{}_real.png".format(tag)), real, cmap="gray", vmin=vmin, vmax=vmax)
            plt.imsave(os.path.join(args.out_dir, "{}_imag.png".format(tag)), imag, cmap="gray", vmin=vmin, vmax=vmax)

            # --- [수정된 부분] ---
            # 문제 2 수정: TB 최종 이미지 저장 시 [-1, 1] 클리핑 제거
            # if writer is not None:
            r = torch.from_numpy(real).unsqueeze(0)  # (1,H,W)
            im = torch.from_numpy(imag).unsqueeze(0)

            # [-1, 1] 클리핑 대신 동적 범위 정규화
            r_min, r_max = r.min(), r.max()
            im_min, im_max = im.min(), im.max()

            r = (r - r_min) / (r_max - r_min + 1e-8)
            im = (im - im_min) / (im_max - im_min + 1e-8)

            # writer.add_image("{}/final_real".format(tag), r, global_step=samples_generated)
            # writer.add_image("{}/final_imag".format(tag), im, global_step=samples_generated)

            samples_generated += 1

        if samples_generated >= total_samples_needed:
            break

    # if writer is not None:
        # writer.close()

    print(f"\n✅ 샘플링 완료 — {samples_generated}개의 샘플이 {args.out_dir}에 저장되었습니다.")

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
    # p.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise_scale", type=float, default=1.0, help="extra noise temperature in reverse step")
    p.add_argument("--vmin", type=float, default=None, help="PNG 저장시 vmin (GT 최소값 넣으면 공정 비교)")
    p.add_argument("--vmax", type=float, default=None, help="PNG 저장시 vmax (GT 최대값)")
    p.add_argument("--ddim_steps", type=int, default=0,
                   help="Use DDIM sampler with N steps (0 = use default DDPM sampler)")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM eta parameter (0.0 = deterministic DDIM, 1.0 = DDPM-like stochastic)")
    args = p.parse_args()
    main(args)