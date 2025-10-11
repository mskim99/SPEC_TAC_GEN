import os, argparse, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from main_unified import UNet, Diffusion, str2bool

# ----------------------------
def load_model(ckpt_path, device="cuda", base_ch=32, timesteps=300,
               conditional=False, num_classes=0, is_phase=False):
    in_ch = 2 if is_phase else 1
    out_ch = 2 if is_phase else 1
    model = UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch,
                 conditional=conditional, num_classes=num_classes).to(device)
    diffusion = Diffusion(timesteps, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded ckpt {ckpt_path}, epoch={ckpt['epoch']}, in_ch={in_ch}")
    return model, diffusion, ckpt

@torch.no_grad()
def sample(model, diffusion, shape, device,
           is_phase, conditional=False, cond=None,
           norm_type="zscore", min_db=-80, max_db=0, db_mu=None, db_sigma=None,
           phase_mode="diff"):
    img = torch.randn(shape, device=device)

    for t in reversed(range(diffusion.timesteps)):
        tt = torch.tensor([t], device=device).long()
        if conditional and cond is not None:
            pred = model(img, tt, cond)
        else:
            pred = model(img, tt)
        a = diffusion.alphas[t]; ah = diffusion.alpha_hat[t]; b = diffusion.betas[t]
        noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
        img = 1/torch.sqrt(a) * (img - ((1-a)/torch.sqrt(1-ah)) * pred) + torch.sqrt(b) * noise

    img = img.squeeze().cpu().numpy()

    if is_phase:
        # Δphase(sin,cos) → Δphase → φ 복원
        if phase_mode != "diff":
            raise ValueError("This checkpoint is not phase-difference mode.")
        sin_out, cos_out = img[0], img[1]     # (129, 375)
        # 안전 처리: clip + unit circle normalization
        sin_out = np.clip(sin_out, -1.0, 1.0)
        cos_out = np.clip(cos_out, -1.0, 1.0)
        norm = np.sqrt(sin_out**2 + cos_out**2 + 1e-8)
        sin_out /= norm; cos_out /= norm
        dphi = np.arctan2(sin_out, cos_out)   # [-π, π], (129, 375)
        phi = np.cumsum(dphi, axis=1)         # (129, 375)
        # 첫 프레임은 0으로 가정해 (129, 376) 복원
        phi0 = np.zeros((phi.shape[0], 1), dtype=phi.dtype)
        phase_recon = np.concatenate([phi0, phi], axis=1)     # (129, 376)
        # wrap [-π, π]
        phase_recon = np.angle(np.exp(1j * phase_recon))
        return phase_recon
    else:
        # magnitude: 역정규화
        if norm_type == "zscore":
            assert db_mu is not None and db_sigma is not None, "zscore reverse requires db_mu/db_sigma"
            spec_db = img * db_sigma + db_mu
        elif norm_type == "0to1":
            spec_db = img * (max_db - min_db) + min_db
        else:
            spec_db = (img + 1.0) / 2.0 * (max_db - min_db) + min_db
        spec_mag = 10 ** (spec_db / 20.0)
        return spec_mag

# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 조건 맵
    num_classes = 0; cond_map = None
    if args.conditional:
        cond_path = os.path.join(os.path.dirname(args.ckpt), "cond2idx.json")
        if not os.path.exists(cond_path):
            raise FileNotFoundError(f"cond2idx.json not found in {os.path.dirname(args.ckpt)}")
        with open(cond_path, "r") as f:
            cond_map = json.load(f)
        num_classes = len(cond_map)

    # ckpt 로드 + 메타
    # is_phase/in_ch는 실행 인자보다 ckpt 메타에 우선하여 맞추는 것이 안전
    tmp_ckpt = torch.load(args.ckpt, map_location="cpu")
    is_phase_ckpt = bool(tmp_ckpt.get("is_phase", args.is_phase))
    phase_mode = tmp_ckpt.get("phase_mode", "mag")  # "diff" or "mag"
    model, diff, ckpt = load_model(
        args.ckpt, device, args.base_ch, args.timesteps,
        args.conditional, num_classes, is_phase=is_phase_ckpt
    )

    norm_type = ckpt.get("norm_type", "zscore")
    min_db = ckpt.get("min_db", -80.0); max_db = ckpt.get("max_db", 0.0)
    db_mu = ckpt.get("db_mu", None); db_sigma = ckpt.get("db_sigma", None)
    print(f"✅ Norm from ckpt: {norm_type}, mu={db_mu}, sigma={db_sigma}, phase_mode={phase_mode}")

    # 생성할 조건 목록
    if args.conditional:
        if args.all_conditions:
            conditions = list(cond_map.keys())
        else:
            if args.condition is None or args.condition not in cond_map:
                raise ValueError("Valid --condition required or use --all_conditions")
            conditions = [args.condition]
    else:
        conditions = [None]

    for cname in conditions:
        if args.conditional:
            cond_idx = cond_map[cname]
            cond_tensor = torch.tensor([cond_idx], device=device).long()
            out_dir = os.path.join(args.out_dir, cname)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n=== Generating: {cname} (idx={cond_idx}) ===")
        else:
            cond_tensor = None
            out_dir = args.out_dir

        for i in range(args.num_samples):
            shape = (1, 2, 129, 375) if is_phase_ckpt else (1, 1, 129, 376)
            spec = sample(
                model, diff, shape, device,
                is_phase=is_phase_ckpt, conditional=args.conditional, cond=cond_tensor,
                norm_type=norm_type, min_db=min_db, max_db=max_db,
                db_mu=db_mu, db_sigma=db_sigma, phase_mode=phase_mode
            )

            np.save(f"{out_dir}/sample_{i+1:03d}.npy", spec)
            if is_phase_ckpt:
                plt.imsave(f"{out_dir}/sample_{i+1:03d}.png", spec, cmap="jet")
            else:
                plt.imsave(f"{out_dir}/sample_{i+1:03d}.png", 20*np.log10(spec + 1e-8), cmap="jet")
        print(f"✅ Saved {args.num_samples} samples to {out_dir}")

# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--is_phase", type=str2bool, default=False, help="(fallback) phase 여부")
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="gen_samples")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--condition", type=str, default=None)
    p.add_argument("--all_conditions", action="store_true")
    args = p.parse_args()
    main(args)