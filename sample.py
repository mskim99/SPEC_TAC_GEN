import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import UNet, Diffusion, str2bool

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use

# ===============================
# Model Loader
# ===============================
def load_model(ckpt_path, device="cuda",
               base_ch=32, timesteps=300,
               conditional=False, num_classes=0,
               is_phase=False):
    """체크포인트 로드 및 모델 초기화"""
    in_ch = 2 if is_phase else 1
    out_ch = 2 if is_phase else 1

    model = UNet(in_ch, out_ch, base_ch, [1, 2, 4, 8], conditional, num_classes).to(device)
    diffusion = Diffusion(timesteps, device)
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")
    return model, diffusion, ckpt


# ===============================
# Sampling
# ===============================
@torch.no_grad()
def sample(model, diffusion, shape, device,
           is_phase, phase_mode="residual",
           n_fft=256, hop=128,
           norm_type="zscore",
           min_db=-80, max_db=0,
           db_mu=None, db_sigma=None):
    """
    Diffusion Sampling
    - Phase: residual 복원
    - Magnitude: inverse normalization
    """
    img = torch.randn(shape, device=device)

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        pred = model(img, t_tensor)
        a = diffusion.alphas[t]
        ah = diffusion.alpha_hat[t]
        b = diffusion.betas[t]
        noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
        img = 1 / torch.sqrt(a) * (img - ((1 - a) / torch.sqrt(1 - ah)) * pred) + torch.sqrt(b) * noise

    img = img.squeeze().cpu().numpy()

    # -------------------
    # Phase Residual 복원
    # -------------------
    if is_phase:
        if phase_mode != "residual":
            raise ValueError("Checkpoint is not residual-phase mode.")

        sin_out, cos_out = img[0], img[1]
        sin_out = np.clip(sin_out, -1.0, 1.0)
        cos_out = np.clip(cos_out, -1.0, 1.0)

        # 단위 원 정규화
        norm = np.sqrt(sin_out**2 + cos_out**2 + 1e-8)
        sin_out /= norm
        cos_out /= norm

        # Residual → Absolute Phase 복원
        resid = np.arctan2(sin_out, cos_out)
        F = resid.shape[0]
        k = np.arange(F)
        expected_adv = 2 * np.pi * hop * k / float(n_fft)
        inc = resid + expected_adv[:, None]

        phi = np.cumsum(inc, axis=1)
        phi0 = np.zeros((F, 1), dtype=np.float32)
        phase = np.concatenate([phi0, phi], axis=1)
        phase = np.angle(np.exp(1j * phase))
        return phase

    # -------------------
    # Magnitude 복원
    # -------------------
    else:
        if norm_type == "zscore":
            assert db_mu is not None and db_sigma is not None, "zscore reverse requires db_mu/db_sigma"
            spec_db = img * db_sigma + db_mu
        elif norm_type == "0to1":
            spec_db = img * (max_db - min_db) + min_db
        else:
            spec_db = (img + 1.0) / 2.0 * (max_db - min_db) + min_db

        spec_mag = 10 ** (spec_db / 20.0)
        return spec_mag


# ===============================
# Main
# ===============================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 조건 로드
    num_classes = 0
    cond_map = None
    if args.conditional:
        cond_path = os.path.join(os.path.dirname(args.ckpt), "cond2idx.json")
        with open(cond_path, "r") as f:
            cond_map = json.load(f)
        num_classes = len(cond_map)

    # 체크포인트 및 메타 정보 로드
    ckpt = torch.load(args.ckpt, map_location="cpu")
    is_phase_ckpt = ckpt.get("is_phase", args.is_phase)
    phase_mode = ckpt.get("phase_mode", "residual")
    n_fft = int(ckpt.get("n_fft", 256))
    hop = int(ckpt.get("hop_length", 128))

    model, diffusion, _ = load_model(
        args.ckpt, device, args.base_ch, args.timesteps,
        args.conditional, num_classes, is_phase=is_phase_ckpt
    )

    # 정규화 정보
    norm_type = ckpt.get("norm_type", "zscore")
    min_db = ckpt.get("min_db", -80.0)
    max_db = ckpt.get("max_db", 0.0)
    db_mu = ckpt.get("db_mu", None)
    db_sigma = ckpt.get("db_sigma", None)

    print(f"✅ Sampling config: norm={norm_type}, phase_mode={phase_mode}")

    # -----------------------
    # 여러 개 샘플 생성
    # -----------------------
    shape = (1, 2, 129, 375) if is_phase_ckpt else (1, 1, 129, 376)

    for i in range(args.num_samples):
        print(f"🎨 Generating sample {i+1}/{args.num_samples} ...")
        sample_data = sample(
            model, diffusion, shape, device,
            is_phase=is_phase_ckpt, phase_mode=phase_mode,
            n_fft=n_fft, hop=hop,
            norm_type=norm_type, min_db=min_db, max_db=max_db,
            db_mu=db_mu, db_sigma=db_sigma
        )

        np.save(f"{args.out_dir}/sample_{i}.npy", sample_data)
        plt.imsave(f"{args.out_dir}/sample_{i}.png", sample_data, cmap="jet")

    print(f"✅ Saved {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--out_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--is_phase", type=str2bool, default=True)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")  # ✅ 추가
    args = parser.parse_args()
    main(args)