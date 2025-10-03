import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import math
import json

from main_cond import UNet, Diffusion   # main_cond.py에 정의된 UNet, Diffusion 불러오기


# ----------------------------
# 모델 로드 함수
# ----------------------------
def load_model(ckpt_path, device="cuda", base_ch=32, timesteps=300, num_classes=10):
    model = UNet(in_ch=1, out_ch=1, base_ch=base_ch, num_classes=num_classes).to(device)
    diffusion = Diffusion(timesteps=timesteps, device=device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded checkpoint from {ckpt_path}, epoch={ckpt['epoch']}, num_classes={num_classes}")
    return model, diffusion, ckpt


# ----------------------------
# 조건부 샘플 생성 함수
# ----------------------------
@torch.no_grad()
def sample_cond(model, diffusion, cond, shape=(1, 1, 129, 376),
                device="cuda", min_db=-80, max_db=0, is_phase=False):
    img = torch.randn(shape, device=device)

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        cond_tensor = torch.tensor([cond], device=device).long()
        noise_pred = model(img, t_tensor, cond_tensor)  # ✅ 조건부 호출
        alpha = diffusion.alphas[t]
        alpha_hat = diffusion.alpha_hat[t]
        beta = diffusion.betas[t]

        noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
        img = 1/torch.sqrt(alpha) * (img - ((1-alpha)/torch.sqrt(1-alpha_hat))*noise_pred) + torch.sqrt(beta)*noise

    # === 역방향 변환 ===
    img = img.squeeze().cpu().numpy()

    if is_phase:
        img = img * math.pi
    else:
        # (1) [-1,1] → [min_db, max_db]
        spec_db = (img + 1) / 2 * (max_db - min_db) + min_db
        # (2) dB → magnitude
        img = 10 ** (spec_db / 20)

    return img


# ----------------------------
# 메인 실행
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # === checkpoint 폴더에서 cond2idx.json 로드 ===
    ckpt_dir = os.path.dirname(args.ckpt)
    cond_map_path = os.path.join(ckpt_dir, "cond2idx.json")
    if not os.path.exists(cond_map_path):
        raise FileNotFoundError(f"⚠️ cond2idx.json not found in {ckpt_dir}")
    with open(cond_map_path, "r") as f:
        cond_map = json.load(f)

    num_classes = len(cond_map)
    model, diffusion, ckpt = load_model(
        args.ckpt,
        device=device,
        base_ch=args.base_ch,
        timesteps=args.timesteps,
        num_classes=num_classes
    )

    # === 전체 조건 생성 vs 특정 조건 생성 ===
    if args.all_conditions:
        conditions_to_generate = cond_map.keys()
    else:
        if args.condition not in cond_map:
            raise ValueError(f"⚠️ condition '{args.condition}' not found in {cond_map_path}")
        conditions_to_generate = [args.condition]

    # === 각 condition별 샘플 생성 ===
    for cond_name in conditions_to_generate:
        cond_idx = cond_map[cond_name]
        cond_out_dir = os.path.join(args.out_dir, cond_name)
        os.makedirs(cond_out_dir, exist_ok=True)

        print(f"\n=== Generating for condition: {cond_name} (idx={cond_idx}) ===")
        for i in range(args.num_samples):
            spec = sample_cond(model, diffusion, cond=cond_idx,
                               shape=(1, 1, 129, 376),
                               device=device,
                               min_db=args.min_db,
                               max_db=args.max_db,
                               is_phase=args.is_phase)

            # 저장 (npy + png)
            np.save(f"{cond_out_dir}/sample_{i+1:03d}.npy", spec)
            if args.is_phase:
                plt.imsave(f"{cond_out_dir}/sample_{i+1:03d}.png",
                           spec / math.pi * 255.,
                           cmap="jet")
            else:
                plt.imsave(f"{cond_out_dir}/sample_{i+1:03d}.png",
                           20*np.log10(spec + 1e-8),
                           cmap="jet")
        print(f"✅ Saved {args.num_samples} samples to {cond_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_phase", type=bool, default=False)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="불러올 checkpoint 경로 (.pt)")
    parser.add_argument("--out_dir", type=str, default="gen_samples",
                        help="샘플 저장 디렉토리")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="각 condition당 생성할 샘플 개수")
    parser.add_argument("--timesteps", type=int, default=300,
                        help="Diffusion 타임스텝 수 (훈련과 동일하게)")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="UNet base channel 수 (훈련과 동일하게)")
    parser.add_argument("--min_db", type=float, default=-80,
                        help="Normalize 시 사용한 최소 dB")
    parser.add_argument("--max_db", type=float, default=0,
                        help="Normalize 시 사용한 최대 dB")

    # === 조건 관련 ===
    parser.add_argument("--condition", type=str, default=None,
                        help="특정 condition 이름 (예: G1EpoxyRasterPlate)")
    parser.add_argument("--all_conditions", action="store_true",
                        help="모든 condition에 대해 샘플 생성")

    args = parser.parse_args()
    main(args)