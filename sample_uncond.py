import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import math

from main_uncond import UNet, Diffusion   # main.py에 정의된 클래스/함수 불러오기


def load_model(ckpt_path, device="cuda", base_ch=32, timesteps=300):
    # 모델 초기화
    model = UNet(in_ch=1, out_ch=1, base_ch=base_ch).to(device)
    diffusion = Diffusion(timesteps=timesteps, device=device)

    # 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Loaded checkpoint from {ckpt_path}, epoch={ckpt['epoch']}")
    return model, diffusion


# =====================
# 샘플 생성 함수
# =====================
@torch.no_grad()
def sample(model, diffusion, shape=(1, 1, 129, 376),
           device="cuda", min_db=-80, max_db=0, is_phase=False):
    img = torch.randn(shape, device=device)

    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        noise_pred = model(img, t_tensor)
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

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 모델 로드
    model, diffusion = load_model(args.ckpt, device=device,
                                  base_ch=args.base_ch,
                                  timesteps=args.timesteps)

    # 샘플 생성
    for i in range(args.num_samples):
        spec = sample(model, diffusion,
                          shape=(1, 1, 129, 376),
                          device=device,
                          min_db=args.min_db,
                          max_db=args.max_db,
                          is_phase=args.is_phase)

        # 저장 (npy + png)
        np.save(f"{args.out_dir}/sample_{i+1:03d}.npy", spec)
        if args.is_phase:
            plt.imsave(f"{args.out_dir}/sample_{i+1:03d}.png",
                   spec / math.pi * 255.,
                   cmap="jet")
        else:
            plt.imsave(f"{args.out_dir}/sample_{i+1:03d}.png",
                   20*np.log10(spec + 1e-8),
                   cmap="jet")
        print(f"Saved sample {i+1} to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_phase", type=bool, default=False)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="불러올 checkpoint 경로 (.pt)")
    parser.add_argument("--out_dir", type=str, default="gen_samples",
                        help="샘플 저장 디렉토리")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="생성할 샘플 개수")
    parser.add_argument("--timesteps", type=int, default=300,
                        help="Diffusion 타임스텝 수 (훈련과 동일하게)")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="UNet base channel 수 (훈련과 동일하게)")
    parser.add_argument("--min_db", type=float, default=-80,
                        help="Normalize 시 사용한 최소 dB")
    parser.add_argument("--max_db", type=float, default=0,
                        help="Normalize 시 사용한 최대 dB")
    args = parser.parse_args()

    main(args)