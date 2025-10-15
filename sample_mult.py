import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import UNet, Diffusion, str2bool

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ----------------------------
# Load model
# ----------------------------
def load_model(ckpt_path, device="cuda", base_ch=32, timesteps=300,
               conditional=False, num_classes=0, is_phase=True):
    """체크포인트 로드 및 모델 초기화"""
    in_ch = 2 if is_phase else 1
    out_ch = 2 if is_phase else 1

    model = UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch,
                 conditional=conditional, num_classes=num_classes).to(device)
    diffusion = Diffusion(timesteps=timesteps, device=device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "epoch": ckpt.get("epoch", -1),
        "phase_mode": ckpt.get("phase_mode", "complex"),
        "re_mu": ckpt.get("re_mu", 0.0),
        "re_std": ckpt.get("re_std", 1.0),
        "im_mu": ckpt.get("im_mu", 0.0),
        "im_std": ckpt.get("im_std", 1.0),
        "db_mu": ckpt.get("db_mu", 0.0),
        "db_sigma": ckpt.get("db_sigma", 1.0),
        "min_db": ckpt.get("min_db", -80.0),
        "max_db": ckpt.get("max_db", 0.0),
        "norm_type": ckpt.get("norm_type", "zscore"),
        "timesteps": ckpt.get("timesteps", timesteps),
    }
    print(f"✅ Loaded ckpt: {ckpt_path} (epoch {meta['epoch']}) | mode={'phase' if is_phase else 'mag'}")
    return model, diffusion, meta


# ----------------------------
# Sampling
# ----------------------------
@torch.no_grad()
def sample_mult(model, diffusion, shape, device, is_phase,
                re_mu=0, re_std=1, im_mu=0, im_std=1,
                db_mu=None, db_sigma=None, min_db=-80, max_db=0,
                norm_type="zscore",
                conditional=False, cond_tensor=None):
    """Diffusion sampling (phase or magnitude)"""
    x = torch.randn(shape, device=device)

    for t in reversed(range(diffusion.timesteps)):
        tt = torch.tensor([t], device=device).long()
        if conditional and cond_tensor is not None:
            eps_hat = model(x, tt, cond_tensor)
        else:
            eps_hat = model(x, tt)

        a = diffusion.alphas[t]
        ah = diffusion.alpha_hat[t]
        b = diffusion.betas[t]
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = 1 / torch.sqrt(a) * (x - ((1 - a) / torch.sqrt(1 - ah)) * eps_hat) + torch.sqrt(b) * z

    x = x.squeeze(0).cpu().numpy()

    if is_phase:
        # Phase 복원 (complex domain)
        re = x[0] * re_std + re_mu
        im = x[1] * im_std + im_mu
        mag = np.sqrt(re ** 2 + im ** 2)
        phs = np.arctan2(im, re)
        return mag, phs
    else:
        # Magnitude 복원
        img = x[0]
        if norm_type == "zscore":
            spec_db = img * db_sigma + db_mu
        elif norm_type == "0to1":
            spec_db = img * (max_db - min_db) + min_db
        else:
            spec_db = (img + 1.0) / 2.0 * (max_db - min_db) + min_db
        spec_mag = 10 ** (spec_db / 20.0)
        return spec_mag, None


# ----------------------------
# Main
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- conditional setup -----
    cond2idx = None
    num_classes = 0
    if args.conditional:
        cond_map_path = os.path.join(os.path.dirname(args.ckpt), "cond2idx.json")
        if not os.path.exists(cond_map_path):
            raise FileNotFoundError(f"cond2idx.json not found in: {os.path.dirname(args.ckpt)}")
        with open(cond_map_path, "r") as f:
            cond2idx = json.load(f)
        num_classes = len(cond2idx)

    # ----- load model -----
    model, diffusion, meta = load_model(
        args.ckpt, device=device, base_ch=args.base_ch, timesteps=args.timesteps,
        conditional=args.conditional, num_classes=num_classes, is_phase=args.is_phase
    )

    diffusion.timesteps = meta.get("timesteps", diffusion.timesteps)

    # ----- condition list -----
    if args.conditional:
        if args.all_conditions:
            cond_names = sorted(cond2idx.keys())
        else:
            if args.condition is None or args.condition not in cond2idx:
                raise ValueError(f"--condition must be one of {list(cond2idx.keys())} or use --all_conditions")
            cond_names = [args.condition]
    else:
        cond_names = [None]

    # ----- sampling -----
    for cname in cond_names:
        cond_tensor = torch.tensor([cond2idx[cname]], device=device).long() if args.conditional else None
        save_dir = os.path.join(args.out_dir, cname) if args.conditional else args.out_dir
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n=== Generating {args.num_samples} sample(s) "
              f"{'for cond='+cname if args.conditional else '(uncond)'} ===")

        for i in range(1, args.num_samples + 1):
            mag, phs = sample_mult(
                model, diffusion,
                shape=(1, 2, 129, 376) if args.is_phase else (1, 1, 129, 376),
                device=device,
                is_phase=args.is_phase,
                re_mu=meta["re_mu"], re_std=meta["re_std"],
                im_mu=meta["im_mu"], im_std=meta["im_std"],
                db_mu=meta["db_mu"], db_sigma=meta["db_sigma"],
                min_db=meta["min_db"], max_db=meta["max_db"],
                norm_type=meta["norm_type"],
                conditional=args.conditional, cond_tensor=cond_tensor
            )

            if args.is_phase:
                np.save(os.path.join(save_dir, f"sample_{i:03d}_mag.npy"), mag)
                np.save(os.path.join(save_dir, f"sample_{i:03d}_phs.npy"), phs)
                plt.imsave(os.path.join(save_dir, f"sample_{i:03d}_mag.png"),
                           20 * np.log10(mag + 1e-8), cmap="jet")
                plt.imsave(os.path.join(save_dir, f"sample_{i:03d}_phs.png"),
                           phs, cmap="twilight")
            else:
                np.save(os.path.join(save_dir, f"sample_{i:03d}_mag.npy"), mag)
                plt.imsave(os.path.join(save_dir, f"sample_{i:03d}_mag.png"),
                           20 * np.log10(mag + 1e-8), cmap="jet")

        print(f"✅ Saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="samples_complex")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--is_phase", type=str2bool, default=True, help="True: phase(complex) mode / False: magnitude mode")

    # conditional
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--all_conditions", action="store_true")

    args = parser.parse_args()
    main(args)