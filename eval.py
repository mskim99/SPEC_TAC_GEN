import os
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10
from glob import glob
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from tqdm import tqdm
from openpyxl import Workbook

try:
    import torch
except Exception:
    torch = None

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def _per_scale_z(x):
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / std

# === 기본 지표 ===
def mse(gt, pred): return np.mean((gt - pred) ** 2)
def mae(gt, pred): return np.mean(np.abs(gt - pred))
def psnr(gt, pred):
    mse_val = mse(gt, pred)
    if mse_val == 0:
        return float("inf")
    data_range = float(np.max(gt) - np.min(gt) + 1e-12)
    return 20.0 * log10(data_range) - 10.0 * log10(mse_val)

def cosine_similarity(gt, pred): return 1 - cosine(gt.flatten(), pred.flatten())
def correlation(gt, pred): return pearsonr(gt.flatten(), pred.flatten())[0]
def spectral_convergence(gt, pred, eps=1e-8):
    G, P = np.abs(gt), np.abs(pred)
    num = np.linalg.norm(G - P, 'fro')
    den = np.linalg.norm(G, 'fro') + eps
    return num / den

def log_spectral_distance(gt, pred, eps=1e-8):
    G, P = np.log(np.abs(gt) + eps), np.log(np.abs(pred) + eps)
    return float(np.sqrt(np.mean((G - P) ** 2)))

def circular_mse(gt, pred):
    diff = gt - pred
    return np.mean(1 - np.cos(diff))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Denorm helper ===
def _denorm_if_needed(x, stats):
    if not stats:
        return x
    mu = stats.get("mu", None); std = stats.get("std", None)
    if mu is None or std is None:
        return x
    if x.ndim >= 2 and (x.shape[0] == 2):
        x0 = x.copy()
        x0[0] = x0[0] * float(std[0]) + float(mu[0])
        x0[1] = x0[1] * float(std[1]) + float(mu[1])
        # === DEBUG: check denorm result ===
        print(f"[DEBUG] Denorm applied: real(mean={x0[0].mean():.3f}, std={x0[0].std():.3f}), "
              f"imag(mean={x0[1].mean():.3f}, std={x0[1].std():.3f})")
        return x0
    else:
        x = x * float(std[0]) + float(mu[0])
        # print(f"[DEBUG] Denorm applied (single): mean={x.mean():.3f}, std={x.std():.3f}")
        return x

# === Optional preprocessing ===
def affine_fit(gen, gt):
    g = gen.flatten().astype(np.float64)
    t = gt.flatten().astype(np.float64)
    gm, tm = g.mean(), t.mean()
    gv = (g*g).mean() - gm*gm
    cov = (g*t).mean() - gm*tm
    a = cov / (gv + 1e-12) if gv > 1e-12 else 1.0
    b = tm - a * gm
    return (a*gen + b).astype(gen.dtype)

import numpy as np

def complex_affine_fix(gen: np.ndarray,
                       gt: np.ndarray,
                       use_bias: bool = False):
    # 1. 복소형 변환
    G = gen.astype(np.complex128)
    T = gt.astype(np.complex128)

    # 2. 최소제곱 복소 스칼라 계산
    denom = np.sum(np.abs(G)**2) + 1e-12
    a = np.sum(T * np.conj(G)) / denom  # scale + rotation

    # 3. 선택적 bias 보정
    c = np.mean(T - a * G) if use_bias else 0.0

    # 4. 보정 적용
    H = a * G + c

    # 5. 디버깅 출력
    print(f"[DEBUG] complex_affine_fix: |a|={np.abs(a):.4f}, "
          f"angle={np.angle(a):.4f} rad, bias={'ON' if use_bias else 'OFF'}")

    return H.astype(gen.dtype)

# === Core evaluation ===
def evaluate_class(gt_files, gen_files, max_pairs=None, class_name="", is_phase=False,
                   denorm_stats=None, affine_fix=False, complex_affine_fix_flag=False):
    if len(gt_files) == 0 or len(gen_files) == 0:
        return None

    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    cos_list, corr_list, sc_list, lsd_list = [], [], [], []
    circ_list = []

    count = 0
    for gi, gt_path in enumerate(tqdm(gt_files, desc=f"Evaluating {class_name}")):
        gt = np.load(gt_path)
        gt = _per_scale_z(gt)
        for gj, gen_path in enumerate(gen_files):
            gen = np.load(gen_path)

            # ==== DEBUG: print raw stats ====
            '''
            if gi == 0 and gj == 0:
                print(f"[DEBUG] GT raw: mean={gt.mean():.3f}, std={gt.std():.3f}")
                print(f"[DEBUG] GEN raw(before denorm): mean={gen.mean():.3f}, std={gen.std():.3f}")
            '''
            # GEN에도 GT와 동일한 per_scale_z 정규화를 적용합니다.
            gen = _per_scale_z(gen)

            # ---- denorm ----
            # gen = _denorm_if_needed(gen, denorm_stats)

            # ---- complex affine fix ----
            if not is_phase and complex_affine_fix_flag:
                gen = complex_affine_fix(gen, gt)
            elif not is_phase and affine_fix:
                gen = affine_fit(gen, gt)

            # ==== DEBUG: compare GT/GEN stats ====
            # if gi == 0 and gj == 0:
            # diff_std = abs(gt.std() - gen.std())
            # ratio = gt.std() / (gen.std() + 1e-8)
            # print(f"[DEBUG] GT mean={gt.mean():.3f}, std={gt.std():.3f}")
            # print(f"[DEBUG] After denorm/affine: GEN mean={gen.mean():.3f}, std={gen.std():.3f}")
            # print(f"[DEBUG] STD ratio (GT/GEN)={ratio:.3f}, abs diff={diff_std:.3f}")
            # if ratio > 10 or ratio < 0.1:
                # print("⚠️ [WARNING] Possible normalization mismatch between GT and GEN!")

            # ---- Metrics ----
            dr = float(gt.max() - gt.min() + 1e-12)
            if is_phase:
                circ_list.append(circular_mse(gt, gen))
            else:
                mse_list.append(mse(gt, gen))
                psnr_list.append(psnr(gt, gen))
            mae_list.append(mae(gt, gen))
            ssim_list.append(ssim(gt, gen, data_range=dr))
            cos_list.append(cosine_similarity(gt, gen))
            corr_list.append(correlation(gt, gen))
            sc_list.append(spectral_convergence(gt, gen))
            lsd_list.append(log_spectral_distance(gt, gen))
            count += 1
            if max_pairs and count >= max_pairs:
                break
        if max_pairs and count >= max_pairs:
            break

    results = {
        "MAE": np.mean(mae_list),
        "SSIM": np.mean(ssim_list),
        "Cosine": np.mean(cos_list),
        "Correlation": np.mean(corr_list),
        "Spectral Convergence": np.mean(sc_list),
        "Log-Spectral Distance": np.mean(lsd_list),
    }
    if is_phase:
        results["Circular MSE"] = np.mean(circ_list)
    else:
        results["MSE"] = np.mean(mse_list)
        results["PSNR"] = np.mean(psnr_list)
    return results

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--gen_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--is_phase", type=bool, default=False)
    parser.add_argument("--affine_fix", action="store_true")
    parser.add_argument("--complex_affine_fix", action="store_true", help="복소 회전+스케일 보정")
    parser.add_argument("--suffix", type=str, default="_real.npy",
                        help="Suffix for generated files (e.g., '_real.npy', '_imag.npy')")
    args = parser.parse_args()

    # ==== DEBUG: Load ckpt normalization stats ====
    denorm_stats = None
    if args.ckpt and torch is not None and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        mu, std = cfg.get("norm_mu", None), cfg.get("norm_std", None)
        if mu is not None and std is not None:
            denorm_stats = {"mu": mu, "std": std}
            print(f"[DEBUG] Loaded ckpt norm_mu={mu}, norm_std={std}")
        else:
            print("⚠️ [WARNING] norm_mu/std not found in ckpt.cfg")

    gt_files = sorted(glob(os.path.join(args.gt_root, "*.npy")))
    gen_pattern = f"*{args.suffix}"
    gen_files = sorted(glob(os.path.join(args.gen_root, gen_pattern)))
    print(f"[INFO] Evaluating GT ({len(gt_files)} files) vs GEN ({len(gen_files)} files matching '{gen_pattern}')")

    print(f"[DEBUG] Total files: GT={len(gt_files)}, GEN={len(gen_files)}")
    results = evaluate_class(gt_files, gen_files, class_name="debug_eval",
                             denorm_stats=denorm_stats,
                             affine_fix=args.affine_fix,
                             complex_affine_fix_flag=args.complex_affine_fix,
                             is_phase=args.is_phase)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")