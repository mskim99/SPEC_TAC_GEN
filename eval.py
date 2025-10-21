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

# ==== NEW: optional torch import for ckpt stats ====
try:
    import torch
except Exception:
    torch = None
# ==== /NEW

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("Expected boolean value.")

# === 기본 지표 ===
def mse(gt, pred): return np.mean((gt - pred) ** 2)
def mae(gt, pred): return np.mean(np.abs(gt - pred))

# ==== CHANGED: PSNR uses GT data_range (peak-to-peak) ====
def psnr(gt, pred):
    mse_val = mse(gt, pred)
    if mse_val == 0:
        return float("inf")
    data_range = float(np.max(gt) - np.min(gt) + 1e-12)  # GT 기준
    return 20.0 * log10(data_range) - 10.0 * log10(mse_val)

def cosine_similarity(gt, pred): return 1 - cosine(gt.flatten(), pred.flatten())
def correlation(gt, pred): return pearsonr(gt.flatten(), pred.flatten())[0]

# ==== CHANGED: spectral metrics with |·| and epsilon ====
def spectral_convergence(gt, pred, eps=1e-8):
    G = np.abs(gt)
    P = np.abs(pred)
    num = np.linalg.norm(G - P, 'fro')
    den = np.linalg.norm(G, 'fro') + eps
    return num / den

def log_spectral_distance(gt, pred, eps=1e-8):
    G = np.log(np.abs(gt) + eps)
    P = np.log(np.abs(pred) + eps)
    return float(np.sqrt(np.mean((G - P) ** 2)))

# === ✅ circular metric (phase) ===
def circular_mse(gt, pred):
    diff = gt - pred
    return np.mean(1 - np.cos(diff))

# === 평가 함수 ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== NEW: denorm helper (from ckpt or CLI) ====
def _denorm_if_needed(x, stats):
    if not stats:
        return x
    mu = stats.get("mu", None); std = stats.get("std", None)
    if mu is None or std is None:
        return x
    if x.ndim >= 2 and (x.shape[0] == 2):  # (2,H,W)
        x0 = x.copy()
        x0[0] = x0[0] * float(std[0]) + float(mu[0])
        x0[1] = x0[1] * float(std[1]) + float(mu[1])
        return x0
    else:
        return x * float(std[0]) + float(mu[0])

# ==== NEW: alignment helpers (옵션 플래그용) ====
def affine_fit(gen, gt):
    """최소제곱으로 a*gen + b가 gt에 맞도록 보정"""
    g = gen.flatten().astype(np.float64)
    t = gt.flatten().astype(np.float64)
    gm, tm = g.mean(), t.mean()
    gv = (g*g).mean() - gm*gm
    cov = (g*t).mean() - gm*tm
    if gv <= 1e-12:
        a = 1.0
    else:
        a = cov / (gv + 1e-12)
    b = tm - a * gm
    return (a*gen + b).astype(gen.dtype)

def best_flip_rotate(gen, gt, data_range):
    """8가지 대칭변환 중 SSIM이 최대인 gen 반환"""
    cands = [
        gen,
        np.fliplr(gen), np.flipud(gen), np.flipud(np.fliplr(gen)),
        np.rot90(gen, 1), np.rot90(gen, 2), np.rot90(gen, 3),
        np.rot90(np.fliplr(gen), 1)
    ]
    best = (None, -1.0)
    for g in cands:
        try:
            s = ssim(gt, g, data_range=float(data_range))
        except Exception:
            s = -1.0
        if s > best[1]:
            best = (g, s)
    return best[0]

def best_shift(gen, gt, max_shift=16):
    """[-max_shift, max_shift] 평행이동 중 SSIM 최대 gen 반환"""
    h, w = gt.shape[:2]
    best_s = -1.0
    best_g = gen
    best_move = (0, 0)
    dr = float(gt.max() - gt.min() + 1e-12)
    for dy in range(-max_shift, max_shift+1):
        g_y = np.roll(gen, dy, axis=0)
        for dx in range(-max_shift, max_shift+1):
            g = np.roll(g_y, dx, axis=1)
            try:
                s = ssim(gt, g, data_range=dr)
            except Exception:
                s = -1.0
            if s > best_s:
                best_s, best_g, best_move = s, g, (dy, dx)
    return best_g, best_move
# ==== /NEW

def evaluate_class(gt_files, gen_files, max_pairs=None, class_name="", is_phase=False, save_diff=True,
                   denorm_stats=None,
                   affine_fix=False, best_flip=False, best_shift_flag=False, max_shift=16):
    if len(gt_files) == 0 or len(gen_files) == 0:
        return None

    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    cos_list, corr_list, sc_list, lsd_list = [], [], [], []
    circ_list = []

    diff_out_dir = os.path.join("evaluation_diffs", class_name)
    if save_diff:
        os.makedirs(diff_out_dir, exist_ok=True)

    total_pairs = len(gt_files) * len(gen_files)
    if max_pairs is not None:
        total_pairs = min(total_pairs, max_pairs)

    count = 0
    for gi, gt_path in enumerate(tqdm(gt_files, desc="Evaluating {}".format(class_name))):
        gt = np.load(gt_path)

        for gj, gen_path in enumerate(gen_files):
            gen = np.load(gen_path)

            # ---- denorm on generated ----
            gen = _denorm_if_needed(gen, denorm_stats)

            # --- Shape align (공통 크기) ---
            if gt.shape != gen.shape:
                min_shape = (min(gt.shape[0], gen.shape[0]),
                             min(gt.shape[1], gen.shape[1]))
                gt_tmp, gen_tmp = gt[:min_shape[0], :min_shape[1]], gen[:min_shape[0], :min_shape[1]]
            else:
                gt_tmp, gen_tmp = gt, gen

            # ==== NEW: optional fixes (평가 전처리) ====
            if not is_phase and affine_fix:
                gen_tmp = affine_fit(gen_tmp, gt_tmp)
            dr = float(gt_tmp.max() - gt_tmp.min() + 1e-12)
            if best_flip:
                gen_tmp = best_flip_rotate(gen_tmp, gt_tmp, data_range=dr)
            if best_shift_flag:
                gen_tmp, _ = best_shift(gen_tmp, gt_tmp, max_shift=max_shift)
            # ==== /NEW

            # --- Compute metrics ---
            if is_phase:
                circ_list.append(circular_mse(gt_tmp, gen_tmp))
            else:
                mse_list.append(mse(gt_tmp, gen_tmp))
                psnr_list.append(psnr(gt_tmp, gen_tmp))

            mae_list.append(mae(gt_tmp, gen_tmp))
            ssim_list.append(ssim(gt_tmp, gen_tmp, data_range=dr))
            cos_list.append(cosine_similarity(gt_tmp, gen_tmp))
            corr_list.append(correlation(gt_tmp, gen_tmp))
            sc_list.append(spectral_convergence(gt_tmp, gen_tmp))
            lsd_list.append(log_spectral_distance(gt_tmp, gen_tmp))

            # --- Visualization (difference image) ---
            if save_diff and count < 100:
                diff = np.abs(gt_tmp - gen_tmp)
                vmax = max(gt_tmp.max(), gen_tmp.max(), diff.max())
                vmin = min(gt_tmp.min(), gen_tmp.min(), diff.min())

                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                for ax, data, title in zip(
                    axes,
                    [gt_tmp, gen_tmp, diff],
                    ["Ground Truth", "Generated", "Difference |Δ|"]
                ):
                    im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(title); ax.axis('off')
                plt.tight_layout()
                fname = os.path.join(diff_out_dir, "diff_gt{:03d}_gen{:03d}.png".format(gi, gj))
                plt.savefig(fname, dpi=200); plt.close(fig)

            count += 1
            if max_pairs is not None and count >= max_pairs:
                break
        if max_pairs is not None and count >= max_pairs:
            break

    # --- Averages ---
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

    if save_diff:
        print("✅ Diff images saved to {}".format(diff_out_dir))
    return results

# === Excel 저장 ===
def save_results_to_excel(results, avg_results, out_path="evaluation_results.xlsx"):
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    header = ["Class"] + list(next(iter(results.values())).keys())
    ws.append(header)
    for cls, res in results.items():
        row = [cls] + [round(v, 6) for v in res.values()]
        ws.append(row)
    ws.append([])
    ws.append(["AVERAGE"] + [round(v, 6) for v in avg_results.values()])
    wb.save(out_path)
    print("✅ Results saved to {}".format(out_path))

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", action="store_true", help="조건부 평가 여부")
    parser.add_argument("--is_phase", type=str2bool, help="phase 데이터 여부 (circular metric 사용)")
    parser.add_argument("--gt_root", type=str, required=True, help="GT 데이터 경로")
    parser.add_argument("--gen_root", type=str, required=True, help="생성 샘플 경로")
    parser.add_argument("--max_pairs", type=int, default=None, help="평가할 최대 pair 수")
    parser.add_argument("--out_excel", type=str, default="evaluation_results.xlsx", help="엑셀 결과 저장 경로")

    # ==== NEW: denorm sources ====
    parser.add_argument("--ckpt", type=str, default=None, help="훈련 ckpt(.pt) 경로 — cfg.norm_mu/std로 denorm")
    parser.add_argument("--denorm_mu", type=float, nargs=2, default=None, help="수동 μ (real, imag)")
    parser.add_argument("--denorm_std", type=float, nargs=2, default=None, help="수동 σ (real, imag)")
    # ==== /NEW

    # ==== NEW: alignment/normalization fix flags ====
    parser.add_argument("--affine_fix", action="store_true", help="a*GEN+b 아핀 보정 후 평가")
    parser.add_argument("--best_flip", action="store_true", help="8가지 flip/rotate 중 SSIM 최대를 사용")
    parser.add_argument("--best_shift", action="store_true", help="작은 평행이동 탐색으로 SSIM 최대를 사용")
    parser.add_argument("--max_shift", type=int, default=16, help="--best_shift 시 최대 이동 픽셀")
    # ==== /NEW

    args = parser.parse_args()

    # ==== NEW: load denorm stats from ckpt or CLI ====
    denorm_stats = {}
    if args.denorm_mu is not None and args.denorm_std is not None:
        denorm_stats = {"mu": args.denorm_mu, "std": args.denorm_std}
        print("[denorm] use CLI stats: {}".format(denorm_stats))
    elif args.ckpt and torch is not None and os.path.isfile(args.ckpt):
        try:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            cfg = ckpt.get("cfg", {})
            mu  = cfg.get("norm_mu", None)
            std = cfg.get("norm_std", None)
            if mu is not None and std is not None:
                denorm_stats = {"mu": mu, "std": std}
                print("[denorm] use ckpt stats: {}".format(denorm_stats))
        except Exception as e:
            print("⚠️ ckpt 로드 실패: {}".format(e))
    else:
        denorm_stats = None
    # ==== /NEW

    if args.conditional:
        gt_files = glob(os.path.join(args.gt_root, "*.npy"))
        gt_by_class = {}
        for f in gt_files:
            cls = os.path.basename(f).split("_")[0]
            gt_by_class.setdefault(cls, []).append(f)

        all_results = {}
        for cls in os.listdir(args.gen_root):
            gen_dir = os.path.join(args.gen_root, cls)
            if not os.path.isdir(gen_dir): continue
            gen_files = sorted(glob(os.path.join(gen_dir, "*_pred.npy")))
            if cls not in gt_by_class:
                print("⚠️ Warning: GT for class {} not found, skipping.".format(cls))
                continue

            res = evaluate_class(gt_by_class[cls], gen_files,
                                 max_pairs=args.max_pairs, class_name=cls,
                                 is_phase=args.is_phase, denorm_stats=denorm_stats,
                                 affine_fix=args.affine_fix, best_flip=args.best_flip,
                                 best_shift_flag=args.best_shift, max_shift=args.max_shift)
            if res:
                all_results[cls] = res
                print("Class {} results: {}".format(cls, res))

        if all_results:
            avg_results = {metric: np.mean([res[metric] for res in all_results.values()])
                           for metric in next(iter(all_results.values())).keys()}
            print("\n=== Average over all classes ===")
            for k, v in avg_results.items():
                print("{}: {:.4f}".format(k, v))
            save_results_to_excel(all_results, avg_results, args.out_excel)
        else:
            print("⚠️ No results to evaluate.")
    else:
        # 비조건부 평가
        gt_files = sorted(glob(os.path.join(args.gt_root, "*.npy")))
        gen_files = sorted(glob(os.path.join(args.gen_root, "*_real.npy")))
        results = evaluate_class(gt_files, gen_files,
                                 max_pairs=args.max_pairs, class_name="uncond",
                                 is_phase=args.is_phase, denorm_stats=denorm_stats,
                                 affine_fix=args.affine_fix, best_flip=args.best_flip,
                                 best_shift_flag=args.best_shift, max_shift=args.max_shift)
        print("Evaluation Results:")
        for k, v in results.items():
            print("{}: {:.4f}".format(k, v))