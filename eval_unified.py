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


# === 기본 지표 ===
def mse(gt, pred): return np.mean((gt - pred) ** 2)
def mae(gt, pred): return np.mean(np.abs(gt - pred))
def psnr(gt, pred):
    mse_val = mse(gt, pred)
    if mse_val == 0: return float("inf")
    max_val = np.max(gt)
    return 20 * log10(max_val / np.sqrt(mse_val))
def cosine_similarity(gt, pred): return 1 - cosine(gt.flatten(), pred.flatten())
def correlation(gt, pred): return pearsonr(gt.flatten(), pred.flatten())[0]
def spectral_convergence(gt, pred): return np.linalg.norm(gt - pred, 'fro') / (np.linalg.norm(gt, 'fro') + 1e-8)
def log_spectral_distance(gt, pred):
    eps = 1e-8
    gt_log = 10 * np.log10(gt + eps)
    pred_log = 10 * np.log10(pred + eps)
    return np.mean(np.sqrt(np.mean((gt_log - pred_log) ** 2, axis=-1)))


# === 평가 함수 (조건부) ===
def evaluate_class(gt_files, gen_files, max_pairs=None, class_name=""):
    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    cos_list, corr_list, sc_list, lsd_list = [], [], [], []

    total_pairs = len(gt_files) * len(gen_files)
    if max_pairs is not None:
        total_pairs = min(total_pairs, max_pairs)

    count = 0
    for gt_path in tqdm(gt_files, desc=f"Evaluating {class_name}"):
        gt = np.load(gt_path)
        for gen_path in gen_files:
            gen = np.load(gen_path)

            if gt.shape != gen.shape:
                min_shape = (min(gt.shape[0], gen.shape[0]),
                             min(gt.shape[1], gen.shape[1]))
                gt_tmp, gen_tmp = gt[:min_shape[0], :min_shape[1]], gen[:min_shape[0], :min_shape[1]]
            else:
                gt_tmp, gen_tmp = gt, gen

            mse_list.append(mse(gt_tmp, gen_tmp))
            mae_list.append(mae(gt_tmp, gen_tmp))
            psnr_list.append(psnr(gt_tmp, gen_tmp))
            ssim_list.append(ssim(gt_tmp, gen_tmp, data_range=gen_tmp.max() - gen_tmp.min()))
            cos_list.append(cosine_similarity(gt_tmp, gen_tmp))
            corr_list.append(correlation(gt_tmp, gen_tmp))
            sc_list.append(spectral_convergence(gt_tmp, gen_tmp))
            lsd_list.append(log_spectral_distance(gt_tmp, gen_tmp))

            count += 1
            if max_pairs is not None and count >= max_pairs:
                break
        if max_pairs is not None and count >= max_pairs:
            break

    return {
        "MSE": np.mean(mse_list),
        "MAE": np.mean(mae_list),
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "Cosine": np.mean(cos_list),
        "Correlation": np.mean(corr_list),
        "Spectral Convergence": np.mean(sc_list),
        "Log-Spectral Distance": np.mean(lsd_list),
    }


# === 평가 함수 (비조건부) ===
def evaluate_uncond(gt_dir, gen_dir, max_pairs=None):
    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    gen_files = sorted(glob(os.path.join(gen_dir, "*.npy")))

    if len(gt_files) == 0 or len(gen_files) == 0:
        raise ValueError("GT 또는 생성 샘플 폴더에 파일이 없습니다.")

    return evaluate_class(gt_files, gen_files, max_pairs=max_pairs, class_name="uncond")


# === 결과 저장 ===
def save_results_to_excel(results, out_path="evaluation_results.xlsx", conditional=False):
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    if conditional:
        header = ["Class"] + list(next(iter(results.values())).keys())
        ws.append(header)
        for cls, res in results.items():
            row = [cls] + [round(v, 6) for v in res.values()]
            ws.append(row)
        avg_results = {metric: np.mean([res[metric] for res in results.values()])
                       for metric in next(iter(results.values())).keys()}
        ws.append([])
        ws.append(["AVERAGE"] + [round(v, 6) for v in avg_results.values()])
    else:
        ws.append(list(results.keys()))
        ws.append([round(v, 6) for v in results.values()])

    wb.save(out_path)
    print(f"✅ Results saved to {out_path}")


# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", action="store_true", help="조건부 평가 여부")
    parser.add_argument("--gt_root", type=str, required=True, help="GT 데이터 경로")
    parser.add_argument("--gen_root", type=str, required=True, help="생성 샘플 경로")
    parser.add_argument("--max_pairs", type=int, default=None, help="평가할 최대 pair 수")
    parser.add_argument("--out_excel", type=str, default="evaluation_results.xlsx", help="엑셀 결과 저장 경로")
    args = parser.parse_args()

    if args.conditional:
        # GT prefix별 그룹핑
        gt_files = glob(os.path.join(args.gt_root, "*.npy"))
        gt_by_class = {}
        for f in gt_files:
            cls = os.path.basename(f).split("_")[0]
            gt_by_class.setdefault(cls, []).append(f)

        all_results = {}
        for cls in os.listdir(args.gen_root):
            gen_dir = os.path.join(args.gen_root, cls)
            if not os.path.isdir(gen_dir): continue
            gen_files = sorted(glob(os.path.join(gen_dir, "*.npy")))
            if cls not in gt_by_class:
                print(f"⚠️ Warning: GT for class {cls} not found, skipping.")
                continue

            res = evaluate_class(gt_by_class[cls], gen_files,
                                 max_pairs=args.max_pairs, class_name=cls)
            if res:
                all_results[cls] = res
                print(f"Class {cls} results:", res)

        if all_results:
            avg_results = {metric: np.mean([res[metric] for res in all_results.values()])
                           for metric in next(iter(all_results.values())).keys()}
            print("\n=== Average over all classes ===")
            for k, v in avg_results.items():
                print(f"{k}: {v:.4f}")
            save_results_to_excel(all_results, args.out_excel, conditional=True)
        else:
            print("⚠️ No results to evaluate.")
    else:
        results = evaluate_uncond(args.gt_root, args.gen_root, max_pairs=args.max_pairs)
        print("Evaluation Results (Unconditional):")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        save_results_to_excel(results, args.out_excel, conditional=False)