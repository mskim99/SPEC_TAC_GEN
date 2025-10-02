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
def mse(gt, pred):
    return np.mean((gt - pred) ** 2)

def mae(gt, pred):
    return np.mean(np.abs(gt - pred))

def psnr(gt, pred):
    mse_val = mse(gt, pred)
    if mse_val == 0:
        return float("inf")
    max_val = np.max(gt)
    return 20 * log10(max_val / np.sqrt(mse_val))

# === 추가 지표 ===
def cosine_similarity(gt, pred):
    return 1 - cosine(gt.flatten(), pred.flatten())

def correlation(gt, pred):
    r, _ = pearsonr(gt.flatten(), pred.flatten())
    return r

def spectral_convergence(gt, pred):
    return np.linalg.norm(gt - pred, 'fro') / (np.linalg.norm(gt, 'fro') + 1e-8)

def log_spectral_distance(gt, pred):
    eps = 1e-8
    gt_log = 10 * np.log10(gt + eps)
    pred_log = 10 * np.log10(pred + eps)
    return np.mean(np.sqrt(np.mean((gt_log - pred_log) ** 2, axis=-1)))


# === 평가 함수 (Cross-matching 지원) ===
def evaluate(gt_dir, gen_dir, max_pairs=None):
    gt_files = sorted(glob(os.path.join(gt_dir, "*.npy")))
    gen_files = sorted(glob(os.path.join(gen_dir, "*.npy")))

    if len(gt_files) == 0 or len(gen_files) == 0:
        raise ValueError("GT 또는 생성 샘플 폴더에 파일이 없습니다.")

    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    cos_list, corr_list, sc_list, lsd_list = [], [], [], []

    total_pairs = len(gt_files) * len(gen_files)
    if max_pairs is not None:
        total_pairs = min(total_pairs, max_pairs)

    count = 0
    for gt_path in tqdm(gt_files, desc="Evaluating GT files"):
        gt = np.load(gt_path)

        for gen_path in gen_files:
            gen = np.load(gen_path)

            # 크기 맞추기
            if gt.shape != gen.shape:
                min_shape = (min(gt.shape[0], gen.shape[0]),
                             min(gt.shape[1], gen.shape[1]))
                gt_tmp = gt[:min_shape[0], :min_shape[1]]
                gen_tmp = gen[:min_shape[0], :min_shape[1]]
            else:
                gt_tmp, gen_tmp = gt, gen

            # === 지표 계산 ===
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

    results = {
        "MSE": np.mean(mse_list),
        "MAE": np.mean(mae_list),
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "Cosine": np.mean(cos_list),
        "Correlation": np.mean(corr_list),
        "Spectral Convergence": np.mean(sc_list),
        "Log-Spectral Distance": np.mean(lsd_list),
    }
    return results


def save_results_to_excel(results, out_path="evaluation_results.xlsx"):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    # 첫 번째 행: Metric 이름
    ws.append(list(results.keys()))

    # 두 번째 행: 값
    ws.append([round(v, 6) for v in results.values()])

    wb.save(out_path)
    print(f"✅ Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="GT 스펙트로그램 npy 파일 경로")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="생성된 스펙트로그램 npy 파일 경로")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="평가할 최대 pair 수 (계산량 절감용)")
    parser.add_argument("--out_excel", type=str, default="evaluation_results.xlsx",
                        help="엑셀 결과 저장 경로")
    args = parser.parse_args()

    results = evaluate(args.gt_dir, args.gen_dir, max_pairs=args.max_pairs)

    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    save_results_to_excel(results, args.out_excel)