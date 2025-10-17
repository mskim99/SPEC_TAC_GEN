import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ==============================
# Normalization
# ==============================
def normalize_minus1_to1(x):
    """[-1, 1] 정규화"""
    x_min, x_max = np.min(x), np.max(x)
    if np.isclose(x_max, x_min):
        return np.zeros_like(x)
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    x_norm = x_norm * 2 - 1
    return x_norm


# ==============================
# Metrics
# ==============================
def compute_metrics(gt, recon):
    mse = np.mean((gt - recon) ** 2)
    mae = np.mean(np.abs(gt - recon))
    snr = np.nan
    if np.sum(gt ** 2) > 0 and np.sum((gt - recon) ** 2) > 0:
        snr = 10 * np.log10(np.sum(gt ** 2) / np.sum((gt - recon) ** 2))
    return mse, mae, snr


def compute_shape_metrics(gt, recon):
    gt_flat, recon_flat = gt.flatten(), recon.flatten()
    gt_z = (gt_flat - np.mean(gt_flat)) / (np.std(gt_flat) + 1e-8)
    recon_z = (recon_flat - np.mean(recon_flat)) / (np.std(recon_flat) + 1e-8)
    pcc = np.corrcoef(gt_flat, recon_flat)[0, 1]
    cos_sim = np.dot(gt_flat, recon_flat) / (np.linalg.norm(gt_flat) * np.linalg.norm(recon_flat) + 1e-8)
    ncc = np.sum(gt_z * recon_z) / len(gt_z)
    return pcc, cos_sim, ncc


# ==============================
# Visualization
# ==============================
def plot_comparison_waveform(gt, recon, idx, snr, pcc, out_dir):
    """1D waveform 비교 그래프 저장"""
    plt.figure(figsize=(10, 3))
    plt.plot(gt, label="GT", color="tab:orange", alpha=0.8)
    plt.plot(recon, label="Recon", color="tab:blue", alpha=0.8)
    plt.legend()
    plt.title(f"Index {idx} | SNR={snr:.2f} dB | PCC={pcc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pair_{idx:04d}_compare.png"), dpi=200)
    plt.close()


# ==============================
# Evaluation Core
# ==============================
def evaluate_by_index(gt_dir, recon_dir, out_dir, plot=False):
    """두 폴더 내 시계열을 인덱스 기준으로 1:1 비교"""
    print(f"\n🔹 Evaluating signals by index:\n GT: {gt_dir}\n RECON: {recon_dir}")
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])
    recon_files = sorted([f for f in os.listdir(recon_dir) if f.endswith(".npy")])

    n_pairs = min(len(gt_files), len(recon_files))
    if n_pairs == 0:
        print("⚠️ 비교할 .npy 파일이 없습니다.")
        return pd.DataFrame()

    print(f"총 {n_pairs}개의 파일 쌍을 인덱스 기반으로 비교합니다.")
    results = []

    for idx in tqdm(range(n_pairs), desc="Comparing signals"):
        gt_path = os.path.join(gt_dir, gt_files[idx])
        recon_path = os.path.join(recon_dir, recon_files[idx])

        gt = np.load(gt_path, allow_pickle=True)
        recon = np.load(recon_path, allow_pickle=True)

        # 🔹 GT를 [-1, 1]로 정규화
        gt = normalize_minus1_to1(gt)

        # shape 불일치 시 최소 크기 맞추기
        if gt.shape != recon.shape:
            min_shape = tuple(min(a, b) for a, b in zip(gt.shape, recon.shape))
            gt = gt[tuple(slice(0, s) for s in min_shape)]
            recon = recon[tuple(slice(0, s) for s in min_shape)]

        # Metrics 계산
        mse, mae, snr = compute_metrics(gt, recon)
        pcc, cos_sim, ncc = compute_shape_metrics(gt, recon)

        results.append({
            "gt_file": gt_files[idx],
            "recon_file": recon_files[idx],
            "MSE": mse,
            "MAE": mae,
            "SNR(dB)": snr,
            "PCC": pcc,
            "CosineSim": cos_sim,
            "NCC": ncc
        })

        if plot:
            plot_comparison_waveform(gt, recon, idx, snr, pcc, out_dir)

    df = pd.DataFrame(results)

    # 평균 행 추가
    mean_vals = df.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row = {col: mean_vals.get(col, np.nan) for col in df.columns}
    mean_row["index"] = "mean"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    df.to_csv(os.path.join(out_dir, "signal_comparison_by_index.csv"), index=False)

    print("\n✅ Mean Metrics:")
    for k, v in mean_vals.items():
        print(f"{k:10s}: {v:.6f}")

    print(f"\n📁 Evaluation complete. Results saved to {out_dir}")
    return df


# ==============================
# CLI Entry
# ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare .npy signals between two folders by index order")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground-truth 시계열 폴더 경로")
    parser.add_argument("--recon_dir", type=str, required=True, help="복원된 시계열 폴더 경로")
    parser.add_argument("--out_dir", type=str, default="eval_results_index", help="결과 저장 폴더")
    parser.add_argument("--plot", action="store_true", help="비교 그래프 저장 여부")
    args = parser.parse_args()

    evaluate_by_index(args.gt_dir, args.recon_dir, args.out_dir, args.plot)