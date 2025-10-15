import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.signal import istft


# ==============================
# Utility
# ==============================
def load_pairs(gt_dir, recon_dir, prefix):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])
    recon_files = sorted([f for f in os.listdir(recon_dir) if f.startswith(prefix) and f.endswith(".npy")])
    pairs = []
    for gfile, rfile in zip(gt_files, recon_files):
        gt = np.load(os.path.join(gt_dir, gfile), allow_pickle=True)
        recon = np.load(os.path.join(recon_dir, rfile), allow_pickle=True)
        if gt.shape != recon.shape:
            min_shape = tuple(min(a, b) for a, b in zip(gt.shape, recon.shape))
            gt = gt[tuple(slice(0, s) for s in min_shape)]
            recon = recon[tuple(slice(0, s) for s in min_shape)]
        pairs.append((gfile, gt, recon))
    return pairs


def reconstruct_signal(mag, phase, n_fft=256, hop_length=128):
    """Magnitude와 Phase로부터 1D waveform 복원"""
    spec_complex = mag * np.exp(1j * phase)
    _, signal = istft(spec_complex, nperseg=n_fft, noverlap=n_fft - hop_length)
    return signal


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
def plot_comparison_waveform(gt, recon, name, fname, snr, pcc, out_dir):
    """1D waveform 비교 그래프 저장"""
    plt.figure(figsize=(10, 3))
    plt.plot(gt, label="GT", color="tab:orange", alpha=0.8)
    plt.plot(recon, label=f"Recon ({name})", color="tab:blue", alpha=0.8)
    plt.legend()
    plt.title(f"{fname} | SNR={snr:.2f} dB | PCC={pcc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_{fname}_compare.png"), dpi=200)
    plt.close()


# ==============================
# Evaluation Core
# ==============================
def evaluate_single(gt_dir, recon_dir, out_dir, prefix, name, plot=False, n_fft=256, hop_length=128):
    """signal / mag / phase별 개별 평가"""
    print(f"\n🔹 Evaluating {name.upper()} ...")
    os.makedirs(out_dir, exist_ok=True)
    pairs = load_pairs(gt_dir, recon_dir, prefix)
    if len(pairs) == 0:
        print(f"⚠️ No valid {name} files found.")
        return pd.DataFrame()

    results = []
    for fname, gt, recon in tqdm(pairs, desc=f"Evaluating {name}"):

        # --- ① 기본 비교 (원본 2D 또는 1D) ---
        if name == "signal":
            gt_eval, recon_eval = gt, recon
        elif name in ["mag", "phase"]:
            # --- ② magnitude/phase를 이용해 waveform으로 변환 후 비교 ---
            try:
                gt_wave = reconstruct_signal(gt, np.zeros_like(gt)) if name == "mag" else reconstruct_signal(np.ones_like(gt), gt)
                recon_wave = reconstruct_signal(recon, np.zeros_like(recon)) if name == "mag" else reconstruct_signal(np.ones_like(recon), recon)
                # waveform 기반 평가
                min_len = min(len(gt_wave), len(recon_wave))
                gt_eval, recon_eval = gt_wave[:min_len], recon_wave[:min_len]
            except Exception as e:
                print(f"⚠️ Waveform reconstruction failed for {fname}: {e}")
                gt_eval, recon_eval = np.mean(gt, axis=0), np.mean(recon, axis=0)
        else:
            continue

        # --- Metrics ---
        mse, mae, snr = compute_metrics(gt_eval, recon_eval)
        pcc, cos_sim, ncc = compute_shape_metrics(gt_eval, recon_eval)

        results.append({
            "file": fname,
            "MSE": mse,
            "MAE": mae,
            "SNR(dB)": snr,
            "PCC": pcc,
            "CosineSim": cos_sim,
            "NCC": ncc
        })

        if plot:
            plot_comparison_waveform(gt_eval, recon_eval, name, fname, snr, pcc, out_dir)

    # --- 결과 저장 ---
    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    df.to_csv(os.path.join(out_dir, f"{name}_metrics.csv"), index=False)
    mean_vals = df.select_dtypes(include=[np.number]).mean().to_dict()

    print(f"✅ {name.upper()} Mean Metrics:")
    for k, v in mean_vals.items():
        print(f"{k:10s}: {v:.6f}")

    mean_row = {col: mean_vals.get(col, np.nan) for col in df.columns}
    mean_row["file"] = "mean"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_csv(os.path.join(out_dir, f"{name}_metrics_summary.csv"), index=False)
    return df


def evaluate_combined_signal(gt_signal_dir, recon_dir, out_dir, plot=False, n_fft=256, hop_length=128):
    """GT signal vs (Recon mag+phase로 복원된 signal)"""
    print(f"\n🔹 Evaluating reconstructed signal from mag+phase ...")
    os.makedirs(out_dir, exist_ok=True)
    gt_files = sorted([f for f in os.listdir(gt_signal_dir) if f.endswith(".npy")])
    mag_files = sorted([f for f in os.listdir(recon_dir) if f.startswith("mag_") and f.endswith(".npy")])
    phase_files = sorted([f for f in os.listdir(recon_dir) if f.startswith("phase_") and f.endswith(".npy")])

    results = []
    for idx, (gfile, mfile, pfile) in enumerate(zip(gt_files, mag_files, phase_files)):
        gt = np.load(os.path.join(gt_signal_dir, gfile))
        mag = np.load(os.path.join(recon_dir, mfile))
        phase = np.load(os.path.join(recon_dir, pfile))

        recon_sig = reconstruct_signal(mag, phase, n_fft, hop_length)
        min_len = min(len(gt), len(recon_sig))
        gt, recon_sig = gt[:min_len], recon_sig[:min_len]

        mse, mae, snr = compute_metrics(gt, recon_sig)
        pcc, cos_sim, ncc = compute_shape_metrics(gt, recon_sig)
        results.append({
            "file": gfile,
            "MSE": mse,
            "MAE": mae,
            "SNR(dB)": snr,
            "PCC": pcc,
            "CosineSim": cos_sim,
            "NCC": ncc
        })

        if plot:
            plot_comparison_waveform(gt, recon_sig, "combined_magphase", f"combined_{idx}", snr, pcc, out_dir)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "combined_magphase_metrics.csv"), index=False)
    mean_vals = df.select_dtypes(include=[np.number]).mean().to_dict()
    print(f"✅ COMBINED MAG+PHASE Mean Metrics:")
    for k, v in mean_vals.items():
        print(f"{k:10s}: {v:.6f}")
    return df


# ==============================
# Evaluation Wrapper
# ==============================
def evaluate_all(gt_signal_dir, gt_mag_dir, gt_phase_dir, recon_dir, out_dir="eval_results", plot=False,
                 n_fft=256, hop_length=128):
    os.makedirs(out_dir, exist_ok=True)
    evaluate_single(gt_signal_dir, recon_dir, out_dir, "signal_", "signal", plot, n_fft, hop_length)
    evaluate_single(gt_mag_dir, recon_dir, out_dir, "mag_", "mag", plot, n_fft, hop_length)
    evaluate_single(gt_phase_dir, recon_dir, out_dir, "phase_", "phase", plot, n_fft, hop_length)
    evaluate_combined_signal(gt_signal_dir, recon_dir, out_dir, plot, n_fft, hop_length)
    print(f"\n📁 Evaluation complete. Results saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate signal/mag/phase and combined (mag+phase→signal)")
    parser.add_argument("--gt_signal_dir", type=str, required=True)
    parser.add_argument("--gt_mag_dir", type=str, required=True)
    parser.add_argument("--gt_phase_dir", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=128)
    args = parser.parse_args()

    evaluate_all(args.gt_signal_dir, args.gt_mag_dir, args.gt_phase_dir, args.recon_dir,
                 args.out_dir, args.plot, args.n_fft, args.hop_length)