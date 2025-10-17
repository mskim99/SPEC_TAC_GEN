import os
import argparse
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================
# Utility Functions
# ======================
def resize_poly_2d(M, Ht, Wt):
    """2D FIR 기반 리샘플"""
    tmp = resample_poly(M, Ht, M.shape[0], axis=0)
    tmp = resample_poly(tmp, Wt, M.shape[1], axis=1)
    return tmp


def reconstruct_signal_from_coef(real_data, imag_data):
    """wavelet 계수(real, imag)로부터 시계열 복원 (수치 근사)"""
    coef_rec = real_data + 1j * imag_data
    n_scales, _ = coef_rec.shape
    scales = np.arange(1, n_scales + 1)
    da = np.gradient(scales)
    rec_raw = np.sum(np.real(coef_rec) * (da[:, None] / (scales[:, None] ** 1.5)), axis=0)
    rec_raw -= np.mean(rec_raw)
    rec_raw /= np.max(np.abs(rec_raw)) + 1e-12
    return rec_raw


# ======================
# Main
# ======================
def main(args):
    input_dir = args.input_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 🔹 동일 폴더 내에서 _real.npy 쌍 자동 탐색
    real_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_real.npy")])
    print(f"총 {len(real_files)}개 파일 쌍 복원 중...")
    H0, W0 = 1024, 4096

    for fname in tqdm(real_files):
        base = fname.replace("_real.npy", "")
        real_path = os.path.join(input_dir, fname)
        imag_path = os.path.join(input_dir, f"{base}_imag.npy")

        if not os.path.exists(imag_path):
            print(f"⚠️ {base}의 imag 파일이 존재하지 않아 스킵합니다.")
            continue

        # 1️⃣ 중간 해상도 불러오기
        real_mid = np.load(real_path)
        imag_mid = np.load(imag_path)

        # 2️⃣ 원 해상도 (1024x4096) 복원
        real_back = resize_poly_2d(real_mid, H0, W0)
        imag_back = resize_poly_2d(imag_mid, H0, W0)

        # 3️⃣ 시계열 복원
        x_rec_small = reconstruct_signal_from_coef(real_back, imag_back)

        # 4️⃣ 최종 리샘플링
        x_rec_full = resample_poly(x_rec_small, args.target_length, len(x_rec_small))

        # 5️⃣ 저장 (npy + 그래프)
        out_npy = os.path.join(out_dir, f"{base}_reconstructed.npy")
        out_png = os.path.join(out_dir, f"{base}_reconstructed.png")

        np.save(out_npy, x_rec_full)

        # 그래프 저장
        plt.figure(figsize=(10, 3))
        plt.plot(x_rec_full, color="tab:blue", linewidth=1)
        plt.title(f"Reconstructed Signal: {base}")
        plt.xlabel("Time Index")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    print(f"✅ 모든 시계열 복원 완료 ({len(real_files)}개)")
    print(f"출력 경로: {out_dir}")


# ======================
# CLI Entry
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wavelet Inverse Reconstruction from single folder (real/imag pairs)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="real/imag 쌍이 포함된 입력 폴더 (예: data/wavelet_output)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="복원된 신호 저장 폴더")
    parser.add_argument("--target_length", type=int, default=40960,
                        help="최종 복원 신호 길이 (기본: 40960)")
    args = parser.parse_args()
    main(args)