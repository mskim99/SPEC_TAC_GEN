import os
import argparse
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # GUI 없이 이미지 저장 가능하도록 설정
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
    """
    wavelet 계수(real, imag)로부터 시계열 복원 (수치 근사)
    """
    coef_rec = real_data + 1j * imag_data
    n_scales, _ = coef_rec.shape
    scales = np.arange(1, n_scales + 1)
    da = np.gradient(scales)
    rec_raw = np.sum(np.real(coef_rec) * (da[:, None] / (scales[:, None] ** 1.5)), axis=0)
    # 정규화 (optional)
    rec_raw -= np.mean(rec_raw)
    rec_raw /= np.max(np.abs(rec_raw)) + 1e-12
    return rec_raw


# ======================
# Main
# ======================
def main(args):
    real_dir = args.real_dir
    imag_dir = args.imag_dir
    out_dir = args.output_dir
    # out_dir = os.path.join(args.output_dir, "reconstructed")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(real_dir) if f.endswith(".npy")])
    print(f"총 {len(files)}개 파일 복원 중...")
    H0, W0 = 1024, 4096

    for fname in tqdm(files):
        base = fname.replace("_real.npy", "").replace("_real_mid.npy", "").replace(".npy", "")
        real_path = os.path.join(real_dir, fname)

        # 가능한 imag 파일명 탐색
        imag_candidates = [
            f"{base}_imag.npy",
            f"{base}_imag_mid.npy",
            f"{base}.npy"
        ]
        imag_path = None
        for cand in imag_candidates:
            cand_path = os.path.join(imag_dir, cand)
            if os.path.exists(cand_path):
                imag_path = cand_path
                break

        if imag_path is None:
            print(f"⚠️ {base}의 imag 파일을 찾을 수 없어 스킵합니다.")
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

    print(f"✅ 모든 시계열 복원 완료 ({len(files)}개)")
    print(f"출력 경로: {out_dir}")


# ======================
# CLI Entry
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wavelet Inverse Reconstruction with separate real/imag folders (save plots)")
    parser.add_argument("--real_dir", type=str, required=True,
                        help="real 계수 폴더 경로 (예: data/real_mid)")
    parser.add_argument("--imag_dir", type=str, required=True,
                        help="imag 계수 폴더 경로 (예: data/imag_mid)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="복원된 신호 저장 폴더")
    parser.add_argument("--target_length", type=int, default=40960,
                        help="최종 복원 신호 길이 (기본: 40960)")
    args = parser.parse_args()
    main(args)