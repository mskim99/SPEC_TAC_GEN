import numpy as np
import pywt
from scipy.signal import resample, resample_poly
import matplotlib.pyplot as plt

# ---------- Settings ----------
input_path = "J:/test/G1EpoxyRasterPlate_Movement_X_train1.npy"
orig_len = 40960  # 원 신호 길이
H0, W0 = 64, 4096  # 원 CWT 해상도 (scale x time)
wavelet = "cmor1.5-1.0"
# <<< 수정된 부분: (1024, 1024) -> (256, 256)
targets = [(64, 4096)]  # 중간 해상도 후보
use_log_scale = True  # 로그 스케일 사용
svd_ranks = [16]  # None이면 SVD 미사용
fold_target = False  # 로그 스케일을 사용하기 위해 False로 유지

# ---------- Utilities ----------
def cwt_1024_4096(x):
    scales = np.arange(1, H0 + 1)
    x_ds = resample(x, W0)  # 시간축 4096으로
    coef, freqs = pywt.cwt(x_ds, scales, wavelet, sampling_period=1.0)
    return coef, scales, x_ds

def normalize_pm1(a):
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return 2.0 * a - 1.0


def normalize_01(a):
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return a


def resize_poly_2d(M, Ht, Wt):
    tmp = resample_poly(M, Ht, M.shape[0], axis=0)
    tmp = resample_poly(tmp, Wt, M.shape[1], axis=1)
    return tmp


def reconstruct_signal_from_coef(realM, imagM, x_ref, scales):
    # (사용자 원본 코드와 동일)
    coef_rec = realM + 1j * imagM
    sc = np.asarray(scales, dtype=float)
    da = np.gradient(sc)
    rec_raw = np.sum(np.real(coef_rec) * (da[:, None] / (sc[:, None] ** 1.5)), axis=0)

    r_mean, r_var = rec_raw.mean(), rec_raw.var() + 1e-12
    x_mean = x_ref.mean()
    cov_rx = np.mean((rec_raw - r_mean) * (x_ref - x_mean))
    a = cov_rx / r_var
    b = x_mean - a * r_mean
    return a * rec_raw + b


def apply_svd_compress(M, r=None):
    # (사용자 원본 코드와 동일)
    if r is None:
        return M, None
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, len(s))
    Mr = (U[:, :r] * s[:r]) @ Vt[:r, :]
    meta = (U[:, :r], s[:r], Vt[:r, :])
    return Mr, meta


def restore_svd(meta):
    # (사용자 원본 코드와 동일)
    U, s, Vt = meta
    return (U * s) @ Vt


def log_select_indices(Hsrc, Ht):
    """로그 스케일 샘플링 인덱스"""
    return np.unique(np.clip(np.round(
        np.exp(np.linspace(np.log(1), np.log(Hsrc), Ht)) - 1
    ).astype(int), 0, Hsrc - 1))

def resize_scale_log(M, Ht):
    """스케일축 로그 리샘플"""
    if Ht >= M.shape[0]:
        return resample_poly(M, Ht, M.shape[0], axis=0)
    idx = log_select_indices(M.shape[0], Ht)
    sel = M[idx, :]
    return resample_poly(sel, Ht, sel.shape[0], axis=0)

# -----------------------------------------------------------------
# <<< 추가: 무손실 Reshape (Folding/Unfolding) 함수
# -----------------------------------------------------------------
def fold_cwt_to_square(M, H_src=16, W_src=4096, H_tgt=256, W_tgt=256):
    """ (16, 4096) -> (256, 256) 무손실 변환 """
    if M.shape != (H_src, W_src):
        raise ValueError(f"Input shape must be ({H_src}, {W_src})")

    patches = M.reshape(H_src, H_tgt // H_src, W_tgt)
    return patches.reshape(H_tgt, W_tgt)


def unfold_cwt_from_square(M, H_tgt=16, W_tgt=4096, H_src=256, W_src=256):
    """ (256, 256) -> (16, 4096) 무손실 역변환 """
    if M.shape != (H_src, W_src):
        raise ValueError(f"Input shape must be ({H_src}, {W_src})")

    patches = M.reshape(H_tgt, H_src // H_tgt, W_src)
    return patches.reshape(H_tgt, W_tgt)


# -----------------------------------------------------------------

# ---------- Load & CWT ----------
x = np.load(input_path)[:orig_len]
x = (x - x.min()) / (x.max() - x.min())
x = np.clip(x, 0., 1.)
coef0, scales0, x_ds = cwt_1024_4096(x)

real0 = normalize_pm1(np.real(coef0))
imag0 = normalize_pm1(np.imag(coef0))

print("Original CWT shape:", real0.shape)  # (1024, 4096)

results = []
target_cache = {}

for (Ht, Wt) in targets:

    # 1) 중간 해상도 생성 (Reshape)
    # fold_target = False 이므로 항상 else 블록 실행
    if Ht * Wt == H0 * W0 and fold_target:
        print(f"Using Lossless Reshape (Fold) for {(Ht, Wt)}")
        real_mid = fold_cwt_to_square(real0, H0, W0, Ht, Wt)
        imag_mid = fold_cwt_to_square(imag0, H0, W0, Ht, Wt)
    else:
        print(f"Using Lossy Resampling for {(Ht, Wt)}")
        # use_log_scale = True 이므로 이 블록 실행
        if use_log_scale:
            print(f"Applying Log-Scale Resampling (H: {H0}->{Ht})")

            # 1. 스케일 축(H)을 로그 스케일로 리샘플 (1024 -> 256)
            real_h_scaled = resize_scale_log(real0, Ht)
            imag_h_scaled = resize_scale_log(imag0, Ht)
            # 결과 shape: (256, 4096)

            # 2. 시간 축(W)을 FFT 기반(resample)으로 리샘플 (4096 -> 256)
            # (scipy.signal.resample은 axis 인자를 받습니다)
            real_mid = resample(real_h_scaled, Wt, axis=1)
            imag_mid = resample(imag_h_scaled, Wt, axis=1)
        else:
            print("Applying Linear Resampling")
            real_mid = resize_poly_2d(real0, Ht, Wt)
            imag_mid = resize_poly_2d(imag0, Ht, Wt)

    for r in svd_ranks:
        # 2) SVD 압축
        real_c, meta_r = apply_svd_compress(real_mid, r)
        imag_c, meta_i = apply_svd_compress(imag_mid, r)

        # 3) 모델이 학습할 "Ground Truth Target"
        real_c_target = normalize_pm1(real_c)
        imag_c_target = normalize_pm1(imag_c)

        target_cache[(Ht, Wt, r)] = (real_c_target, imag_c_target)

        # --------------------------------------------------------
        # <<< 요청 사항 1: 중간 이미지 show >>>
        # --------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(real_c_target, cmap='gray', aspect='auto', vmin=-1, vmax=1)
        plt.title(f"Target: Real (Ht={Ht}, Wt={Wt}, r={r})")
        plt.colorbar(label="Value (Normalized)")

        plt.subplot(1, 2, 2)
        plt.imshow(imag_c_target, cmap='gray', aspect='auto', vmin=-1, vmax=1)
        plt.title(f"Target: Imag (Ht={Ht}, Wt={Wt}, r={r})")
        plt.colorbar(label="Value (Normalized)")

        plt.suptitle(f"Intermediate Target Images for {(Ht, Wt, r)}")
        plt.tight_layout()
        plt.show()  # <<< 이미지 표시
        # --------------------------------------------------------

        # 4) 원 해상도로 복원 (Unfold)
        # fold_target = False 이므로 항상 else 블록 실행
        if Ht * Wt == H0 * W0 and fold_target:
            real_back = unfold_cwt_from_square(real_c_target, H0, W0, Ht, Wt)
            imag_back = unfold_cwt_from_square(imag_c_target, H0, W0, Ht, Wt)
        else:
            # 타겟 이미지를 다시 원본 CWT 해상도로 (선형) 복원
            real_back = resize_poly_2d(real_c_target, H0, W0)
            imag_back = resize_poly_2d(imag_c_target, H0, W0)

        # 5) iCWT 근사 (정규화된 값 그대로 사용)
        x_rec_small = reconstruct_signal_from_coef(real_back, imag_back, x_ds, scales0)
        x_rec_full = resample_poly(x_rec_small, len(x), len(x_ds))

        mse = np.mean((x - x_rec_full) ** 2)
        results.append(((Ht, Wt, r), mse))

        print(f"Mid {(Ht, Wt)}, SVD r={r}: MSE={mse:.6e} (Using Log-Scale, No Denorm)")

# ======================
# 각 case별 복원 결과 시각화
# ======================
plt.figure(figsize=(12, 4))
plt.plot(x, label="Original", alpha=0.7, color='black')

# colors = plt.cm.viridis(np.linspace(0, 1, len(results))) # <<< 불필요

for i, ((Ht, Wt, r), mse) in enumerate(results):

    real_c_target, imag_c_target = target_cache[(Ht, Wt, r)]

    # --- 동일한 복원 파이프라인 적용 ---
    if Ht * Wt == H0 * W0 and fold_target:
        real_back = unfold_cwt_from_square(real_c_target, H0, W0, Ht, Wt)
        imag_back = unfold_cwt_from_square(imag_c_target, H0, W0, Ht, Wt)
    else:
        # 타겟 이미지를 다시 원본 CWT 해상도로 (선형) 복원
        real_back = resize_poly_2d(real_c_target, H0, W0)
        imag_back = resize_poly_2d(imag_c_target, H0, W0)
    # ----------------------------------------

    x_rec_small = reconstruct_signal_from_coef(real_back, imag_back, x_ds, scales0)
    x_rec_full = resample_poly(x_rec_small, len(x), len(x_ds))

    # Plot
    # --------------------------------------------------------
    # <<< 요청 사항 2: 생성 결과(복원 신호) 빨간색으로 표기 >>>
    # --------------------------------------------------------
    plt.plot(x_rec_full, color='r', alpha=0.6,
             label=f"({Ht}×{Wt}), r={r}, MSE={mse:.2e}")

plt.title("Reconstruction Comparison (Using Log-Scale Resampling)")
plt.xlabel("Time Index")
plt.ylabel("Value")
plt.legend(fontsize=8, loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()  # <<< 최종 시계열 그래프 표시

# ---------- Best ----------
best = min(results, key=lambda z: z[1])
print("\nBEST config (Ht, Wt, svd_rank), MSE =", best)