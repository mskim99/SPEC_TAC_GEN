import os
import numpy as np
import pywt
from scipy.signal import resample, resample_poly
from tqdm import tqdm

# ---------- Settings ----------
input_dir  = "/data/jionkim/LMT_108_surface"            # 입력 시계열 폴더
output_dir = "/data/jionkim/LMT_108_surface_wavelet"    # 결과 저장 폴더
orig_len   = 40960                                      # 원 신호 길이
H0, W0     = 1024, 4096                                 # 원 CWT 해상도 (scale x time)
wavelet    = "cmor1.5-1.0"
targets    = [(1024, 1024)]                             # 생성할 중간 해상도 목록
use_log_scale = True                                    # 로그 스케일 리샘플 적용 여부

# ---------- Utilities ----------
def cwt_1024_4096(x):
    scales = np.arange(1, H0 + 1)
    x_ds = resample(x, W0)  # 시간축 리샘플링 (4096 포인트)
    coef, _ = pywt.cwt(x_ds, scales, wavelet, sampling_period=1.0)
    return coef

def normalize_pm1(a):
    """[-1, 1] 정규화"""
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return 2.0 * a - 1.0

def resize_poly_2d(M, Ht, Wt):
    """2D FIR 기반 리샘플"""
    tmp = resample_poly(M, Ht, M.shape[0], axis=0)
    tmp = resample_poly(tmp, Wt, M.shape[1], axis=1)
    return tmp

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

# ---------- Output folder structure ----------
dirs = {
    "real_mid": os.path.join(output_dir, "real_mid"),
    "imag_mid": os.path.join(output_dir, "imag_mid"),
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# ---------- Process all signals ----------
files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
print(f"총 {len(files)}개의 시계열 처리 중...")

for fname in tqdm(files):
    fpath = os.path.join(input_dir, fname)
    base = os.path.splitext(fname)[0]
    skip_flag = True

    # 변환 완료 여부 체크
    for (Ht, Wt) in targets:
        real_out = os.path.join(dirs["real_mid"], f"{base}_real_{Ht}x{Wt}.npy")
        imag_out = os.path.join(dirs["imag_mid"], f"{base}_imag_{Ht}x{Wt}.npy")
        if not (os.path.exists(real_out) and os.path.exists(imag_out)):
            skip_flag = False
            break

    # 이미 완료된 경우 스킵
    if skip_flag:
        continue

    # 1️⃣ 데이터 로드
    x = np.load(fpath).astype(np.float32)[:orig_len]

    # 2️⃣ CWT 계산
    coef = cwt_1024_4096(x)

    # 3️⃣ Real / Imag 분리 + 정규화
    real = normalize_pm1(np.real(coef))
    imag = normalize_pm1(np.imag(coef))

    # 4️⃣ 중간 해상도 생성 및 저장
    for (Ht, Wt) in targets:
        real_out = os.path.join(dirs["real_mid"], f"{base}_real_{Ht}x{Wt}.npy")
        imag_out = os.path.join(dirs["imag_mid"], f"{base}_imag_{Ht}x{Wt}.npy")

        # 중간 결과 파일이 이미 있으면 skip
        if os.path.exists(real_out) and os.path.exists(imag_out):
            continue

        if use_log_scale:
            real_mid = resize_poly_2d(resize_scale_log(real, Ht), Ht, Wt)
            imag_mid = resize_poly_2d(resize_scale_log(imag, Ht), Ht, Wt)
        else:
            real_mid = resize_poly_2d(real, Ht, Wt)
            imag_mid = resize_poly_2d(imag, Ht, Wt)

        np.save(real_out, real_mid)
        np.save(imag_out, imag_mid)

print("✅ 모든 변환 완료 (이미 변환된 파일은 스킵됨)")
print(f"출력 경로: {output_dir}")