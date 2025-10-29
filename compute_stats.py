import os, glob, argparse
import numpy as np
import json
from tqdm import tqdm  # 시각적인 진행률 표시를 위해 tqdm 사용 (없다면 pip install tqdm)


def compute_statistics(real_dir, imag_dir, output_json):
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.npy")))
    imag_files = sorted(glob.glob(os.path.join(imag_dir, "*.npy")))

    if not real_files or not imag_files:
        print(f"[ERROR] No .npy files found in {real_dir} or {imag_dir}")
        return

    assert len(real_files) == len(imag_files), "Real/Imag file counts mismatch!"

    mus, stds = [], []

    print(f"[INFO] Processing {len(real_files)} files to compute statistics...")

    # tqdm을 사용하여 진행률 표시
    for rfile, ifile in tqdm(zip(real_files, imag_files), total=len(real_files)):
        try:
            r = np.load(rfile).astype(np.float32)
            i = np.load(ifile).astype(np.float32)

            mus.append([r.mean(), i.mean()])
            stds.append([r.std(), i.std()])
        except Exception as e:
            print(f"\n[WARN] Skipping file due to error: {e} (File: {rfile})")

    # 전체 평균 계산
    mu = np.mean(mus, axis=0).tolist()
    std = np.mean(stds, axis=0).tolist()

    stats = {
        "mu": mu,
        "std": std
    }

    # JSON 파일로 저장
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\n[INFO] Statistics computed and saved to {output_json}")
    print(f"[INFO] norm_mu = {mu}")
    print(f"[INFO] norm_std = {std}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True,
                    help="Directory containing real part .npy files")
    ap.add_argument("--imag_dir", type=str, required=True,
                    help="Directory containing imaginary part .npy files")
    ap.add_argument("--output_json", type=str, default="stats.json",
                    help="Path to save the computed statistics JSON file")
    args = ap.parse_args()

    compute_statistics(args.real_dir, args.imag_dir, args.output_json)