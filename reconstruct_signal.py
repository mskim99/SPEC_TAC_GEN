import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ✅ GUI 없는 환경에서도 안전하게 작동
import matplotlib.pyplot as plt
from scipy.signal import istft
from tqdm import tqdm


def load_spectrograms(mag_dir, phase_dir, sort=True):
    """각 폴더의 magnitude / phase 파일을 로드하여 쌍으로 반환"""
    mag_files = sorted(os.listdir(mag_dir)) if sort else os.listdir(mag_dir)
    phase_files = sorted(os.listdir(phase_dir)) if sort else os.listdir(phase_dir)

    pairs = []
    for mfile, pfile in zip(mag_files, phase_files):
        if not (mfile.endswith(".npy") and pfile.endswith(".npy")):
            continue
        mag = np.load(os.path.join(mag_dir, mfile))
        phase = np.load(os.path.join(phase_dir, pfile))
        pairs.append((mfile, mag, phase))
    return pairs


def reconstruct_signal_from_mag_phase(mag, phase, n_fft=256, hop_length=128):
    """
    magnitude와 phase를 이용하여 시간 신호를 복원.
    mag: (freq, time)
    phase: (freq, time)
    """
    # 복소 스펙트럼 복원
    spec_complex = mag * np.exp(1j * phase)
    _, signal = istft(spec_complex, nperseg=n_fft, noverlap=n_fft - hop_length)
    return signal


def save_spectrogram_image(data, path, title=None, cmap="jet", vmin=None, vmax=None):
    """스펙트럼 데이터를 이미지로 저장"""
    plt.figure(figsize=(6, 4))
    plt.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.colorbar(format="%.2f")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def combine_signals(mag_dir, phase_dir, out_dir,
                    n_fft=256, hop_length=128, plot=True):
    """두 디렉토리의 magnitude/phase 스펙트럼을 합쳐 신호 복원 및 저장"""
    os.makedirs(out_dir, exist_ok=True)
    pairs = load_spectrograms(mag_dir, phase_dir)

    all_signals = []
    for idx, (fname, mag, phase) in enumerate(tqdm(pairs, desc="Reconstructing signals")):
        # 신호 복원
        sig = reconstruct_signal_from_mag_phase(mag, phase, n_fft, hop_length)
        all_signals.append(sig)

        # ===============================
        # 🔹 Magnitude & Phase 저장
        # ===============================
        np.save(os.path.join(out_dir, f"mag_{idx}.npy"), mag)
        np.save(os.path.join(out_dir, f"phase_{idx}.npy"), phase)

        # PNG 저장
        save_spectrogram_image(mag, os.path.join(out_dir, f"mag_{idx}.png"),
                               title=f"Magnitude {idx}", cmap="jet")
        save_spectrogram_image(phase, os.path.join(out_dir, f"phase_{idx}.png"),
                               title=f"Phase {idx}", cmap="twilight_shifted")

        # ===============================
        # 🔹 신호 저장
        # ===============================
        np.save(os.path.join(out_dir, f"signal_{idx}.npy"), sig)

        if plot:
            plt.figure(figsize=(10, 3))
            plt.plot(sig)
            plt.title(f"Reconstructed Signal {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"signal_{idx}.png"))
            plt.close()

    # 전체 신호 이어붙이기
    combined = np.concatenate(all_signals)
    np.save(os.path.join(out_dir, "combined_signal.npy"), combined)
    print(f"✅ Saved {len(all_signals)} signals + mag/phase PNGs to {out_dir}")

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine magnitude & phase spectrograms into waveform")
    parser.add_argument("--mag_dir", type=str, required=True, help="Directory containing magnitude .npy files")
    parser.add_argument("--phase_dir", type=str, required=True, help="Directory containing phase .npy files")
    parser.add_argument("--out_dir", type=str, default="reconstructed", help="Output directory")
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--plot", action="store_true", help="Save waveform plots")
    args = parser.parse_args()

    combine_signals(args.mag_dir, args.phase_dir, args.out_dir,
                    n_fft=args.n_fft, hop_length=args.hop_length, plot=args.plot)