# wavelet_or_stft_data_gen.py (MULTI-CHANNEL PRESERVED)
import os
import json
import glob
import argparse
import random
import numpy as np
import pywt
from scipy.signal import get_window, stft
from tqdm import tqdm

eps = 1e-8


def collect_wave_files(wave_dir: str, recursive: bool) -> list[str]:
    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(glob.glob(os.path.join(wave_dir, pattern), recursive=recursive))


def to_time_channel(x: np.ndarray, channel_axis: str = "auto") -> np.ndarray:
    """
    Returns x_tc: (T, C), float32
    Supports:
      - (T,)
      - (T,C)
      - (C,T)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32)[:, None]

    if x.ndim != 2:
        raise ValueError(f"Input must be 1D or 2D. got shape={x.shape}")

    x = x.astype(np.float32)

    if channel_axis == "last":
        return x
    elif channel_axis == "first":
        return x.T
    elif channel_axis == "auto":
        # 보통 T >> C 이므로 작은 축을 채널로 가정
        if x.shape[0] <= x.shape[1]:
            # (C,T) 가능성
            if x.shape[0] <= 64 and x.shape[1] > x.shape[0]:
                return x.T
        return x
    else:
        raise ValueError(f"Unknown channel_axis={channel_axis}")


# -----------------------------
# CWT helpers (multichannel)
# -----------------------------
def frame_signal_tc(x_tc: np.ndarray, win_len: int, hop_len: int, win_type: str) -> np.ndarray:
    """
    x_tc: (T,C)
    return frames: (N, C, win_len)
    """
    if x_tc.ndim != 2:
        raise ValueError(f"x_tc must be (T,C), got {x_tc.shape}")

    T, C = x_tc.shape
    x_tc = x_tc.astype(np.float32)

    if T < win_len:
        pad = win_len - T
        x_tc = np.pad(x_tc, ((0, pad), (0, 0)), mode="constant")
        T = x_tc.shape[0]

    n_frames = int(np.ceil((T - win_len) / hop_len)) + 1
    target_len = (n_frames - 1) * hop_len + win_len
    pad_right = max(0, target_len - T)
    if pad_right > 0:
        x_tc = np.pad(x_tc, ((0, pad_right), (0, 0)), mode="constant")

    frames = np.stack(
        [x_tc[i * hop_len: i * hop_len + win_len, :] for i in range(n_frames)],
        axis=0
    )  # (N, win_len, C)
    frames = np.transpose(frames, (0, 2, 1))  # (N,C,win_len)

    w = get_window(win_type, win_len, fftbins=False).astype(np.float32)
    frames = frames * w[None, None, :]
    return frames.astype(np.float32)


def make_scales(fs: float, wavelet: str, n_scales: int, f_min: float, f_max: float, spacing: str):
    if spacing == "log":
        freqs = np.geomspace(f_min, f_max, n_scales)
    else:
        freqs = np.linspace(f_min, f_max, n_scales)
    fc = pywt.central_frequency(wavelet)
    scales = (fc * fs) / freqs
    return scales.astype(np.float32), freqs.astype(np.float32)


def cwt_frame_multich(frame_cw: np.ndarray, scales: np.ndarray, wavelet: str) -> np.ndarray:
    """
    frame_cw: (C, W)
    return coef: (C, S, W) complex64
    """
    C, W = frame_cw.shape
    out = []
    for c in range(C):
        coef, _ = pywt.cwt(frame_cw[c], scales, wavelet)  # (S,W)
        out.append(coef.astype(np.complex64))
    return np.stack(out, axis=0).astype(np.complex64)  # (C,S,W)


# -----------------------------
# STFT helpers (multichannel)
# -----------------------------
def stft_full_multich(
    x_tc: np.ndarray,
    fs: float,
    win_len: int,
    hop_len: int,
    win_type: str,
    n_fft: int | None,
    onesided: bool,
    boundary: str | None,
    padded: bool
):
    """
    x_tc: (T,C)
    return:
      f: (F,)
      t: (TT,)
      Z: (C,F,TT) complex
    """
    x_tc = np.asarray(x_tc).astype(np.float32)
    if x_tc.ndim != 2:
        raise ValueError(f"x_tc must be (T,C), got {x_tc.shape}")

    T, C = x_tc.shape
    if n_fft is None:
        n_fft = win_len
    noverlap = max(0, win_len - hop_len)

    Z_list = []
    f_ref, t_ref = None, None
    for c in range(C):
        f, t, Z = stft(
            x_tc[:, c],
            fs=fs,
            window=win_type,
            nperseg=win_len,
            noverlap=noverlap,
            nfft=n_fft,
            detrend=False,
            return_onesided=onesided,
            boundary=boundary,
            padded=padded,
            axis=-1,
        )  # Z: (F,TT)
        if f_ref is None:
            f_ref, t_ref = f.astype(np.float32), t.astype(np.float32)
        Z_list.append(Z.astype(np.complex64))
    Z_cft = np.stack(Z_list, axis=0).astype(np.complex64)  # (C,F,TT)
    return f_ref, t_ref, Z_cft


def chunk_time_axis_multich(Z_cft: np.ndarray, time_bins: int | None, time_hop: int | None) -> np.ndarray:
    """
    Z_cft: (C,F,T)
    return: (N,C,F,time_bins') complex
    """
    C, F, T = Z_cft.shape

    if time_bins is None or time_bins <= 0:
        return Z_cft[None, :, :, :]  # (1,C,F,T)

    if time_hop is None or time_hop <= 0:
        time_hop = time_bins

    if T <= 0:
        return np.zeros((1, C, F, time_bins), dtype=np.complex64)

    if T < time_bins:
        pad = time_bins - T
        Zp = np.pad(Z_cft, ((0, 0), (0, 0), (0, pad)), mode="constant")
        return Zp[None, :, :, :]

    n_chunks = int(np.ceil((T - time_bins) / time_hop)) + 1
    total_len = (n_chunks - 1) * time_hop + time_bins
    pad_right = max(0, total_len - T)
    if pad_right > 0:
        Z_cft = np.pad(Z_cft, ((0, 0), (0, 0), (0, pad_right)), mode="constant")

    chunks = np.stack(
        [Z_cft[:, :, i * time_hop: i * time_hop + time_bins] for i in range(n_chunks)],
        axis=0
    )  # (N,C,F,time_bins)
    return chunks.astype(np.complex64)


def select_and_resample_freq_multich(
    Z_cft: np.ndarray,     # (C,F,T)
    freqs: np.ndarray,     # (F,)
    f_min: float,
    f_max: float,
    out_bins: int | None,
    spacing: str
):
    freqs = freqs.astype(np.float32)
    if f_min is None:
        f_min = float(freqs.min())
    if f_max is None:
        f_max = float(freqs.max())

    sel = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(sel):
        sel = np.ones_like(freqs, dtype=bool)

    f_sel = freqs[sel]
    Z_sel = Z_cft[:, sel, :]  # (C,F2,T)

    if out_bins is None or out_bins <= 0 or f_sel.size < 2:
        return Z_sel.astype(np.complex64), f_sel.astype(np.float32)

    if spacing == "log":
        f_pos = f_sel[f_sel > 0]
        f1 = max(float(f_min), float(np.min(f_pos)) if f_pos.size > 0 else 1e-3)
        f2 = float(f_max)
        f_tgt = np.geomspace(f1, f2, out_bins).astype(np.float32)
    else:
        f_tgt = np.linspace(float(f_min), float(f_max), out_bins).astype(np.float32)

    C, _, T = Z_sel.shape
    re = np.empty((C, out_bins, T), dtype=np.float32)
    im = np.empty((C, out_bins, T), dtype=np.float32)

    x = f_sel.astype(np.float32)
    for c in range(C):
        for ti in range(T):
            re[c, :, ti] = np.interp(f_tgt, x, Z_sel[c].real[:, ti].astype(np.float32))
            im[c, :, ti] = np.interp(f_tgt, x, Z_sel[c].imag[:, ti].astype(np.float32))

    Z_out = (re + 1j * im).astype(np.complex64)  # (C,out_bins,T)
    return Z_out, f_tgt


# -----------------------------
# common
# -----------------------------
def asinh_compand(x: np.ndarray, scale: float) -> np.ndarray:
    return np.arcsinh(x / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--recursive", action="store_true")

    # 입력 waveform shape 처리
    ap.add_argument("--channel_axis", choices=["auto", "last", "first"], default="auto",
                    help="입력 .npy가 2D일 때 채널축 위치")

    # transform
    ap.add_argument("--transform", choices=["cwt", "stft"], default="cwt")

    # sampling rate
    ap.add_argument("--fs", type=float, default=2800.0)

    # common framing params
    ap.add_argument("--win_len", type=int, default=512)
    ap.add_argument("--hop_len", type=int, default=512)
    ap.add_argument("--win_type", type=str, default="hann")

    # CWT params
    ap.add_argument("--wavelet", type=str, default="cmor1.5-1.0")
    ap.add_argument("--n_scales", type=int, default=64)
    ap.add_argument("--f_min", type=float, default=10.0)
    ap.add_argument("--f_max", type=float, default=700.0)
    ap.add_argument("--freq_spacing", choices=["log", "linear"], default="log")

    # STFT params
    ap.add_argument("--n_fft", type=int, default=0)
    ap.add_argument("--stft_onesided", action="store_true")
    ap.add_argument("--stft_twosided", action="store_true")
    ap.add_argument("--stft_boundary", choices=["none", "zeros"], default="none")
    ap.add_argument("--stft_padded", action="store_true")
    ap.add_argument("--spec_time_bins", type=int, default=0)
    ap.add_argument("--spec_time_hop", type=int, default=0)
    ap.add_argument("--match_bins", action="store_true")

    # global_mag sampling
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mag_sample_files", type=int, default=1500)
    ap.add_argument("--mag_sample_frames", type=int, default=1000)

    ap.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    MAG_DIR, RE_DIR, IM_DIR = "mag", "re", "im"
    os.makedirs(args.output_dir, exist_ok=True)
    for d in [MAG_DIR, RE_DIR, IM_DIR]:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    wave_files = collect_wave_files(args.wave_dir, args.recursive)
    if not wave_files:
        raise RuntimeError(f"No .npy files found in {args.wave_dir}")

    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32

    if args.transform == "cwt":
        scales, freqs = make_scales(
            fs=args.fs, wavelet=args.wavelet, n_scales=args.n_scales,
            f_min=args.f_min, f_max=args.f_max, spacing=args.freq_spacing
        )
        cwt_time_bins = args.win_len
    else:
        n_fft = None if args.n_fft == 0 else args.n_fft
        onesided = True
        if args.stft_twosided:
            onesided = False
        if args.stft_onesided:
            onesided = True
        boundary = None if args.stft_boundary == "none" else "zeros"
        padded = bool(args.stft_padded)
        stft_time_bins = None if args.spec_time_bins == 0 else int(args.spec_time_bins)
        stft_time_hop = None if args.spec_time_hop == 0 else int(args.spec_time_hop)

    # -----------------------------
    # Pass 1: global_mag estimate
    # -----------------------------
    print(f"🔍 Pass 1: Estimating global_mag ({args.transform.upper()})")
    rng = random.Random(args.seed)
    sample_files = wave_files[:]
    rng.shuffle(sample_files)
    sample_files = sample_files[: min(args.mag_sample_files, len(sample_files))]

    mag_medians = []
    num_channels_ref = None

    for wpath in tqdm(sample_files, desc="Sampling files"):
        x_raw = np.load(wpath, allow_pickle=False)
        x_tc = to_time_channel(x_raw, channel_axis=args.channel_axis)  # (T,C)
        T, C = x_tc.shape
        if num_channels_ref is None:
            num_channels_ref = C
        elif C != num_channels_ref:
            raise ValueError(f"Inconsistent channel count. got {C}, expected {num_channels_ref} ({wpath})")

        if args.transform == "cwt":
            frames = frame_signal_tc(x_tc, args.win_len, args.hop_len, args.win_type)  # (N,C,W)
            n_frames = frames.shape[0]
            k = min(args.mag_sample_frames, n_frames)
            idxs = np.linspace(0, n_frames - 1, num=k, dtype=int)

            for i in idxs:
                coef = cwt_frame_multich(frames[i], scales, args.wavelet)  # (C,S,W)
                mag = np.abs(coef).astype(np.float32)
                # 채널별 median 다 반영
                for c in range(C):
                    mag_medians.append(float(np.median(mag[c])))

        else:
            f, t, Z_cft = stft_full_multich(
                x_tc, fs=args.fs,
                win_len=args.win_len, hop_len=args.hop_len, win_type=args.win_type,
                n_fft=n_fft, onesided=onesided, boundary=boundary, padded=padded
            )  # (C,F,T)

            out_bins = args.n_scales if args.match_bins else None
            Z2, _ = select_and_resample_freq_multich(Z_cft, f, args.f_min, args.f_max, out_bins, args.freq_spacing)
            mag = np.abs(Z2).astype(np.float32)  # (C,F2,T)
            TT = mag.shape[2]
            if TT <= 0:
                continue
            k = min(args.mag_sample_frames, TT)
            idxs = np.linspace(0, TT - 1, num=k, dtype=int)
            for ti in idxs:
                for c in range(C):
                    mag_medians.append(float(np.median(mag[c, :, ti])))

    global_mag = float(np.median(np.array(mag_medians, dtype=np.float32))) if mag_medians else 1.0
    global_mag = max(global_mag, eps)
    print(f"✔ global_mag = {global_mag:.6f}")

    # -----------------------------
    # Pass 2: compute & save
    # -----------------------------
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    total_done, total_skip = 0, 0
    freqs_out_last = None
    saved_num_channels = None

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for wpath in tqdm(wave_files, desc="Processing"):
            rel = os.path.relpath(wpath, args.wave_dir)
            rel_dir = os.path.dirname(rel)
            base = os.path.splitext(os.path.basename(wpath))[0]

            mag_out_dir = os.path.join(args.output_dir, MAG_DIR, rel_dir)
            re_out_dir  = os.path.join(args.output_dir, RE_DIR, rel_dir)
            im_out_dir  = os.path.join(args.output_dir, IM_DIR, rel_dir)
            os.makedirs(mag_out_dir, exist_ok=True)
            os.makedirs(re_out_dir, exist_ok=True)
            os.makedirs(im_out_dir, exist_ok=True)

            mag_out = os.path.join(mag_out_dir, f"{base}_mag.npy")
            re_out  = os.path.join(re_out_dir, f"{base}_re.npy")
            im_out  = os.path.join(im_out_dir, f"{base}_im.npy")

            if os.path.exists(mag_out) and os.path.exists(re_out) and os.path.exists(im_out):
                total_skip += 1
                continue

            x_raw = np.load(wpath, allow_pickle=False)
            x_tc = to_time_channel(x_raw, channel_axis=args.channel_axis)  # (T,C)
            T, C = x_tc.shape
            if saved_num_channels is None:
                saved_num_channels = C
            elif C != saved_num_channels:
                raise ValueError(f"Inconsistent channel count in dataset. {wpath}: {C} vs {saved_num_channels}")

            if args.transform == "cwt":
                frames = frame_signal_tc(x_tc, args.win_len, args.hop_len, args.win_type)  # (N,C,W)
                n_frames = frames.shape[0]

                # 저장 shape: (N,C,S,W)
                mag_c_all = np.empty((n_frames, C, args.n_scales, cwt_time_bins), dtype=save_dtype)
                re_all    = np.empty((n_frames, C, args.n_scales, cwt_time_bins), dtype=save_dtype)
                im_all    = np.empty((n_frames, C, args.n_scales, cwt_time_bins), dtype=save_dtype)

                for i in range(n_frames):
                    coef = cwt_frame_multich(frames[i], scales, args.wavelet)  # (C,S,W)
                    mag = np.abs(coef).astype(np.float32)

                    mag_c_all[i] = asinh_compand(mag, global_mag).astype(save_dtype)
                    re_all[i]    = np.real(coef).astype(save_dtype)
                    im_all[i]    = np.imag(coef).astype(save_dtype)

                freqs_out = freqs.astype(np.float32)

            else:
                f, t, Z_cft = stft_full_multich(
                    x_tc, fs=args.fs,
                    win_len=args.win_len, hop_len=args.hop_len, win_type=args.win_type,
                    n_fft=n_fft, onesided=onesided, boundary=boundary, padded=padded
                )  # (C,F,T)

                out_bins = args.n_scales if args.match_bins else None
                Z2, freqs_out = select_and_resample_freq_multich(Z_cft, f, args.f_min, args.f_max, out_bins, args.freq_spacing)

                chunks = chunk_time_axis_multich(Z2, time_bins=stft_time_bins, time_hop=stft_time_hop)  # (N,C,F,Tb)
                n_frames = int(chunks.shape[0])
                tb = int(chunks.shape[3])

                mag = np.abs(chunks).astype(np.float32)
                mag_c_all = asinh_compand(mag, global_mag).astype(save_dtype)  # (N,C,F,Tb)
                re_all    = np.real(chunks).astype(save_dtype)
                im_all    = np.imag(chunks).astype(save_dtype)

            np.save(mag_out, mag_c_all)
            np.save(re_out, re_all)
            np.save(im_out, im_all)

            freqs_out_last = freqs_out

            rec = {
                "wave": rel.replace("\\", "/"),
                "mag":  os.path.relpath(mag_out, args.output_dir).replace("\\", "/"),
                "re":   os.path.relpath(re_out,  args.output_dir).replace("\\", "/"),
                "im":   os.path.relpath(im_out,  args.output_dir).replace("\\", "/"),
                "transform": args.transform,
                "n_frames": int(n_frames),
                "num_channels": int(C),
                "shape": [int(s) for s in mag_c_all.shape],  # (N,C,F,W)
            }
            mf.write(json.dumps(rec) + "\n")
            total_done += 1

    w = get_window(args.win_type, args.win_len, fftbins=False).astype(np.float32)
    window_rms = float(np.sqrt(np.mean(w ** 2)))

    meta = {
        "transform": args.transform,
        "fs": args.fs,
        "win_len": args.win_len,
        "hop_len": args.hop_len,
        "win_type": args.win_type,
        "window_rms": window_rms,

        "f_min": args.f_min,
        "f_max": args.f_max,
        "freq_spacing": args.freq_spacing,
        "freqs_hz": freqs_out_last.tolist() if freqs_out_last is not None else None,

        "global_mag": float(global_mag),
        "num_channels": int(saved_num_channels if saved_num_channels is not None else 1),
        "feature_shape": "(frames, channels, freq_bins, time_bins)",

        "features": ["mag_c", "real", "imag"],
        "companding": {
            "type": "asinh",
            "applied_to": "magnitude_only",
            "scale": "global_mag",
            "forward": "mag_c = asinh(mag / global_mag)",
            "inverse": "mag = sinh(mag_c) * global_mag",
        },
        "complex_storage": {
            "note": "re/im are RAW complex coefficients.",
            "coef": "coef = re + 1j*im",
        },
        "input_wave_shape": "saved .npy may be (T,), (T,C), or (C,T); converted to (T,C) internally",
        "channel_axis_mode": args.channel_axis,
    }

    if args.transform == "cwt":
        meta["cwt"] = {
            "wavelet": args.wavelet,
            "n_scales": args.n_scales,
            "time_bins": int(args.win_len),
            "note": "Per-frame CWT, saved as (n_frames, C, n_scales, win_len).",
        }
    else:
        meta["stft"] = {
            "n_fft": (args.win_len if args.n_fft == 0 else args.n_fft),
            "onesided": (not args.stft_twosided),
            "boundary": (None if args.stft_boundary == "none" else "zeros"),
            "padded": bool(args.stft_padded),
            "spec_time_bins": (None if args.spec_time_bins == 0 else int(args.spec_time_bins)),
            "spec_time_hop": (None if args.spec_time_hop == 0 else int(args.spec_time_hop)),
            "match_bins_to_n_scales": bool(args.match_bins),
            "note": "Full-signal STFT -> time-axis chunking, saved as (n_chunks, C, F, time_bins).",
        }

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print("🎉 Done")
    print(f"Processed newly: {total_done}, skipped(existing): {total_skip}")
    print(f"📁 Output dir: {os.path.abspath(args.output_dir)}")
    print(f"🧾 Manifest: {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()