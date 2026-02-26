# csv_to_npy_windows.py
import os
import argparse
import numpy as np
import pandas as pd

def load_csv_timeseries(path: str, is_ett: bool) -> np.ndarray:
    """
    Returns x of shape (T, C) float32.
    - is_ett=True: drop first column (usually 'date') unconditionally
    - is_ett=False: keep numeric columns only (drops any non-numeric cols safely)
    """
    df = pd.read_csv(path)

    if is_ett:
        if df.shape[1] < 2:
            raise ValueError(f"[ETT] CSV must have at least 2 columns (date + values). Got {df.shape}")
        # drop first column regardless of dtype (ETT usually has 'date')
        x = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        return x

    # non-ETT: keep numeric columns only
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] == 0:
        raise ValueError(f"[Non-ETT] No numeric columns found in {path}. Columns={list(df.columns)}")

    dropped = [c for c in df.columns if c not in df_num.columns]
    if dropped:
        print(f"[WARN] Dropped non-numeric columns: {dropped}")

    x = df_num.to_numpy(dtype=np.float32)
    return x

def save_windows(
    x_tc: np.ndarray,
    out_dir: str,
    seq_len: int,
    stride: int,
    max_samples: int = 0,
    one_channel: int | None = None,
    save_channel_last: bool = True,
):
    """
    x_tc: (T,C)
    Saves each window as .npy:
      - if C==1 -> saves (L,) by default
      - if C>1  -> saves (L,C) (channel_last) unless save_channel_last=False then (C,L)
    """
    os.makedirs(out_dir, exist_ok=True)

    if x_tc.ndim != 2:
        raise ValueError(f"x_tc must be 2D (T,C). Got {x_tc.shape}")

    T, C = x_tc.shape

    if one_channel is not None:
        if not (0 <= one_channel < C):
            raise ValueError(f"one_channel out of range: {one_channel} for C={C}")
        x_tc = x_tc[:, one_channel:one_channel+1]
        T, C = x_tc.shape

    n = 0
    for s in range(0, T - seq_len + 1, stride):
        w = x_tc[s:s+seq_len]  # (L,C)

        if C == 1:
            w_to_save = w.squeeze(-1)  # (L,)
        else:
            w_to_save = w if save_channel_last else w.T  # (L,C) or (C,L)

        np.save(os.path.join(out_dir, f"{n:06d}.npy"), w_to_save)
        n += 1
        if max_samples > 0 and n >= max_samples:
            break

    print(f"✅ saved {n} windows -> {out_dir}")
    print(f"   seq_len={seq_len}, stride={stride}, channels={C}, saved_shape_example={w_to_save.shape if n>0 else None}")

def main():
    ap = argparse.ArgumentParser("CSV -> windowed .npy dataset maker (ETT-aware)")
    ap.add_argument("--csv_path", type=str, required=True, help="path to CSV (e.g., ./data/waveform/ETTh1.csv)")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory for .npy windows")

    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--stride", type=int, default=None, help="default: seq_len (no overlap)")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all possible windows")

    # 핵심 플래그: ETTh냐 아니냐
    ap.add_argument("--is_ett", action="store_true",
                    help="If set, drops the first column (date) like ETTh/ETTm datasets")

    # 옵션
    ap.add_argument("--one_channel", type=int, default=None,
                    help="If set, keep only this channel index (0-based)")
    ap.add_argument("--channel_first", action="store_true",
                    help="If set, save each window as (C,L) instead of (L,C) for multi-channel")

    args = ap.parse_args()

    stride = args.stride if args.stride is not None else args.seq_len

    x = load_csv_timeseries(args.csv_path, is_ett=args.is_ett)
    save_windows(
        x_tc=x,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        stride=stride,
        max_samples=args.max_samples,
        one_channel=args.one_channel,
        save_channel_last=(not args.channel_first),
    )

if __name__ == "__main__":
    main()
