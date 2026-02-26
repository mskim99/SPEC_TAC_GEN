# qualitative_eval.py
# =========================================================
# PCA / t-SNE / PDF qualitative evaluation for .npy signals
#
# Supports:
# - 1D signal: (L,)
# - Multi-channel signal: (L,C) or (C,L)
#
# Outputs:
# - embeddings_2d.csv
# - pca_scatter.png, tsne_scatter.png
# - per-system overlay plots (optional)
# - channel-wise PDF plots + metrics CSV
#
# NEW:
# - Pairwise independent eval (GT vs each system):
#   --pairwise_eval / --pairwise_only
# - Pairwise fixed colors:
#   GT=red, SYS=blue (scatter + PDF)
# - Optional distribution matching for embeddings:
#   --dist_match {none,affine,quantile}
# - Partial blending between original and matched:
#   --dist_match_alpha (0~1)
# =========================================================

import os
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional scipy for exact Wasserstein + KDE plotting
try:
    from scipy.stats import wasserstein_distance as scipy_wasserstein_distance, gaussian_kde
    _HAS_SCIPY_WD = True
    _HAS_SCIPY_KDE = True
except Exception:
    _HAS_SCIPY_WD = False
    _HAS_SCIPY_KDE = False

# Pairwise fixed colors (GT vs one system)
PAIRWISE_GT_COLOR = "#FF0000"   # red
PAIRWISE_SYS_COLOR = "#003CFF"  # blue


# ==============================
# File selection helpers
# ==============================
def list_npy_files(root: str, recursive: bool = True):
    files = []
    if recursive:
        for dp, _, fnames in os.walk(root):
            for f in fnames:
                if f.endswith(".npy"):
                    files.append(os.path.join(dp, f))
    else:
        files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npy")]
    return sorted(files)


def pick_files(files, n: int, mode: str, seed: int):
    """
    mode:
      - sorted: take first n from sorted list
      - random: random choice without replacement
    """
    if n <= 0 or n >= len(files):
        return files
    if mode == "sorted":
        return files[:n]
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(files), size=n, replace=False)
        return [files[i] for i in sorted(idx)]
    raise ValueError(f"Unknown pick mode: {mode}")


# ==============================
# Signal loading (1D + multi-channel)
# ==============================
def load_signal_npy(path: str, seq_len_hint: int = 0) -> np.ndarray:
    """
    Return signal in shape (L, C).

    Supports:
      - (L,)     -> (L,1)
      - (L,C)    -> (L,C)
      - (C,L)    -> transpose to (L,C)

    seq_len_hint:
      If > 0, helps infer (L,C) vs (C,L) for 2D arrays.
    """
    x = np.load(path, allow_pickle=True)
    x = np.asarray(x).squeeze()

    if x.ndim == 1:
        return x.astype(np.float32)[:, None]  # (L,1)

    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D signal after squeeze. Got {x.shape} at {path}")

    # Use hint first if provided
    if seq_len_hint and seq_len_hint > 0:
        if x.shape[0] == seq_len_hint and x.shape[1] != seq_len_hint:
            return x.astype(np.float32)       # (L,C)
        if x.shape[1] == seq_len_hint and x.shape[0] != seq_len_hint:
            return x.T.astype(np.float32)     # (C,L) -> (L,C)

    # Heuristic: time axis usually longer than channels
    if x.shape[0] >= x.shape[1]:
        return x.astype(np.float32)           # (L,C)
    else:
        return x.T.astype(np.float32)         # (C,L) -> (L,C)


# ==============================
# Global normalization (REAL stats, channel-wise)
# ==============================
def compute_real_stats(real_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    real_list: list of arrays, each (L,C)
    Returns channel-wise stats (shape: (C,))
    """
    if len(real_list) == 0:
        raise ValueError("real_list is empty")

    C0 = real_list[0].shape[1]
    for i, x in enumerate(real_list):
        if x.ndim != 2:
            raise ValueError(f"Expected (L,C), got {x.shape} at idx={i}")
        if x.shape[1] != C0:
            raise ValueError(f"GT channel mismatch: first C={C0}, got {x.shape[1]} at idx={i}")

    cat = np.concatenate(real_list, axis=0).astype(np.float64)  # (sumL, C)
    mu = np.mean(cat, axis=0)
    sd = np.std(cat, axis=0) + 1e-8
    mn = np.min(cat, axis=0)
    mx = np.max(cat, axis=0)
    return {"mean": mu, "std": sd, "min": mn, "max": mx}


def apply_norm_global(x: np.ndarray, stats: dict, mode: str) -> np.ndarray:
    """
    x: (L,C)
    stats: channel-wise stats from GT
    """
    x = x.astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected (L,C), got {x.shape}")

    c_stats = stats["mean"].shape[0]
    if x.shape[1] != c_stats:
        raise ValueError(f"Channel mismatch during normalization: x has C={x.shape[1]}, stats have C={c_stats}")

    if mode == "none":
        return x

    if mode == "zscore":
        return ((x - stats["mean"]) / stats["std"]).astype(np.float32)

    if mode == "minmax":
        mn, mx = stats["min"], stats["max"]
        denom = (mx - mn + 1e-12)
        denom = np.where(np.isclose(mx, mn), 1.0, denom)
        y = (x - mn) / denom
        return (y * 2.0 - 1.0).astype(np.float32)

    raise ValueError(f"Unknown norm mode: {mode}")


# ==============================
# Resample helper (time-axis)
# ==============================
def resample_signal(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    x: (L,C) -> (target_len,C) by linear interpolation on time axis.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (L,C), got {x.shape}")

    if target_len <= 0:
        return x.astype(np.float32)

    L, C = x.shape
    if L == target_len:
        return x.astype(np.float32)
    if L < 2:
        return np.zeros((target_len, C), dtype=np.float32)

    xp = np.linspace(0.0, 1.0, num=L, dtype=np.float64)
    xq = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)

    y = np.empty((target_len, C), dtype=np.float32)
    for c in range(C):
        y[:, c] = np.interp(xq, xp, x[:, c].astype(np.float64)).astype(np.float32)
    return y


# ==============================
# Chunk helper (time-axis)
# ==============================
def make_chunks_signal(x: np.ndarray, chunk_len: int, stride: int):
    """
    x: (L,C)
    Return list of (chunk_id, start, chunk_array), chunk_array shape=(chunk_len,C)
    - Non-overlapping if stride == chunk_len
    - Drops tail that doesn't fit fully
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (L,C), got {x.shape}")

    if chunk_len <= 0:
        return [(0, 0, x.astype(np.float32))]

    if stride <= 0:
        stride = chunk_len

    L = x.shape[0]
    if L < chunk_len:
        return []

    out = []
    cid = 0
    for start in range(0, L - chunk_len + 1, stride):
        out.append((cid, start, x[start:start + chunk_len].astype(np.float32)))
        cid += 1
    return out


# ==============================
# System spec parsing
# ==============================
def parse_sys(s: str):
    """
    --sys "name=TimeDiff,dir=/path/to/out,recursive=1"
    """
    kv = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()

    if "name" not in kv or "dir" not in kv:
        raise ValueError("Each --sys must include name=... and dir=...")

    recursive = True
    if "recursive" in kv:
        recursive = bool(int(kv["recursive"]))

    return {"name": kv["name"], "dir": kv["dir"], "recursive": recursive}


# ==============================
# Plot helpers (style only)
# ==============================
def _make_color_map(labels):
    uniq = sorted(list(set(labels)))
    cmap = plt.get_cmap("tab10")
    colors = {}
    for i, u in enumerate(uniq):
        colors[u] = cmap(i % 10)
    return colors


def _set_paper_style():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.facecolor": "white",
        "axes.edgecolor": "#000000",
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Serif",
        "legend.frameon": True,
    })


def plot_scatter_2d(
    df: pd.DataFrame, xcol: str, ycol: str, out_png: str, title: str = "",
    pairwise_fixed_colors: bool = False
):
    _set_paper_style()

    labels = df["system"].tolist()
    colors = _make_color_map(labels)

    sys_color_override = {
        "GT": "#F4B400",
        "ImagenI2R": "#79A24B",
        "ImagenTime": "#E45756",
        "TimeDiff": "#9D77C9",
        "FLOW-ITS": "#4C61A8",
    }

    plt.figure(figsize=(9, 7))

    draw_order = ["GT", "ImagenI2R", "ImagenTime", "TimeDiff", "FLOW-ITS"]
    present = df["system"].dropna().unique().tolist()
    ordered_systems = [s for s in draw_order if s in present] + [s for s in present if s not in draw_order]

    pairwise_color_map = {}
    if pairwise_fixed_colors:
        for s in ordered_systems:
            pairwise_color_map[s] = PAIRWISE_GT_COLOR if s == "GT" else PAIRWISE_SYS_COLOR

    for sys_name in ordered_systems:
        sub = df[df["system"] == sys_name]
        if len(sub) == 0:
            continue

        color = (
            pairwise_color_map[sys_name]
            if pairwise_fixed_colors
            else sys_color_override.get(sys_name, colors[sys_name])
        )

        plt.scatter(
            sub[xcol].values,
            sub[ycol].values,
            s=100,
            alpha=0.24,
            label=sys_name,
            color=color,
            edgecolors="none",
            rasterized=True,
        )

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xticks(np.linspace(xmin, xmax, 5))
    ax.set_yticks(np.linspace(ymin, ymax, 5))

    # paper style (hide tick labels)
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

    # If you want title/labels back, uncomment:
    # plt.title(title)
    # plt.xlabel(xcol)
    # plt.ylabel(ycol)
    # plt.legend(markerscale=0.9, fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_gt_vs_sys(
    df: pd.DataFrame, sys_name: str, xcol: str, ycol: str, out_png: str, title: str,
    pairwise_fixed_colors: bool = False
):
    _set_paper_style()

    plt.figure(figsize=(9, 7))
    gt = df[df["system"] == "GT"]
    sy = df[df["system"] == sys_name]

    if pairwise_fixed_colors:
        gt_color = PAIRWISE_GT_COLOR
        sys_color = PAIRWISE_SYS_COLOR
    else:
        sys_color_override = {
            "FLOW-ITS": "#4C78A8",
            "ImagenI2R": "#54A24B",
            "ImagenTime": "#E45756",
            "TimeDiff": "#9D77C9",
        }
        gt_color = "#F4B400"
        sys_color = sys_color_override.get(sys_name, None)

    plt.scatter(
        gt[xcol].values, gt[ycol].values,
        s=24, alpha=0.28, label="GT",
        color=gt_color, edgecolors="none", rasterized=True
    )
    plt.scatter(
        sy[xcol].values, sy[ycol].values,
        s=24, alpha=0.42, label=sys_name,
        color=sys_color, edgecolors="none", rasterized=True
    )

    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend(markerscale=1.8, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ==============================
# PDF metrics + plots
# ==============================
def _hist_density(x: np.ndarray, bins: int, vmin: float, vmax: float):
    """
    Returns:
      centers: (bins,)
      density: (bins,) (integrates ~1 over x-axis)
      probs:   (bins,) (sums to 1)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        raise ValueError("Empty array for histogram.")

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite histogram range.")
    if vmax <= vmin:
        vmax = vmin + 1e-6

    counts, edges = np.histogram(x, bins=bins, range=(vmin, vmax), density=False)
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        probs = np.ones_like(counts) / len(counts)
    else:
        probs = counts / total

    widths = np.diff(edges)
    density = probs / np.maximum(widths, 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float32), density.astype(np.float32), probs.astype(np.float64)


def _kde_curve(x: np.ndarray, vmin: float, vmax: float, n_grid: int = 400):
    """
    Smooth KDE curve for plotting PDF-like density.
    Returns:
      grid: (n_grid,)
      y:    (n_grid,)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite KDE range.")
    if vmax <= vmin:
        vmax = vmin + 1e-6

    grid = np.linspace(vmin, vmax, n_grid, dtype=np.float64)

    # too few points -> fallback zero line
    if len(x) < 2:
        return grid.astype(np.float32), np.zeros_like(grid, dtype=np.float32)

    # constant signal -> gaussian_kde may fail (singular covariance)
    if np.allclose(x, x[0]):
        y = np.zeros_like(grid, dtype=np.float64)
        idx = int(np.argmin(np.abs(grid - x[0])))
        y[max(0, idx - 1): min(len(y), idx + 2)] = 1.0
        return grid.astype(np.float32), y.astype(np.float32)

    # if scipy KDE unavailable, fallback to histogram density interpolated
    if not _HAS_SCIPY_KDE:
        centers, dens, _ = _hist_density(x, bins=128, vmin=vmin, vmax=vmax)
        y = np.interp(grid, centers.astype(np.float64), dens.astype(np.float64))
        return grid.astype(np.float32), y.astype(np.float32)

    try:
        kde = gaussian_kde(x)
        y = kde(grid)
    except Exception:
        # robust fallback
        centers, dens, _ = _hist_density(x, bins=128, vmin=vmin, vmax=vmax)
        y = np.interp(grid, centers.astype(np.float64), dens.astype(np.float64))

    return grid.astype(np.float32), np.asarray(y, dtype=np.float32)


def _js_divergence_from_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence (natural log)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return np.sum(a * np.log(a / b))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(js)


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, bins: int = 128) -> float:
    """
    1D Wasserstein distance.
    Uses scipy if available; otherwise histogram-based approximation.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    if _HAS_SCIPY_WD:
        return float(scipy_wasserstein_distance(x, y))

    # fallback: histogram-based approximation
    vmin = float(min(np.min(x), np.min(y)))
    vmax = float(max(np.max(x), np.max(y)))
    if vmax <= vmin:
        return 0.0

    _, _, px = _hist_density(x, bins=bins, vmin=vmin, vmax=vmax)
    _, _, py = _hist_density(y, bins=bins, vmin=vmin, vmax=vmax)

    # CDF L1 integral approximation
    cdf_x = np.cumsum(px)
    cdf_y = np.cumsum(py)
    bin_width = (vmax - vmin) / bins
    wd = np.sum(np.abs(cdf_x - cdf_y)) * bin_width
    return float(wd)


def save_pdf_evaluation(
    cropped_by_system: Dict[str, List[np.ndarray]],
    out_dir: str,
    bins: int = 128,
    max_points_per_system: int = 0,
    seed: int = 0,
    pairwise_fixed_colors: bool = False,
):
    """
    cropped_by_system[system] = list of arrays, each (L,C), already normalized and cropped to same L0

    Saves:
      - pdf_metrics_channelwise.csv
      - pdf_plots/channel_XX_pdf_all.png (GT + all systems; KDE curve visualization)
      - pdf_plots/channel_XX_gt_vs_<sys>.png (KDE curve visualization)
      - pdf_all_channels_grid.png

    Notes
    -----
    - Metrics (JSD / Wasserstein) are computed on full histogram range [vmin, vmax].
    - Plotting only uses a trimmed x-range (quantile-based) to remove visually useless tails.
    - Legends are intentionally removed for cleaner figures.
    """
    if "GT" not in cropped_by_system or len(cropped_by_system["GT"]) == 0:
        raise ValueError("GT data is required for PDF evaluation.")

    os.makedirs(out_dir, exist_ok=True)
    pdf_plot_dir = os.path.join(out_dir, "pdf_plots")
    os.makedirs(pdf_plot_dir, exist_ok=True)

    # Channel consistency
    all_channels = []
    for sys_name, xs in cropped_by_system.items():
        if len(xs) == 0:
            continue
        chs = sorted(list({x.shape[1] for x in xs}))
        if len(chs) != 1:
            raise ValueError(f"Channel mismatch within system '{sys_name}': {chs}")
        all_channels.append(chs[0])

    if len(all_channels) == 0:
        raise ValueError("No valid data in cropped_by_system.")
    if len(set(all_channels)) != 1:
        raise ValueError(f"Channel mismatch across systems for PDF eval: {sorted(set(all_channels))}")

    C = all_channels[0]
    systems = [s for s in cropped_by_system.keys() if len(cropped_by_system[s]) > 0]

    rng = np.random.default_rng(seed)

    # Flatten per system/channel
    flat_by_system = {}
    for sys_name in systems:
        xs = cropped_by_system[sys_name]
        vals = np.concatenate(xs, axis=0)  # (sumL, C)
        if max_points_per_system and max_points_per_system > 0 and vals.shape[0] > max_points_per_system:
            idx = rng.choice(vals.shape[0], size=max_points_per_system, replace=False)
            vals = vals[idx]
        flat_by_system[sys_name] = vals.astype(np.float32)

    # Plotting style + colors
    _set_paper_style()
    if pairwise_fixed_colors:
        color_map = {s: (PAIRWISE_GT_COLOR if s == "GT" else PAIRWISE_SYS_COLOR) for s in systems}
        fallback_colors = color_map.copy()
    else:
        color_map = {
            "GT": "#F4B400",
            "ImagenI2R": "#79A24B",
            "ImagenTime": "#E45756",
            "TimeDiff": "#9D77C9",
            "FLOW-ITS": "#4C61A8",
        }
        fallback_colors = _make_color_map(systems)

    # Channel-wise metrics + plots
    rows = []
    gt_vals_all = flat_by_system["GT"]  # (N, C)

    # Combined PDF grid figure (all channels in one image)
    ncols = min(4, C) if C > 0 else 1
    nrows = int(np.ceil(C / ncols))
    fig_grid, axes_grid = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    axes_grid = np.array(axes_grid).reshape(-1) if isinstance(axes_grid, np.ndarray) else np.array([axes_grid])

    # plotting-only tail trim quantiles (can tweak)
    qtrim_lo = 0.002
    qtrim_hi = 0.998

    for c in range(C):
        # Determine common histogram range across included systems for this channel (metrics range)
        channel_values = [flat_by_system[s][:, c] for s in systems]
        vmin = float(min(np.min(v) for v in channel_values))
        vmax = float(max(np.max(v) for v in channel_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            continue
        if vmax <= vmin:
            vmax = vmin + 1e-6

        # ---- plotting-only x-range (tail trim) ----
        # metrics keep [vmin, vmax], plotting uses trimmed range
        all_concat = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in channel_values], axis=0)
        all_concat = all_concat[np.isfinite(all_concat)]

        if len(all_concat) > 0:
            q_lo, q_hi = np.quantile(all_concat, [qtrim_lo, qtrim_hi])
            plot_vmin = float(max(vmin, q_lo))
            plot_vmax = float(min(vmax, q_hi))
            if (not np.isfinite(plot_vmin)) or (not np.isfinite(plot_vmax)) or (plot_vmax <= plot_vmin):
                plot_vmin, plot_vmax = vmin, vmax
        else:
            plot_vmin, plot_vmax = vmin, vmax

        # GT histogram/probs baseline (for metrics) -> full range
        _, _, probs_gt = _hist_density(gt_vals_all[:, c], bins=bins, vmin=vmin, vmax=vmax)

        # Plot: all systems on one figure (KDE for visualization) -> trimmed range
        plt.figure(figsize=(9, 6))
        axg = axes_grid[c]

        for sys_name in systems:
            vals = flat_by_system[sys_name][:, c]

            # metrics use histogram probs (stable, full range)
            _, _, probs = _hist_density(vals, bins=bins, vmin=vmin, vmax=vmax)

            # visualization uses smooth KDE on trimmed range
            xg, yg = _kde_curve(vals, vmin=plot_vmin, vmax=plot_vmax, n_grid=400)

            color = color_map.get(sys_name, fallback_colors.get(sys_name, "#333333"))
            lw =4 if sys_name == "GT" else 4
            alpha = 0.95 if sys_name == "GT" else 0.95
            ls = "-"

            plt.plot(xg, yg, color=color, linewidth=lw, alpha=alpha, linestyle=ls)
            axg.plot(xg, yg, color=color, linewidth=lw, alpha=alpha, linestyle=ls)

            # Metrics vs GT (for non-GT systems)
            if sys_name != "GT":
                jsd = _js_divergence_from_probs(probs_gt, probs)
                wd = _wasserstein_1d(gt_vals_all[:, c], vals, bins=bins)
                rows.append({
                    "system": sys_name,
                    "channel": c,
                    "jsd": float(jsd),
                    "wasserstein": float(wd),
                    "gt_mean": float(np.mean(gt_vals_all[:, c])),
                    "sys_mean": float(np.mean(vals)),
                    "gt_std": float(np.std(gt_vals_all[:, c])),
                    "sys_std": float(np.std(vals)),
                    "n_gt_points": int(gt_vals_all.shape[0]),
                    "n_sys_points": int(vals.shape[0]),
                    "hist_bins": int(bins),
                    "range_min": float(vmin),      # metric range
                    "range_max": float(vmax),      # metric range
                    "plot_range_min": float(plot_vmin),  # visualization range
                    "plot_range_max": float(plot_vmax),  # visualization range
                })

        plt.title(f"Channel {c} PDF (KDE)")
        plt.xlabel("value")
        plt.ylabel("density")
        plt.xlim(plot_vmin, plot_vmax)
        # plt.legend(fontsize=9)  # intentionally removed
        plt.tight_layout()
        plt.savefig(os.path.join(pdf_plot_dir, f"channel_{c:02d}_pdf_all.png"), dpi=220)
        plt.close()

        # Grid subplot style (clean)
        axg.set_xlim(plot_vmin, plot_vmax)
        axg.tick_params(axis="both", labelbottom=False, labelleft=False)
        axg.grid(True, alpha=0.2)

        # Optional: GT vs each system (separate plots, KDE) -> trimmed range
        xg_gt, yg_gt = _kde_curve(gt_vals_all[:, c], vmin=plot_vmin, vmax=plot_vmax, n_grid=400)

        for sys_name in systems:
            if sys_name == "GT":
                continue

            vals = flat_by_system[sys_name][:, c]
            _, _, probs_sy = _hist_density(vals, bins=bins, vmin=vmin, vmax=vmax)  # full range metrics
            jsd = _js_divergence_from_probs(probs_gt, probs_sy)
            wd = _wasserstein_1d(gt_vals_all[:, c], vals, bins=bins)

            xg_sy, yg_sy = _kde_curve(vals, vmin=plot_vmin, vmax=plot_vmax, n_grid=400)

            plt.figure(figsize=(8, 5))
            plt.plot(
                xg_gt, yg_gt,
                color=color_map.get("GT", PAIRWISE_GT_COLOR if pairwise_fixed_colors else "#F4B400"),
                linewidth=5,
                linestyle="-",
                alpha=0.95,
            )
            plt.plot(
                xg_sy, yg_sy,
                color=color_map.get(sys_name, PAIRWISE_SYS_COLOR if pairwise_fixed_colors else "#4C61A8"),
                linewidth=5,
                linestyle="-",
                alpha=0.95,
            )
            plt.title(f"Channel {c}: GT vs {sys_name}\nJSD={jsd:.4f}, WD={wd:.4f}")
            plt.xlabel("value")
            plt.ylabel("density")
            plt.xlim(plot_vmin, plot_vmax)
            # plt.legend(fontsize=9)  # intentionally removed
            plt.tight_layout()
            plt.savefig(os.path.join(pdf_plot_dir, f"channel_{c:02d}_gt_vs_{sys_name}.png"), dpi=220)
            plt.close()

    # Finalize combined PDF grid figure (no global legend)
    for k in range(C, len(axes_grid)):
        axes_grid[k].axis("off")

    # fig_grid.legend(...) intentionally removed
    fig_grid.tight_layout(rect=[0, 0, 1, 0.96])
    fig_grid.savefig(os.path.join(out_dir, "pdf_all_channels_grid.png"), dpi=220)
    plt.close(fig_grid)

    # Save metrics
    if len(rows) > 0:
        mdf = pd.DataFrame(rows)
        mdf.to_csv(os.path.join(out_dir, "pdf_metrics_channelwise.csv"), index=False)

        sdf = (
            mdf.groupby("system")[["jsd", "wasserstein"]]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        sdf.columns = [
            col if isinstance(col, str) else f"{col[0]}_{col[1]}"
            for col in sdf.columns
        ]
        sdf.to_csv(os.path.join(out_dir, "pdf_metrics_summary.csv"), index=False)
        print(f"[INFO] saved: {os.path.join(out_dir, 'pdf_metrics_channelwise.csv')}")
        print(f"[INFO] saved: {os.path.join(out_dir, 'pdf_metrics_summary.csv')}")
    else:
        print("[WARN] No PDF metric rows were generated (check systems/inputs).")


# ==============================
# Optional distribution matching helpers (for embeddings)
# ==============================
def blend_signals(x_orig: np.ndarray, x_matched: np.ndarray, alpha: float) -> np.ndarray:
    """
    alpha in [0,1]
    0 -> original, 1 -> fully matched
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    xo = np.asarray(x_orig, dtype=np.float32)
    xm = np.asarray(x_matched, dtype=np.float32)

    if xo.shape != xm.shape:
        raise ValueError(f"blend_signals shape mismatch: {xo.shape} vs {xm.shape}")

    if a <= 0.0:
        return xo.copy()
    if a >= 1.0:
        return xm.copy()
    return ((1.0 - a) * xo + a * xm).astype(np.float32)


def _quantile_match_1d(src: np.ndarray, ref: np.ndarray, n_quantiles: int = 2048) -> np.ndarray:
    """
    Match 1D marginal distribution of src to ref via quantile mapping.
    Preserves shape and finite mask.
    """
    src_arr = np.asarray(src, dtype=np.float64).reshape(-1)
    ref_arr = np.asarray(ref, dtype=np.float64).reshape(-1)

    out = src_arr.copy()

    src_mask = np.isfinite(src_arr)
    ref_mask = np.isfinite(ref_arr)

    src_f = src_arr[src_mask]
    ref_f = ref_arr[ref_mask]

    if len(src_f) == 0 or len(ref_f) == 0:
        return out.astype(np.float32)

    # constant edge cases
    if np.allclose(ref_f, ref_f[0]):
        out[src_mask] = ref_f[0]
        return out.astype(np.float32)

    if np.allclose(src_f, src_f[0]):
        out[src_mask] = np.median(ref_f)
        return out.astype(np.float32)

    n_q = int(min(max(16, n_quantiles), max(len(src_f), len(ref_f))))
    q = np.linspace(0.0, 1.0, num=n_q, dtype=np.float64)

    src_q = np.quantile(src_f, q)
    ref_q = np.quantile(ref_f, q)

    # make non-decreasing for interpolation stability
    src_q = np.maximum.accumulate(src_q)

    # src value -> quantile rank -> ref quantile
    ranks = np.interp(src_f, src_q, q, left=0.0, right=1.0)
    mapped = np.interp(ranks, q, ref_q, left=ref_q[0], right=ref_q[-1])

    out[src_mask] = mapped
    return out.astype(np.float32)


def quantile_match_signal_to_gt(x: np.ndarray, gt_ref_by_channel: List[np.ndarray], n_quantiles: int = 2048) -> np.ndarray:
    """
    x: (L,C)
    gt_ref_by_channel[c]: pooled GT values for channel c (1D)
    returns: (L,C) distribution-matched to GT marginally, channel-wise
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (L,C), got {x.shape}")

    _, C = x.shape
    if len(gt_ref_by_channel) != C:
        raise ValueError(f"Channel mismatch: x has {C}, gt_ref_by_channel has {len(gt_ref_by_channel)}")

    y = np.empty_like(x, dtype=np.float32)
    for c in range(C):
        y[:, c] = _quantile_match_1d(x[:, c], gt_ref_by_channel[c], n_quantiles=n_quantiles)
    return y


# ==============================
# Pairwise subset eval (NEW)
# ==============================
def run_subset_eval(
    subset_samples,   # list of (meta, x) where x is normalized (L,C)
    out_dir: str,
    args,
):
    """
    subset_samples: GT + one system only
    Recompute L0 / PDF / PCA / t-SNE for this subset (pairwise).

    Notes
    -----
    - x in subset_samples is assumed to be already normalized (args.norm).
    - PDF metrics default to pre-matched signals.
    - Dist matching (if enabled) is applied to embeddings stream only by default.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Basic checks
    if len(subset_samples) == 0:
        print(f"[WARN] Empty subset_samples for {out_dir}")
        return

    sys_set = sorted(list({m["system"] for m, _ in subset_samples}))
    if "GT" not in sys_set:
        raise ValueError("run_subset_eval requires GT in subset_samples")
    if len(sys_set) < 2:
        print(f"[WARN] Only one system in subset ({sys_set}); skipping pairwise eval @ {out_dir}")
        return
    if len(sys_set) > 2:
        print(f"[WARN] More than 2 systems in subset ({sys_set}); running on all present systems.")

    # Channel consistency
    chs = sorted(list({m["channels"] for m, _ in subset_samples}))
    if len(chs) != 1:
        raise ValueError(f"Channel mismatch in subset: {chs}")
    C_all = chs[0]

    # Pairwise crop length
    L0 = min(x.shape[0] for _, x in subset_samples)
    if L0 < 2:
        raise ValueError(f"Too short subset L0={L0}")

    if args.chunk_len > 0 and L0 < args.chunk_len:
        raise ValueError(f"subset L0={L0} < chunk_len={args.chunk_len}")

    print(f"[INFO][subset] out_dir={out_dir}")
    print(f"[INFO][subset] systems={sys_set}, channels={C_all}, L0(pairwise)={L0}")

    # Dist matching config (optional)
    dist_match_mode = getattr(args, "dist_match", "none")  # none | affine | quantile
    dist_match_apply_to_pdf = bool(getattr(args, "dist_match_apply_to_pdf", False))
    dist_match_n_quantiles = int(getattr(args, "dist_match_n_quantiles", 2048))

    dist_match_alpha = float(np.clip(getattr(args, "dist_match_alpha", 1.0), 0.0, 1.0))
    dist_match_alpha_pdf_arg = float(getattr(args, "dist_match_alpha_pdf", -1.0))
    if dist_match_alpha_pdf_arg < 0:
        dist_match_alpha_pdf = dist_match_alpha
    else:
        dist_match_alpha_pdf = float(np.clip(dist_match_alpha_pdf_arg, 0.0, 1.0))

    # Build GT references from pairwise-cropped signals
    gt_crops = [x[:L0].astype(np.float32) for meta, x in subset_samples if meta["system"] == "GT"]
    if len(gt_crops) == 0:
        raise RuntimeError("No GT samples after cropping in subset")

    gt_cat = np.concatenate(gt_crops, axis=0).astype(np.float32)  # (sumL, C)
    gt_mu = np.mean(gt_cat, axis=0).astype(np.float32)
    gt_sd = (np.std(gt_cat, axis=0) + 1e-8).astype(np.float32)
    gt_ref_by_channel = [gt_cat[:, c].copy() for c in range(gt_cat.shape[1])]

    # Two streams:
    #   - subset_for_metrics: for PDF/JSD/WD (default: no dist matching)
    #   - subset_for_embed:   for PCA/t-SNE (optional dist matching)
    subset_for_metrics = []
    subset_for_embed = []

    def _affine_match_to_gt(x_arr: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_arr, dtype=np.float32)
        mu = np.mean(x_arr, axis=0).astype(np.float32)
        sd = (np.std(x_arr, axis=0) + 1e-8).astype(np.float32)
        y = (x_arr - mu[None, :]) / sd[None, :]
        y = y * gt_sd[None, :] + gt_mu[None, :]
        return y.astype(np.float32)

    for meta, x in subset_samples:
        x_crop = x[:L0].astype(np.float32)

        # metrics stream (default: original normalized)
        x_metrics = x_crop.copy()

        # embedding stream (optionally matched/blended)
        x_embed = x_crop.copy()

        if meta["system"] != "GT":
            x_dm = x_crop.copy()

            if dist_match_mode == "affine":
                x_dm = _affine_match_to_gt(x_crop)
            elif dist_match_mode == "quantile":
                x_dm = quantile_match_signal_to_gt(
                    x_crop, gt_ref_by_channel, n_quantiles=dist_match_n_quantiles
                ).astype(np.float32)
            elif dist_match_mode in ("none", "", None):
                x_dm = x_crop.copy()
            else:
                print(f"[WARN][subset] Unknown dist_match mode={dist_match_mode}. Using no matching.")
                x_dm = x_crop.copy()

            # Embedding stream: blend original <-> matched
            if dist_match_mode in ("affine", "quantile"):
                x_embed = blend_signals(x_crop, x_dm, dist_match_alpha)

            # Optional PDF stream matching/blending (usually not recommended)
            if dist_match_apply_to_pdf and dist_match_mode in ("affine", "quantile"):
                x_metrics = blend_signals(x_crop, x_dm, dist_match_alpha_pdf)

        subset_for_metrics.append((meta, x_metrics))
        subset_for_embed.append((meta, x_embed))

    if dist_match_mode not in ("none", "", None):
        print(
            f"[INFO][subset] dist_match={dist_match_mode}, "
            f"alpha(embed)={dist_match_alpha:.3f}, "
            f"apply_to_pdf={dist_match_apply_to_pdf}, alpha(pdf)={dist_match_alpha_pdf:.3f}"
        )

    # PDF (pairwise recomputation)
    if args.enable_pdf:
        cropped_by_system = defaultdict(list)
        for meta, x in subset_for_metrics:
            cropped_by_system[meta["system"]].append(x.astype(np.float32))

        save_pdf_evaluation(
            cropped_by_system=cropped_by_system,
            out_dir=out_dir,
            bins=int(args.pdf_bins),
            max_points_per_system=int(args.pdf_max_points_per_system),
            seed=int(args.seed),
            pairwise_fixed_colors=True,
        )

    # Build feature matrix for PCA/t-SNE (points = chunks)
    X_list = []
    data_records = []
    total_chunks = 0
    rng_global = np.random.default_rng(args.seed)

    for meta, x in subset_for_embed:
        sys_name = meta["system"]
        path = meta["path"]
        fname = os.path.basename(path)

        x_crop = x.astype(np.float32)  # already cropped to L0

        chunks = make_chunks_signal(x_crop, args.chunk_len, args.chunk_stride)

        # Optional limit chunks per file
        if args.max_chunks_per_file and args.max_chunks_per_file > 0 and len(chunks) > args.max_chunks_per_file:
            idx = rng_global.choice(len(chunks), size=args.max_chunks_per_file, replace=False)
            idx = sorted(idx.tolist())
            chunks = [chunks[i] for i in idx]

        for (chunk_id, start, chunk) in chunks:
            feat = chunk
            if args.resample_len and args.resample_len > 0:
                feat = resample_signal(feat, args.resample_len)

            feat_vec = feat.reshape(-1).astype(np.float32)
            X_list.append(feat_vec)

            data_records.append({
                "system": sys_name,
                "file": fname,
                "path": path,
                "len_raw": int(meta["len_raw"]),
                "channels": int(meta["channels"]),
                "crop_L0": int(L0),
                "chunk_id": int(chunk_id),
                "chunk_start": int(start),
                "chunk_len": int(chunk.shape[0]),
                "dist_match": str(dist_match_mode),
                "dist_match_alpha": float(dist_match_alpha),
                "dist_match_apply_to_pdf": bool(dist_match_apply_to_pdf),
                "dist_match_alpha_pdf": float(dist_match_alpha_pdf),
            })
            total_chunks += 1

    if total_chunks <= 2:
        print(f"[WARN] Too few points after chunking in subset {out_dir}: {total_chunks}")
        return

    X = np.stack(X_list, axis=0)
    print(f"[INFO][subset] stacked X: {X.shape} (N_points, D) @ {out_dir}")

    if args.chunk_len > 0:
        est = make_chunks_signal(np.zeros((L0, C_all), np.float32), args.chunk_len, args.chunk_stride)
        stride_eff = args.chunk_stride if args.chunk_stride > 0 else args.chunk_len
        print(f"[INFO][subset] chunking=ON -> chunks_per_file ~= {len(est)} "
              f"(chunk_len={args.chunk_len}, stride={stride_eff})")

    # PCA 2D
    pca2 = PCA(n_components=2, random_state=args.seed)
    pca_xy = pca2.fit_transform(X)

    evr = pca2.explained_variance_ratio_
    pd.DataFrame([{
        "pc1_evr": float(evr[0]),
        "pc2_evr": float(evr[1]),
        "dist_match": str(dist_match_mode),
        "dist_match_alpha": float(dist_match_alpha),
        "dist_match_apply_to_pdf": bool(dist_match_apply_to_pdf),
        "dist_match_alpha_pdf": float(dist_match_alpha_pdf),
    }]).to_csv(os.path.join(out_dir, "pca_explained_variance.csv"), index=False)

    # PCA pre-reduction for t-SNE
    pca_dim = int(max(2, min(args.pca_dim_for_tsne, X.shape[1], X.shape[0] - 1)))
    pca_pre = PCA(n_components=pca_dim, random_state=args.seed)
    X_pre = pca_pre.fit_transform(X)

    n_pts = X_pre.shape[0]
    if n_pts < 3:
        raise RuntimeError(f"Too few points for t-SNE in subset: n_points={n_pts}")

    perp = float(args.tsne_perplexity)
    max_valid = (n_pts - 1) - 1e-6
    safe_practical = max(2.0, (n_pts - 1) / 3.0 - 1e-6)
    target_max = min(max_valid, safe_practical)

    if perp >= target_max:
        new_perp = max(2.0, target_max)
        if new_perp >= max_valid:
            new_perp = max(1.0, max_valid - 1e-6)
        print(f"[WARN][subset] tsne_perplexity={perp} too large for n_points={n_pts}. "
              f"Clamping to {new_perp:.3f}.")
        perp = new_perp

    # t-SNE
    print("[INFO][subset] running t-SNE...")
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=float(args.tsne_lr),
        init=args.tsne_init,
        random_state=args.seed,
        verbose=1,
    )

    try:
        tsne = TSNE(**tsne_kwargs, n_iter=int(args.tsne_n_iter))
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=int(args.tsne_n_iter))

    tsne_xy = tsne.fit_transform(X_pre)

    # Save embeddings dataframe
    df = pd.DataFrame(data_records)
    df["pca1"] = pca_xy[:, 0].astype(np.float32)
    df["pca2"] = pca_xy[:, 1].astype(np.float32)
    df["tsne1"] = tsne_xy[:, 0].astype(np.float32)
    df["tsne2"] = tsne_xy[:, 1].astype(np.float32)
    df["norm"] = args.norm
    df["resample_len"] = int(args.resample_len) if (args.resample_len and args.resample_len > 0) else 0
    df["chunk_len_arg"] = int(args.chunk_len)
    df["chunk_stride_arg"] = int(
        args.chunk_stride if args.chunk_stride > 0 else (args.chunk_len if args.chunk_len > 0 else 0)
    )

    out_csv = os.path.join(out_dir, "embeddings_2d.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO][subset] saved: {out_csv}")

    # Scatter plots (pairwise fixed colors)
    plot_scatter_2d(
        df, "pca1", "pca2",
        os.path.join(out_dir, "pca_scatter.png"),
        title=f"PCA 2D (pairwise, L0={L0}, dist_match={dist_match_mode}, alpha={dist_match_alpha:.2f})",
        pairwise_fixed_colors=True,
    )
    plot_scatter_2d(
        df, "tsne1", "tsne2",
        os.path.join(out_dir, "tsne_scatter.png"),
        title=f"t-SNE 2D (pairwise, perp={perp:.2f}, dist_match={dist_match_mode}, alpha={dist_match_alpha:.2f})",
        pairwise_fixed_colors=True,
    )

    # Optional overlay (GT vs sys)
    if args.save_per_system:
        sys_names = sorted([s for s in df["system"].unique().tolist() if s != "GT"])
        for sname in sys_names:
            plot_gt_vs_sys(
                df, sname, "pca1", "pca2",
                os.path.join(out_dir, "pca_gt_vs_sys.png"),
                title=f"PCA: GT vs {sname} (pairwise, dist_match={dist_match_mode}, alpha={dist_match_alpha:.2f})",
                pairwise_fixed_colors=True,
            )
            plot_gt_vs_sys(
                df, sname, "tsne1", "tsne2",
                os.path.join(out_dir, "tsne_gt_vs_sys.png"),
                title=f"t-SNE: GT vs {sname} (pairwise, dist_match={dist_match_mode}, alpha={dist_match_alpha:.2f})",
                pairwise_fixed_colors=True,
            )


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Qualitative PCA/t-SNE/PDF eval for .npy signals (1D or multi-channel)")

    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="qualitative_eval_out")

    parser.add_argument("--n_samples", type=int, default=256, help="files to pick from each folder (0=all)")
    parser.add_argument("--pick_mode", type=str, default="sorted", choices=["sorted", "random"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--recursive", action="store_true", help="GT search recursive")
    parser.add_argument("--norm", type=str, default="zscore", choices=["none", "zscore", "minmax"])

    parser.add_argument("--balance", action="store_true", help="use common N across all systems and GT (recommended)")
    parser.add_argument("--no_balance", action="store_true", help="disable balancing (each system uses its own N)")

    # chunking for PCA/t-SNE points
    parser.add_argument("--chunk_len", type=int, default=0,
                        help="chunk length along time axis (e.g. 512). 0=disable (1 point per file)")
    parser.add_argument("--chunk_stride", type=int, default=0,
                        help="chunk stride along time axis. 0 -> same as chunk_len")
    parser.add_argument("--max_chunks_per_file", type=int, default=0,
                        help="limit chunks per file (0=all). Useful to reduce correlation/runtime")

    # resample for PCA/t-SNE only (time-axis)
    parser.add_argument("--resample_len", type=int, default=0,
                        help="resample each cropped/chunked signal to this time length (0=disable)")

    # shape hint for 2D npy orientation
    parser.add_argument("--seq_len_hint", type=int, default=0,
                        help="If >0, helps infer (L,C) vs (C,L) for 2D npy (e.g. 128)")

    # PCA + t-SNE
    parser.add_argument("--pca_dim_for_tsne", type=int, default=50,
                        help="PCA dims before t-SNE (speed/stability). Typical 30~100")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--tsne_n_iter", type=int, default=1000)
    parser.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"])

    # PDF evaluation
    parser.add_argument("--enable_pdf", action="store_true", help="Compute channel-wise PDF plots + JSD/Wasserstein")
    parser.add_argument("--pdf_bins", type=int, default=128, help="Histogram bins for PDF/JSD")
    parser.add_argument("--pdf_max_points_per_system", type=int, default=0,
                        help="If >0, randomly subsample flattened time points per system for PDF (speed/memory)")

    # Pairwise independent recomputation (GT vs each system)
    parser.add_argument("--pairwise_eval", action="store_true",
                        help="Recompute PCA/t-SNE/PDF independently for each pair: GT vs each system")
    parser.add_argument("--pairwise_only", action="store_true",
                        help="Run only pairwise evals and skip global all-in-one eval")

    # Optional distribution matching for embeddings (mostly visualization)
    parser.add_argument("--dist_match", type=str, default="none", choices=["none", "affine", "quantile"],
                        help="Optional channel-wise distribution matching applied to embeddings stream in pairwise eval")
    parser.add_argument("--dist_match_alpha", type=float, default=1.0,
                        help="Blend ratio for embeddings stream: 0=original, 1=fully matched")
    parser.add_argument("--dist_match_apply_to_pdf", action="store_true",
                        help="Also apply dist-matched/blended signals to PDF metrics in pairwise eval (usually NOT recommended)")
    parser.add_argument("--dist_match_alpha_pdf", type=float, default=-1.0,
                        help="If >=0 and dist_match_apply_to_pdf, use separate alpha for PDF stream. -1 means use dist_match_alpha")
    parser.add_argument("--dist_match_n_quantiles", type=int, default=2048,
                        help="Quantiles used for quantile matching")

    parser.add_argument("--sys", type=str, action="append", required=True,
                        help='e.g. --sys "name=TimeDiff,dir=/path/to/out,recursive=1"')

    parser.add_argument("--save_per_system", action="store_true",
                        help="save GT vs each system overlay scatter plots")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # pairwise_only implies pairwise_eval
    if args.pairwise_only:
        args.pairwise_eval = True

    # Resolve balance flag
    balance = True
    if args.no_balance:
        balance = False
    if args.balance:
        balance = True

    # GT files
    gt_all = list_npy_files(args.gt_dir, recursive=args.recursive)
    if len(gt_all) == 0:
        raise FileNotFoundError(f"No .npy in gt_dir={args.gt_dir}")
    gt_pick = pick_files(gt_all, args.n_samples, args.pick_mode, args.seed)

    systems = [parse_sys(s) for s in args.sys]
    if len(systems) == 0:
        raise ValueError("No --sys provided")

    # Pre-pick system files
    sys_picks = {}
    for sys in systems:
        sys_all = list_npy_files(sys["dir"], recursive=sys["recursive"])
        if len(sys_all) == 0:
            print(f"[WARN] {sys['name']}: no .npy in {sys['dir']} (skip)")
            continue
        sys_pick = pick_files(sys_all, args.n_samples, args.pick_mode, args.seed + 1007)
        sys_picks[sys["name"]] = sys_pick

    if len(sys_picks) == 0:
        raise RuntimeError("No systems had any .npy files")

    # Decide common N (files) for initial loading pool
    if balance:
        N_common = len(gt_pick)
        for _, files in sys_picks.items():
            N_common = min(N_common, len(files))
        if args.n_samples > 0:
            N_common = min(N_common, args.n_samples)
        if N_common <= 0:
            raise RuntimeError("N_common computed as 0")

        gt_use = gt_pick[:N_common]
        for k in list(sys_picks.keys()):
            sys_picks[k] = sys_picks[k][:N_common]
        print(f"[INFO] balance=ON -> N_common(files)={N_common}")
    else:
        gt_use = gt_pick
        print(f"[INFO] balance=OFF -> GT_files={len(gt_use)}; each system uses its own N(files)")

    # Load REAL subset for stats (channel-wise)
    print("[INFO] loading GT for stats...")
    gt_list_raw = [load_signal_npy(p, seq_len_hint=args.seq_len_hint) for p in gt_use]

    # Ensure GT channels are consistent
    gt_channels = sorted(list({x.shape[1] for x in gt_list_raw}))
    if len(gt_channels) != 1:
        raise ValueError(f"GT channel mismatch across files: {gt_channels}")
    print(f"[INFO] GT channels={gt_channels[0]}")

    stats = compute_real_stats(gt_list_raw)
    mean_mean = float(np.mean(stats["mean"]))
    mean_std = float(np.mean(stats["std"]))
    global_min = float(np.min(stats["min"]))
    global_max = float(np.max(stats["max"]))
    print(f"[INFO] REAL stats(ch-wise): mean(avg)={mean_mean:.6g}, std(avg)={mean_std:.6g}, "
          f"min(global)={global_min:.6g}, max(global)={global_max:.6g}")

    # Load all signals (GT + systems), apply norm
    all_arrays = []
    all_meta = []  # dicts: {system, path, len_raw, channels}

    # GT
    for p in gt_use:
        x = load_signal_npy(p, seq_len_hint=args.seq_len_hint)  # (L,C)
        len_raw = x.shape[0]
        ch = x.shape[1]
        x = apply_norm_global(x, stats, args.norm)
        all_arrays.append(x)
        all_meta.append({"system": "GT", "path": p, "len_raw": len_raw, "channels": ch})

    # Systems
    for sys_name, files in sys_picks.items():
        for p in files:
            x = load_signal_npy(p, seq_len_hint=args.seq_len_hint)
            len_raw = x.shape[0]
            ch = x.shape[1]
            x = apply_norm_global(x, stats, args.norm)
            all_arrays.append(x)
            all_meta.append({"system": sys_name, "path": p, "len_raw": len_raw, "channels": ch})

    # Channel consistency across GT + all systems
    channels_set = sorted(list({m["channels"] for m in all_meta}))
    if len(channels_set) != 1:
        raise ValueError(
            f"Channel mismatch across inputs: {channels_set}. "
            f"All compared .npy files must have the same number of channels."
        )
    C_all = channels_set[0]
    print(f"[INFO] channels(all)={C_all}")

    # Group normalized samples by system
    samples_by_system = defaultdict(list)
    for meta, x in zip(all_meta, all_arrays):
        samples_by_system[meta["system"]].append((meta, x))

    # Pairwise eval: GT vs each system (independent recomputation)
    if args.pairwise_eval:
        if "GT" not in samples_by_system or len(samples_by_system["GT"]) == 0:
            raise RuntimeError("GT samples are missing for pairwise evaluation.")

        pairwise_root = os.path.join(args.out_dir, "pairwise")
        os.makedirs(pairwise_root, exist_ok=True)

        sys_names_all = sorted([k for k in samples_by_system.keys() if k != "GT"])
        for sys_name in sys_names_all:
            gt_samples = samples_by_system["GT"]
            sy_samples = samples_by_system[sys_name]

            # Pairwise balancing
            if balance:
                n_pair = min(len(gt_samples), len(sy_samples))
                if n_pair <= 0:
                    print(f"[WARN] pairwise skip {sys_name}: n_pair=0")
                    continue
                gt_use_pair = gt_samples[:n_pair]
                sy_use_pair = sy_samples[:n_pair]
            else:
                gt_use_pair = gt_samples
                sy_use_pair = sy_samples
                if len(gt_use_pair) == 0 or len(sy_use_pair) == 0:
                    print(f"[WARN] pairwise skip {sys_name}: empty side")
                    continue

            subset = gt_use_pair + sy_use_pair
            out_pair = os.path.join(pairwise_root, sys_name)

            print(f"[INFO] pairwise eval -> GT vs {sys_name} "
                  f"(GT={len(gt_use_pair)}, SYS={len(sy_use_pair)})")
            run_subset_eval(subset, out_pair, args)

        if args.pairwise_only:
            print("[INFO] pairwise_only=True -> skipping global all-in-one eval.")
            print("✅ Done.")
            return

    # ==============================
    # Global all-in-one eval (original behavior)
    # ==============================

    # Global crop length across ALL included samples (time-axis)
    L0 = min(x.shape[0] for x in all_arrays)
    if L0 < 2:
        raise ValueError(f"Too short global L0={L0}")
    if args.chunk_len > 0 and L0 < args.chunk_len:
        raise ValueError(
            f"global L0={L0} < chunk_len={args.chunk_len}. "
            f"Use smaller chunk_len or ensure longer signals."
        )
    print(f"[INFO] global crop length L0={L0}")

    # Build cropped_by_system for PDF (before chunking/resampling)
    cropped_by_system = defaultdict(list)
    for meta, x in zip(all_meta, all_arrays):
        x_crop = x[:L0].astype(np.float32)  # (L0,C)
        cropped_by_system[meta["system"]].append(x_crop)

    # PDF evaluation (channel-wise)
    if args.enable_pdf:
        print("[INFO] running PDF evaluation...")
        save_pdf_evaluation(
            cropped_by_system=cropped_by_system,
            out_dir=args.out_dir,
            bins=int(args.pdf_bins),
            max_points_per_system=int(args.pdf_max_points_per_system),
            seed=int(args.seed),
            pairwise_fixed_colors=False,
        )

    # Build feature matrix for PCA/t-SNE (points = chunks)
    X_list = []
    data_records = []
    total_chunks = 0

    rng_global = np.random.default_rng(args.seed)

    for meta, x in zip(all_meta, all_arrays):
        sys_name = meta["system"]
        path = meta["path"]
        fname = os.path.basename(path)

        x_crop = x[:L0].astype(np.float32)  # (L0,C)

        chunks = make_chunks_signal(x_crop, args.chunk_len, args.chunk_stride)

        # Optional limit chunks per file
        if args.max_chunks_per_file and args.max_chunks_per_file > 0 and len(chunks) > args.max_chunks_per_file:
            idx = rng_global.choice(len(chunks), size=args.max_chunks_per_file, replace=False)
            idx = sorted(idx.tolist())
            chunks = [chunks[i] for i in idx]

        for (chunk_id, start, chunk) in chunks:  # chunk: (Lc,C)
            feat = chunk
            if args.resample_len and args.resample_len > 0:
                feat = resample_signal(feat, args.resample_len)  # (Lr,C)

            feat_vec = feat.reshape(-1).astype(np.float32)  # (L*C,)
            X_list.append(feat_vec)

            data_records.append({
                "system": sys_name,
                "file": fname,
                "path": path,
                "len_raw": int(meta["len_raw"]),
                "channels": int(meta["channels"]),
                "crop_L0": int(L0),
                "chunk_id": int(chunk_id),
                "chunk_start": int(start),
                "chunk_len": int(chunk.shape[0]),
            })
            total_chunks += 1

    if total_chunks <= 2:
        raise RuntimeError(f"Too few points after chunking: {total_chunks}")

    X = np.stack(X_list, axis=0)  # (N_points, D)
    print(f"[INFO] stacked X: {X.shape} (N_points, D=flattened L*C)")

    if args.chunk_len > 0:
        est = make_chunks_signal(np.zeros((L0, C_all), np.float32), args.chunk_len, args.chunk_stride)
        stride_eff = args.chunk_stride if args.chunk_stride > 0 else args.chunk_len
        print(f"[INFO] chunking=ON -> chunks_per_file ~= {len(est)} "
              f"(chunk_len={args.chunk_len}, stride={stride_eff})")

    # PCA 2D
    print("[INFO] running PCA...")
    pca2 = PCA(n_components=2, random_state=args.seed)
    pca_xy = pca2.fit_transform(X)

    evr = pca2.explained_variance_ratio_
    pd.DataFrame([{
        "pc1_evr": float(evr[0]),
        "pc2_evr": float(evr[1]),
    }]).to_csv(os.path.join(args.out_dir, "pca_explained_variance.csv"), index=False)

    # PCA pre-reduction for t-SNE
    pca_dim = int(max(2, min(args.pca_dim_for_tsne, X.shape[1], X.shape[0] - 1)))
    pca_pre = PCA(n_components=pca_dim, random_state=args.seed)
    X_pre = pca_pre.fit_transform(X)

    # t-SNE safety
    n_pts = X_pre.shape[0]
    if n_pts < 3:
        raise RuntimeError(f"Too few points for t-SNE: n_points={n_pts}")

    perp = float(args.tsne_perplexity)
    max_valid = (n_pts - 1) - 1e-6
    safe_practical = max(2.0, (n_pts - 1) / 3.0 - 1e-6)
    target_max = min(max_valid, safe_practical)

    if perp >= target_max:
        new_perp = max(2.0, target_max)
        if new_perp >= max_valid:
            new_perp = max(1.0, max_valid - 1e-6)
        print(f"[WARN] tsne_perplexity={perp} too large for n_points={n_pts}. "
              f"Clamping to {new_perp:.3f}.")
        perp = new_perp

    # t-SNE
    print("[INFO] running t-SNE (this can be slow)...")
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=float(args.tsne_lr),
        init=args.tsne_init,
        random_state=args.seed,
        verbose=1,
    )

    try:
        tsne = TSNE(**tsne_kwargs, n_iter=int(args.tsne_n_iter))
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=int(args.tsne_n_iter))

    tsne_xy = tsne.fit_transform(X_pre)

    # Save embeddings dataframe
    df = pd.DataFrame(data_records)
    df["pca1"] = pca_xy[:, 0].astype(np.float32)
    df["pca2"] = pca_xy[:, 1].astype(np.float32)
    df["tsne1"] = tsne_xy[:, 0].astype(np.float32)
    df["tsne2"] = tsne_xy[:, 1].astype(np.float32)
    df["norm"] = args.norm
    df["resample_len"] = int(args.resample_len) if (args.resample_len and args.resample_len > 0) else 0
    df["chunk_len_arg"] = int(args.chunk_len)
    df["chunk_stride_arg"] = int(
        args.chunk_stride if args.chunk_stride > 0 else (args.chunk_len if args.chunk_len > 0 else 0)
    )

    out_csv = os.path.join(args.out_dir, "embeddings_2d.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] saved: {out_csv}")

    # Scatter plots (global colors)
    plot_scatter_2d(
        df, "pca1", "pca2",
        os.path.join(args.out_dir, "pca_scatter.png"),
        title=f"PCA 2D (norm={args.norm}, L0={L0}, chunk={args.chunk_len}, "
              f"stride={df['chunk_stride_arg'].iloc[0]}, resample={df['resample_len'].iloc[0]})",
        pairwise_fixed_colors=False,
    )
    plot_scatter_2d(
        df, "tsne1", "tsne2",
        os.path.join(args.out_dir, "tsne_scatter.png"),
        title=f"t-SNE 2D (perp={perp:.2f}, lr={args.tsne_lr}, it={args.tsne_n_iter})",
        pairwise_fixed_colors=False,
    )

    # Per-system overlays (GT vs sys) on global embedding
    if args.save_per_system:
        sys_names = sorted([s for s in df["system"].unique().tolist() if s != "GT"])
        for sname in sys_names:
            out_dir_sys = os.path.join(args.out_dir, sname)
            os.makedirs(out_dir_sys, exist_ok=True)

            plot_gt_vs_sys(
                df, sname, "pca1", "pca2",
                os.path.join(out_dir_sys, "pca_gt_vs_sys.png"),
                title=f"PCA: GT vs {sname}",
                pairwise_fixed_colors=False,
            )
            plot_gt_vs_sys(
                df, sname, "tsne1", "tsne2",
                os.path.join(out_dir_sys, "tsne_gt_vs_sys.png"),
                title=f"t-SNE: GT vs {sname}",
                pairwise_fixed_colors=False,
            )

    print("✅ Done.")


if __name__ == "__main__":
    main()