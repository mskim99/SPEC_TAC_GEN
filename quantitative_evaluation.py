# eval_all_metrics_multichannel.py
# =========================================================
# Evaluate many result folders against GT, WITHOUT sample matching.
# Multi-channel version combining Long-range metrics, c-FID, and Correlation.
#
# Features:
#   - Long-range metrics (per-channel average or joint)
#   - Context FID (c-FID) using REAL-only trained S4 context encoder
#   - Pearson correlation with arbitrary pairing (index or random)
#     - Multi-channel correlation with reduction (mean/median)
#     - Optional per-channel correlation saving
#
# Output:
#   out_dir/summary_all_systems.csv
#   out_dir/<system_name>/metrics_long_range_plus_cfid_corr.csv
#   out_dir/<system_name>/pairwise_correlation.csv   (only if --do_corr)
# =========================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metrics.metrics_long_range import compute_all_metrics, setup_optimizer
from models.testing_models.s4d import S4D, dropout_fn


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
# Multi-channel loading
# ==============================
def load_signal_npy(path: str, channel_first: bool = False) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    x = np.asarray(x).squeeze()

    if x.ndim == 1:
        x = x[:, None]  # (L,1)
    elif x.ndim == 2:
        if channel_first:
            x = x.T  # (L,C)
    else:
        raise ValueError(f"Expected 1D or 2D array. Got {x.shape} at {path}")

    if x.shape[0] < 2:
        raise ValueError(f"Too short sequence length {x.shape[0]} at {path}")

    return x.astype(np.float32)


# ==============================
# Per-channel normalization (REAL stats)
# ==============================
def compute_real_stats(real_list):
    cat = np.concatenate(real_list, axis=0).astype(np.float64)  # (sumL, C)
    mu = cat.mean(axis=0)
    sd = cat.std(axis=0) + 1e-8
    mn = cat.min(axis=0)
    mx = cat.max(axis=0)
    return {"mean": mu, "std": sd, "min": mn, "max": mx}


def apply_norm_global(x: np.ndarray, stats: dict, mode: str) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "none":
        return x
    if mode == "zscore":
        return ((x - stats["mean"]) / stats["std"]).astype(np.float32)
    if mode == "minmax":
        mn, mx = stats["min"], stats["max"]
        denom = (mx - mn) + 1e-12
        y = (x - mn) / denom
        return (y * 2.0 - 1.0).astype(np.float32)
    raise ValueError(f"Unknown norm mode: {mode}")


# ==============================
# Correlation
# ==============================
def pearson_corr_1d(a, b, eps=1e-8):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = (float(np.sqrt(np.sum(a * a))) * float(np.sqrt(np.sum(b * b)))) + eps
    if denom <= eps:
        return np.nan
    return float(np.sum(a * b) / denom)


def pearson_corr_multich(a_lc, b_lc, reduce="mean"):
    assert a_lc.ndim == 2 and b_lc.ndim == 2
    assert a_lc.shape == b_lc.shape
    C = a_lc.shape[1]

    per_ch = [pearson_corr_1d(a_lc[:, c], b_lc[:, c]) for c in range(C)]
    arr = np.array(per_ch, dtype=np.float64)

    if reduce == "mean":
        red = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else np.nan
    elif reduce == "median":
        red = float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else np.nan
    else:
        raise ValueError(f"Unknown reduce: {reduce}")

    return red, per_ch


# ==============================
# Context FID (c-FID) - Multi-channel (joint)
# ==============================
class S4ContextEncoder(nn.Module):
    def __init__(
        self,
        d_input=1,
        d_state=16,
        d_model=32,
        n_layers=2,
        dropout=0.0,
        prenorm=False,
        bidirectional=False,
        emb_dim=64,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    d_model=d_model,
                    d_state=d_state,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    transposed=True,
                    lr=0.001,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout) if dropout > 0 else nn.Identity())

        self.emb_head = nn.Sequential(
            nn.Linear(d_model, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.pred_head = nn.Linear(d_model, d_input)

    def forward(self, x):
        B, L, C = x.shape
        h = self.encoder(x)                 # (B,L,d_model)
        h = h.transpose(-1, -2)             # (B,d_model,L)

        for layer, norm, drop in zip(self.s4_layers, self.norms, self.dropouts):
            z = h
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = drop(z)
            h = z + h
            if not self.prenorm:
                h = norm(h.transpose(-1, -2)).transpose(-1, -2)

        h = h.transpose(-1, -2)             # (B,L,d_model)

        pooled = h.mean(dim=1)              # (B,d_model)
        emb = self.emb_head(pooled)         # (B,emb_dim)
        pred_next = self.pred_head(h[:, :-1])  # (B,L-1,C)
        return emb, pred_next

    @torch.no_grad()
    def encode(self, x):
        emb, _ = self.forward(x)
        return emb


def _fid_frechet(mu1, cov1, mu2, cov2, eps=1e-6):
    mu1 = mu1.astype(np.float64)
    mu2 = mu2.astype(np.float64)
    cov1 = cov1.astype(np.float64)
    cov2 = cov2.astype(np.float64)

    diff = mu1 - mu2
    cov1 = cov1 + np.eye(cov1.shape[0]) * eps
    cov2 = cov2 + np.eye(cov2.shape[0]) * eps

    A = cov1 @ cov2
    A = (A + A.T) * 0.5

    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    sqrtA = (V * np.sqrt(w)[None, :]) @ V.T

    fid = float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(sqrtA))
    return fid


def compute_context_fid_s4(
    x_real_t, x_fake_t,
    device,
    context_frac=0.5,
    emb_dim=64,
    d_model=32,
    n_layers=2,
    d_state=16,
    dropout=0.0,
    epochs=30,
    batch_size=128,
    lr=0.01,
    weight_decay=0.0,
):
    assert x_real_t.ndim == 3 and x_fake_t.ndim == 3
    N, L0, C = x_real_t.shape
    assert x_fake_t.shape[0] == N and x_fake_t.shape[1] == L0 and x_fake_t.shape[2] == C

    Lc = int(max(2, round(L0 * float(context_frac))))
    xr_ctx = x_real_t[:, :Lc].contiguous()
    xf_ctx = x_fake_t[:, :Lc].contiguous()

    model = S4ContextEncoder(
        d_input=C,
        d_state=d_state,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        prenorm=False,
        bidirectional=False,
        emb_dim=emb_dim,
    ).to(device)

    ds = TensorDataset(xr_ctx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt, _ = setup_optimizer(model, lr=lr, weight_decay=weight_decay, epochs=epochs)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        for (xb,) in dl:
            xb = xb.to(device)
            _, pred_next = model(xb)
            target = xb[:, 1:]  # (B,L-1,C)
            loss = loss_fn(pred_next, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        er, ef = [], []
        bs = batch_size
        for s in range(0, xr_ctx.shape[0], bs):
            er.append(model.encode(xr_ctx[s:s+bs].to(device)).detach().cpu())
            ef.append(model.encode(xf_ctx[s:s+bs].to(device)).detach().cpu())
        er = torch.cat(er, dim=0).numpy()
        ef = torch.cat(ef, dim=0).numpy()

    mu_r = np.mean(er, axis=0)
    mu_f = np.mean(ef, axis=0)
    cov_r = np.cov(er, rowvar=False)
    cov_f = np.cov(ef, rowvar=False)

    cfid = _fid_frechet(mu_r, cov_r, mu_f, cov_f)
    return float(cfid), int(Lc)


# ==============================
# System spec parsing
# ==============================
def parse_sys(s: str):
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
# Long-range metrics helpers
# ==============================
def compute_metrics_multichannel(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    device: torch.device,
    metrics_mode: str = "per_channel",
):
    assert x_real.ndim == 3 and x_fake.ndim == 3
    N, L, C = x_real.shape
    assert x_fake.shape == (N, L, C)

    if metrics_mode == "joint":
        scores = compute_all_metrics(
            x_real.to(device),
            x_fake.to(device),
            setup_optimizer,
            torch.nn.Identity(),
            device
        )
        return {k: float(v) for k, v in scores.items()}

    if metrics_mode != "per_channel":
        raise ValueError(f"Unknown metrics_mode: {metrics_mode}")

    all_scores = []
    for c in range(C):
        sc = compute_all_metrics(
            x_real[:, :, c:c+1].to(device),
            x_fake[:, :, c:c+1].to(device),
            setup_optimizer,
            torch.nn.Identity(),
            device
        )
        all_scores.append({k: float(v) for k, v in sc.items()})

    keys = sorted(all_scores[0].keys())
    out = {}
    for k in keys:
        out[k] = float(np.mean([d[k] for d in all_scores]))
    return out


# ==============================
# Core evaluation
# ==============================
def evaluate_system(
    sys_name: str,
    gt_files: list,
    sys_files: list,
    out_dir_sys: str,
    device: torch.device,
    norm: str,
    channel_first: bool,
    metrics_mode: str,

    # c-FID options
    cfid_context_frac: float,
    cfid_emb_dim: int,
    cfid_d_model: int,
    cfid_layers: int,
    cfid_d_state: int,
    cfid_dropout: float,
    cfid_epochs: int,
    cfid_batch: int,
    cfid_lr: float,
    cfid_weight_decay: float,
    
    # Correlation options
    do_corr: bool,
    corr_mode: str,
    corr_reduce: str,
    save_per_channel: bool,
    seed: int,
):
    os.makedirs(out_dir_sys, exist_ok=True)

    gt_list = [load_signal_npy(p, channel_first=channel_first) for p in gt_files]
    fk_list = [load_signal_npy(p, channel_first=channel_first) for p in sys_files]

    C = gt_list[0].shape[1]
    if any(x.shape[1] != C for x in gt_list):
        raise ValueError(f"[{sys_name}] GT channel mismatch in selected files.")
    if any(x.shape[1] != C for x in fk_list):
        raise ValueError(f"[{sys_name}] Fake channel mismatch (expected C={C}).")

    stats = compute_real_stats(gt_list)
    gt_list = [apply_norm_global(x, stats, norm) for x in gt_list]
    fk_list = [apply_norm_global(x, stats, norm) for x in fk_list]

    L0 = min(min(x.shape[0] for x in gt_list), min(x.shape[0] for x in fk_list))
    if L0 < 2:
        raise ValueError(f"[{sys_name}] Too short L0={L0}")

    gt_arr = np.stack([x[:L0, :] for x in gt_list], axis=0)  # (N,L0,C)
    fk_arr = np.stack([x[:L0, :] for x in fk_list], axis=0)  # (N,L0,C)

    x_real = torch.tensor(gt_arr, dtype=torch.float32)
    x_fake = torch.tensor(fk_arr, dtype=torch.float32)

    # 1. Long-range metrics
    scores = compute_metrics_multichannel(
        x_real=x_real,
        x_fake=x_fake,
        device=device,
        metrics_mode=metrics_mode
    )

    # 2. c-FID (joint over channels)
    cfid, Lc = compute_context_fid_s4(
        x_real_t=x_real.to(device),
        x_fake_t=x_fake.to(device),
        device=device,
        context_frac=cfid_context_frac,
        emb_dim=cfid_emb_dim,
        d_model=cfid_d_model,
        n_layers=cfid_layers,
        d_state=cfid_d_state,
        dropout=cfid_dropout,
        epochs=cfid_epochs,
        batch_size=cfid_batch,
        lr=cfid_lr,
        weight_decay=cfid_weight_decay,
    )

    # 3. Correlation (optional, arbitrary pairing)
    corr_mean, corr_std = np.nan, np.nan
    corr_rows = []
    
    if do_corr:
        rng = np.random.default_rng(seed)
        idxs = np.arange(len(gt_list))
        if corr_mode == "random":
            rng.shuffle(idxs)

        corr_vals = []
        for i, j in enumerate(idxs):
            a = gt_list[i][:L0, :]  # (L0,C)
            b = fk_list[int(j)][:L0, :]
            
            c_red, c_ch = pearson_corr_multich(a, b, reduce=corr_reduce)
            corr_vals.append(c_red)
            
            row = {
                "gt_index": int(i),
                "fake_index": int(j),
                "gt_file": os.path.basename(gt_files[i]),
                "fake_file": os.path.basename(sys_files[int(j)]),
                "len_used": int(L0),
                "channels": int(C),
                f"corr_{corr_reduce}": c_red
            }
            
            if save_per_channel:
                for k, v in enumerate(c_ch):
                    row[f"corr_ch{k:03d}"] = v
                    
            corr_rows.append(row)
            
        corr_np = np.array(corr_vals, dtype=np.float64)
        corr_mean = float(np.nanmean(corr_np)) if np.any(~np.isnan(corr_np)) else np.nan
        corr_std  = float(np.nanstd(corr_np)) if np.any(~np.isnan(corr_np)) else np.nan
        
        pd.DataFrame(corr_rows).to_csv(os.path.join(out_dir_sys, "pairwise_correlation.csv"), index=False)

    # Summary integration
    summary = {
        "system": sys_name,
        "n": int(len(gt_list)),
        "length_L0": int(L0),
        "channels_C": int(C),
        "norm": norm,
        "metrics_mode": metrics_mode,

        "real_mean": float(np.mean(stats["mean"])),
        "real_std": float(np.mean(stats["std"])),
        "real_min": float(np.min(stats["min"])),
        "real_max": float(np.max(stats["max"])),

        **{k: float(v) for k, v in scores.items()},

        "cFID_context": float(cfid),
        "cFID_context_len": int(Lc),
        "cFID_emb_dim": int(cfid_emb_dim),
    }
    
    if do_corr:
        summary["corr_mean"] = float(corr_mean) if corr_mean == corr_mean else np.nan
        summary["corr_std"] = float(corr_std) if corr_std == corr_std else np.nan
        summary["corr_mode"] = corr_mode
        summary["corr_reduce"] = corr_reduce

    pd.DataFrame([summary]).to_csv(os.path.join(out_dir_sys, "metrics_long_range_plus_cfid_corr.csv"), index=False)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate many folders vs GT (Long-range + cFID + Correlation)")

    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eval_many_results_combined")

    parser.add_argument("--n_samples", type=int, default=128, help="samples to pick from each folder (0=all)")
    parser.add_argument("--pick_mode", type=str, default="sorted", choices=["sorted", "random"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--recursive", action="store_true", help="GT search recursive")
    parser.add_argument("--norm", type=str, default="none", choices=["none", "zscore", "minmax"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--channel_first", action="store_true", help="Interpret each .npy sample as (C,L) instead of (L,C)")
    parser.add_argument("--metrics_mode", type=str, default="joint", choices=["per_channel", "joint"])

    # Correlation options
    parser.add_argument("--do_corr", action="store_true", help="compute Pearson corr with arbitrary pairing")
    parser.add_argument("--corr_mode", type=str, default="index", choices=["index", "random"])
    parser.add_argument("--corr_reduce", type=str, default="mean", choices=["mean", "median"], help="reduce per-channel correlations")
    parser.add_argument("--save_per_channel", action="store_true", help="Save corr per channel columns in pairwise_correlation.csv")

    # Systems
    parser.add_argument("--sys", type=str, action="append", required=True, help='e.g. --sys "name=vocoder,dir=/path/to/vocoder_out,recursive=1"')

    # c-FID options
    parser.add_argument("--cfid_context_frac", type=float, default=0.5)
    parser.add_argument("--cfid_emb_dim", type=int, default=64)
    parser.add_argument("--cfid_d_model", type=int, default=32)
    parser.add_argument("--cfid_layers", type=int, default=2)
    parser.add_argument("--cfid_d_state", type=int, default=16)
    parser.add_argument("--cfid_dropout", type=float, default=0.0)
    parser.add_argument("--cfid_epochs", type=int, default=30)
    parser.add_argument("--cfid_batch", type=int, default=128)
    parser.add_argument("--cfid_lr", type=float, default=0.01)
    parser.add_argument("--cfid_weight_decay", type=float, default=0.0)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    gt_all = list_npy_files(args.gt_dir, recursive=args.recursive)
    if len(gt_all) == 0:
        raise FileNotFoundError(f"No .npy in gt_dir={args.gt_dir}")

    gt_pick = pick_files(gt_all, args.n_samples, args.pick_mode, args.seed)
    systems = [parse_sys(s) for s in args.sys]
    all_summaries = []

    print(f"\nGT: {args.gt_dir}")
    print(f"GT files: {len(gt_all)}  -> picked: {len(gt_pick)}  (mode={args.pick_mode}, n_samples={args.n_samples})")
    print(f"norm={args.norm}, device={dev}, channel_first={args.channel_first}, metrics_mode={args.metrics_mode}")
    if args.do_corr:
        print(f"corr_mode={args.corr_mode}, corr_reduce={args.corr_reduce}, save_per_channel={args.save_per_channel}")
    print(f"systems={len(systems)}")

    for sys in systems:
        sys_all = list_npy_files(sys["dir"], recursive=sys["recursive"])
        if len(sys_all) == 0:
            print(f"[WARN] {sys['name']}: no .npy in {sys['dir']} (skip)")
            continue

        N = min(len(gt_pick), len(sys_all))
        if args.n_samples > 0:
            N = min(N, args.n_samples)

        gt_files = gt_pick[:N]
        sys_files = pick_files(sys_all, N, args.pick_mode, args.seed + 1007)

        print(f"\n[{sys['name']}] files={len(sys_all)} -> picked={len(sys_files)}  (N={N})")

        out_dir_sys = os.path.join(args.out_dir, sys["name"])
        summ = evaluate_system(
            sys_name=sys["name"],
            gt_files=gt_files,
            sys_files=sys_files,
            out_dir_sys=out_dir_sys,
            device=dev,
            norm=args.norm,
            channel_first=args.channel_first,
            metrics_mode=args.metrics_mode,

            # c-FID
            cfid_context_frac=args.cfid_context_frac,
            cfid_emb_dim=args.cfid_emb_dim,
            cfid_d_model=args.cfid_d_model,
            cfid_layers=args.cfid_layers,
            cfid_d_state=args.cfid_d_state,
            cfid_dropout=args.cfid_dropout,
            cfid_epochs=args.cfid_epochs,
            cfid_batch=args.cfid_batch,
            cfid_lr=args.cfid_lr,
            cfid_weight_decay=args.cfid_weight_decay,
            
            # Correlation
            do_corr=args.do_corr,
            corr_mode=args.corr_mode,
            corr_reduce=args.corr_reduce,
            save_per_channel=args.save_per_channel,
            seed=args.seed,
        )
        all_summaries.append(summ)

        print(f"  -> saved: {out_dir_sys}")
        print(f"  cFID={summ['cFID_context']:.6f}, L0={summ['length_L0']}, C={summ['channels_C']}")
        if args.do_corr:
            print(f"  corr_{args.corr_reduce}={summ['corr_mean']:.6f}")

    if len(all_summaries) == 0:
        print("\n⚠️ No systems evaluated.")
        return

    df = pd.DataFrame(all_summaries)
    df.to_csv(os.path.join(args.out_dir, "summary_all_systems.csv"), index=False)
    print(f"\n✅ Done. Summary saved to: {os.path.join(args.out_dir, 'summary_all_systems.csv')}")


if __name__ == "__main__":
    main()