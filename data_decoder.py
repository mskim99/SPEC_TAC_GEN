# data_decoder.py (MULTI-CHANNEL READY)
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _collect_wave_files(wave_dir: str, recursive: bool = True):
    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(glob.glob(os.path.join(wave_dir, pattern), recursive=recursive))


def _material_from_folder(wave_dir: str, wpath: str) -> str:
    """
    material = relpath의 첫 번째 폴더명
    flat dataset이면 'unknown'
    """
    rel = os.path.relpath(wpath, wave_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def _load_triplet_paths(wave_dir, feat_root, recursive=True):
    mag_dir = os.path.join(feat_root, "mag")
    re_dir  = os.path.join(feat_root, "re")
    im_dir  = os.path.join(feat_root, "im")

    wave_files = _collect_wave_files(wave_dir, recursive=recursive)
    items = []
    for w in wave_files:
        rel = os.path.relpath(w, wave_dir)
        rel_dir = os.path.dirname(rel)
        base = os.path.splitext(os.path.basename(w))[0]

        m  = os.path.join(mag_dir, rel_dir, f"{base}_mag.npy")
        re = os.path.join(re_dir,  rel_dir, f"{base}_re.npy")
        im = os.path.join(im_dir,  rel_dir, f"{base}_im.npy")

        if os.path.exists(m) and os.path.exists(re) and os.path.exists(im):
            items.append((w, m, re, im))

    if len(items) == 0:
        raise FileNotFoundError("No matched (wave, mag, re, im) pairs (check feat_root structure).")
    return items


def _to_time_channel(y: np.ndarray) -> np.ndarray:
    """
    y -> (T,C) float32
    supports (T,), (T,C), (C,T)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.float32)[:, None]
    if y.ndim != 2:
        raise ValueError(f"waveform must be 1D or 2D. got {y.shape}")

    y = y.astype(np.float32)
    # heuristic: smaller dim is channel
    if y.shape[0] <= y.shape[1] and y.shape[0] <= 64 and y.shape[1] > y.shape[0]:
        return y.T  # (C,T)->(T,C)
    return y       # assume (T,C)


def _crop_or_pad_tc(y_tc: np.ndarray, start: int, length: int) -> np.ndarray:
    """
    y_tc: (T,C)
    returns (length,C)
    """
    T, C = y_tc.shape
    end = start + length

    if start < 0:
        y_tc = np.pad(y_tc, ((abs(start), 0), (0, 0)), mode="constant")
        start = 0
        end = start + length

    if end > y_tc.shape[0]:
        y_tc = np.pad(y_tc, ((0, end - y_tc.shape[0]), (0, 0)), mode="constant")

    return y_tc[start:end].astype(np.float32)


def _ensure_feat_ncfw(x: np.ndarray, name: str) -> np.ndarray:
    """
    feature loader normalize shape:
      old single-channel saved: (N,F,W)      -> (N,1,F,W)
      new multi-channel saved:  (N,C,F,W)    -> as is
    """
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[:, None, :, :]  # (N,1,F,W)
    elif x.ndim != 4:
        raise ValueError(f"{name} must have ndim 3 or 4. got shape={x.shape}")
    return x.astype(np.float32)


def _z_cache_path_from_wave(z_dir: str, wave_dir: str, wpath: str) -> str:
    """
    relpath-preserving z cache path (avoid basename collision)
    """
    rel = os.path.relpath(wpath, wave_dir)
    rel_dir = os.path.dirname(rel)
    base = os.path.splitext(os.path.basename(wpath))[0]
    out_dir = os.path.join(z_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_z.npy")


class SpecWaveWindowTripletDataset(Dataset):
    """
    Returns:
      mag/re/im windows: (K,C,F,W)
      positive / negative same
      y_seg: (K*hop, C)
      mat_id: scalar long
    """
    def __init__(self, wave_dir, feat_root, hop=256, window_k=8, neg_margin=16, seed=0, recursive=True):
        self.wave_dir = wave_dir
        self.feat_root = feat_root
        self.items = _load_triplet_paths(wave_dir, feat_root, recursive=recursive)
        self.wave_files = [it[0] for it in self.items]

        mats = [_material_from_folder(wave_dir, w) for (w, *_rest) in self.items]
        self.materials = sorted(list(set(mats)))
        self.material_to_id = {m: i for i, m in enumerate(self.materials)}

        self.hop = int(hop)
        self.K = int(window_k)
        self.neg_margin = int(neg_margin)
        self.rng = np.random.default_rng(seed)

        # infer spec shape / num_channels
        _, m0, r0, i0 = self.items[0]
        mag0 = _ensure_feat_ncfw(np.load(m0), "mag")
        self.num_channels = int(mag0.shape[1])
        self.spec_f = int(mag0.shape[2])
        self.spec_w = int(mag0.shape[3])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wpath, mpath, repath, impath = self.items[idx]
        mat = _material_from_folder(self.wave_dir, wpath)
        mat_id = np.int64(self.material_to_id.get(mat, 0))

        y_tc = _to_time_channel(np.load(wpath).astype(np.float32))  # (T,Cy)
        mag = _ensure_feat_ncfw(np.load(mpath), "mag")              # (N,C,F,W)
        re  = _ensure_feat_ncfw(np.load(repath), "re")
        im  = _ensure_feat_ncfw(np.load(impath), "im")

        if not (mag.shape == re.shape == im.shape):
            raise ValueError(f"feature shape mismatch for {wpath}: {mag.shape}, {re.shape}, {im.shape}")

        Ffrm, C, Fbin, Wbin = mag.shape
        K = self.K

        if Ffrm < K + 1:
            pad = (K + 1) - Ffrm
            mag = np.concatenate([mag, np.repeat(mag[-1:], pad, axis=0)], axis=0)
            re  = np.concatenate([re,  np.repeat(re[-1:],  pad, axis=0)], axis=0)
            im  = np.concatenate([im,  np.repeat(im[-1:],  pad, axis=0)], axis=0)
            Ffrm = mag.shape[0]

        s_max = Ffrm - K
        s = int(self.rng.integers(0, max(1, s_max)))
        sp = min(s + K, Ffrm - K)

        tries = 0
        while True:
            sn = int(self.rng.integers(0, Ffrm - K))
            if abs(sn - s) >= self.neg_margin or tries > 50:
                break
            tries += 1

        mag_a = mag[s:s+K]; re_a = re[s:s+K]; im_a = im[s:s+K]
        mag_p = mag[sp:sp+K]; re_p = re[sp:sp+K]; im_p = im[sp:sp+K]
        mag_n = mag[sn:sn+K]; re_n = re[sn:sn+K]; im_n = im[sn:sn+K]

        y_seg = _crop_or_pad_tc(y_tc, start=s * self.hop, length=K * self.hop)  # (K*hop, Cy)

        # 채널 수 불일치 방어 (feature channels vs waveform channels)
        if y_seg.shape[1] != C:
            raise ValueError(
                f"Wave channels ({y_seg.shape[1]}) != feature channels ({C}) for {wpath}. "
                f"Check wave_data_gen input shape / channel_axis."
            )

        return (
            torch.from_numpy(mag_a), torch.from_numpy(re_a), torch.from_numpy(im_a),
            torch.from_numpy(mag_p), torch.from_numpy(re_p), torch.from_numpy(im_p),
            torch.from_numpy(mag_n), torch.from_numpy(re_n), torch.from_numpy(im_n),
            torch.from_numpy(y_seg),                              # (L,C)
            torch.tensor(mat_id, dtype=torch.long),
        )


class SpecZWaveWindowTripletError(Exception):
    pass


class SpecZWaveWindowDataset(Dataset):
    """
    For Stage2/Stage3 vocoder training with cached z:
      returns (mag,re,im,z,y_seg,mat_id)
    z expected shape: (N, til_dim_eff)
    """
    def __init__(self, wave_dir, feat_root, z_dir, hop=256, window_k=8, til_dim=128, seed=0, recursive=True):
        self.wave_dir = wave_dir
        self.feat_root = feat_root
        self.items = _load_triplet_paths(wave_dir, feat_root, recursive=recursive)

        self.wave_files = [it[0] for it in self.items]
        mats = [_material_from_folder(wave_dir, w) for (w, *_rest) in self.items]
        self.materials = sorted(list(set(mats)))
        self.material_to_id = {m: i for i, m in enumerate(self.materials)}

        self.z_dir = z_dir
        self.hop = int(hop)
        self.K = int(window_k)
        self.til_dim = int(til_dim)
        self.rng = np.random.default_rng(seed)

        # infer channels
        _, m0, _, _ = self.items[0]
        mag0 = _ensure_feat_ncfw(np.load(m0), "mag")
        self.num_channels = int(mag0.shape[1])
        self.spec_f = int(mag0.shape[2])
        self.spec_w = int(mag0.shape[3])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wpath, mpath, repath, impath = self.items[idx]
        zpath = _z_cache_path_from_wave(self.z_dir, self.wave_dir, wpath)
        if not os.path.exists(zpath):
            raise SpecZWaveWindowTripletError(f"Missing cached z: {zpath}")

        mat = _material_from_folder(self.wave_dir, wpath)
        mat_id = np.int64(self.material_to_id.get(mat, 0))

        y_tc = _to_time_channel(np.load(wpath).astype(np.float32))  # (T,C)
        mag  = _ensure_feat_ncfw(np.load(mpath), "mag")             # (N,C,F,W)
        re   = _ensure_feat_ncfw(np.load(repath), "re")
        im   = _ensure_feat_ncfw(np.load(impath), "im")
        z    = np.load(zpath).astype(np.float32)                    # (N,til_dim_eff)

        Ffrm = mag.shape[0]
        K = self.K

        if z.ndim != 2 or z.shape[0] != Ffrm or z.shape[1] != self.til_dim:
            raise SpecZWaveWindowTripletError(f"bad z shape {z.shape}, expected ({Ffrm},{self.til_dim}) in {zpath}")

        if Ffrm < K:
            pad = K - Ffrm
            mag = np.concatenate([mag, np.repeat(mag[-1:], pad, axis=0)], axis=0)
            re  = np.concatenate([re,  np.repeat(re[-1:],  pad, axis=0)], axis=0)
            im  = np.concatenate([im,  np.repeat(im[-1:],  pad, axis=0)], axis=0)
            z   = np.concatenate([z,   np.repeat(z[-1:],   pad, axis=0)], axis=0)
            Ffrm = mag.shape[0]

        s = int(self.rng.integers(0, max(1, Ffrm - K)))

        mag_a = mag[s:s+K]
        re_a  = re[s:s+K]
        im_a  = im[s:s+K]
        z_a   = z[s:s+K]

        y_seg = _crop_or_pad_tc(y_tc, start=s * self.hop, length=K * self.hop)  # (L,C)

        if y_seg.shape[1] != mag.shape[1]:
            raise SpecZWaveWindowTripletError(
                f"Wave channels {y_seg.shape[1]} != feature channels {mag.shape[1]} for {wpath}"
            )

        return (
            torch.from_numpy(mag_a), torch.from_numpy(re_a), torch.from_numpy(im_a),
            torch.from_numpy(z_a),
            torch.from_numpy(y_seg),                      # (L,C)
            torch.tensor(mat_id, dtype=torch.long),
        )


def _blur_nd_spec(x):
    """
    x: (B,K,C,F,W) or (B,K,F,W)
    blur only on (F,W), preserve K/C
    """
    added_c = False
    if x.dim() == 4:
        x = x.unsqueeze(2)   # (B,K,1,F,W)
        added_c = True
    if x.dim() != 5:
        raise ValueError(f"_blur_nd_spec expects 4D or 5D, got {x.shape}")

    B, K, C, Fh, Ww = x.shape
    x = x.reshape(B * K * C, 1, Fh, Ww)
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    x = x.reshape(B, K, C, Fh, Ww)

    if added_c:
        x = x.squeeze(2)
    return x


def spec_corrupt(mag, re, im, p_drop=0.15, noise_std=0.03, ri_noise_std=0.02, blur_p=0.1):
    """
    mag/re/im:
      - old: (B,K,F,W)
      - new: (B,K,C,F,W)
    """
    if noise_std > 0:
        mag = mag + noise_std * torch.randn_like(mag)

    if ri_noise_std > 0:
        re = re + ri_noise_std * torch.randn_like(re)
        im = im + ri_noise_std * torch.randn_like(im)

    # asinh-companded mag can be negative (asinh of very small positive is >=0, but noisy mag may go below)
    mag = mag.clamp(min=0.0)

    # unify to 5D for dropout masking
    added_c = False
    if mag.dim() == 4:
        mag = mag.unsqueeze(2)
        re  = re.unsqueeze(2)
        im  = im.unsqueeze(2)
        added_c = True

    if mag.dim() != 5:
        raise ValueError(f"spec_corrupt expects 4D/5D tensors, got {mag.shape}")

    B, K, C, S, W = mag.shape

    if p_drop > 0:
        if torch.rand(1, device=mag.device).item() < p_drop:
            w0 = int(torch.randint(0, max(1, W - 16), (1,), device=mag.device).item())
            ww = int(torch.randint(8, 32, (1,), device=mag.device).item())
            mag[..., w0:w0+ww] = 0
            re[...,  w0:w0+ww] = 0
            im[...,  w0:w0+ww] = 0

        if torch.rand(1, device=mag.device).item() < p_drop:
            s0 = int(torch.randint(0, max(1, S - 8), (1,), device=mag.device).item())
            ss = int(torch.randint(4, 16, (1,), device=mag.device).item())
            mag[..., s0:s0+ss, :] = 0
            re[...,  s0:s0+ss, :] = 0
            im[...,  s0:s0+ss, :] = 0

    if blur_p > 0 and torch.rand(1, device=mag.device).item() < blur_p:
        mag = _blur_nd_spec(mag)
        re  = _blur_nd_spec(re)
        im  = _blur_nd_spec(im)

    if added_c:
        mag = mag.squeeze(2)
        re  = re.squeeze(2)
        im  = im.squeeze(2)

    return mag, re, im