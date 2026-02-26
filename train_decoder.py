# train_decoder.py  (Strategy A FINAL + Stage2 latent augmentation + anti-averaging + ablation-ready)
import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flow_its_model import FLOWITSModule
from flow_matching import ConditionalFlowMatcher
from decoder_model import ZOnlyDecoder1D

from losses import mrstft_loss, envelope_loss
from data_decoder import (
    SpecWaveWindowTripletDataset,
    SpecZWaveWindowDataset,
    spec_corrupt
)


def encode_flowits_multichannel(flowits, mag, re, im, return_aux=False):
    """
    mag/re/im: (B,T,C,F,W) or old (B,T,F,W)
    flowits expects: (B,T,F,W)
    channel별로 인코딩 후 반환:
      z_ch: (B,T,C,D)
      aux_ref: flowits aux (single pass layout 기준)
    """
    if mag.dim() == 4:
        # old single-channel path -> (B,T,1,F,W)
        mag = mag.unsqueeze(2)
        re  = re.unsqueeze(2)
        im  = im.unsqueeze(2)

    if mag.dim() != 5:
        raise ValueError(f"Expected mag dim 5 (B,T,C,F,W), got {mag.shape}")

    B, T, C, Fh, Ww = mag.shape

    mag_f = mag.permute(0, 2, 1, 3, 4).reshape(B * C, T, Fh, Ww).contiguous()
    re_f  = re.permute(0, 2, 1, 3, 4).reshape(B * C, T, Fh, Ww).contiguous()
    im_f  = im.permute(0, 2, 1, 3, 4).reshape(B * C, T, Fh, Ww).contiguous()

    if return_aux:
        z_bc, aux = flowits(mag_f, re_f, im_f, return_aux=True)  # (B*C,T,D)
    else:
        z_bc = flowits(mag_f, re_f, im_f)                         # (B*C,T,D)
        aux = None

    D = z_bc.size(-1)
    z_ch = z_bc.reshape(B, C, T, D).permute(0, 2, 1, 3).contiguous()  # (B,T,C,D)
    return z_ch, aux


def fuse_channel_latent(z_ch, mode="concat"):
    """
    z_ch: (B,T,C,D)
    return z_seq_fused: (B,T,D_eff)
    """
    if z_ch.dim() != 4:
        raise ValueError(f"z_ch must be (B,T,C,D), got {z_ch.shape}")
    B, T, C, D = z_ch.shape

    if mode == "concat":
        return z_ch.reshape(B, T, C * D)
    elif mode == "mean":
        return z_ch.mean(dim=2)
    elif mode == "sum":
        return z_ch.sum(dim=2)
    else:
        raise ValueError(f"Unknown channel fusion mode: {mode}")


# --- irregularity / container distill blocks ---
class ContextAwareAdaptiveMask(nn.Module):
    def __init__(self, f_bins=64, h_dim=160, hidden=128, fusion="concat"):
        super().__init__()
        self.f_bins = int(f_bins)
        self.h_dim = int(h_dim)
        self.hidden = int(hidden)
        assert fusion in ["concat", "add"]
        self.fusion = fusion

        if self.fusion == "concat":
            in_dim = self.f_bins + self.h_dim
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, self.hidden),
                nn.SiLU(),
                nn.Linear(self.hidden, self.f_bins),
            )
            self.h_to_f = None
        else:
            self.h_to_f = nn.Sequential(
                nn.Linear(self.h_dim, self.hidden),
                nn.SiLU(),
                nn.Linear(self.hidden, self.f_bins),
            )
            self.mlp = nn.Sequential(
                nn.Linear(self.f_bins, self.hidden),
                nn.SiLU(),
                nn.Linear(self.hidden, self.f_bins),
            )

    def forward(self, S, h):
        # S: (B,T,F,W), h: (B,H) or (B,T,H)
        B, T, F, W = S.shape
        assert F == self.f_bins, f"Expected F={self.f_bins}, got {F}"

        prof = S.mean(dim=-1)  # (B,T,F)

        if h.dim() == 2:
            h_exp = h[:, None, :].expand(B, T, h.size(-1))
        elif h.dim() == 3:
            h_exp = h
        else:
            raise ValueError(f"h.dim() must be 2 or 3, got {h.dim()}")

        if self.fusion == "concat":
            x = torch.cat([prof, h_exp], dim=-1)
        else:
            x = prof + self.h_to_f(h_exp)

        logits = self.mlp(x)
        M = torch.sigmoid(logits)          # (B,T,F)
        S_high = S * M.unsqueeze(-1)       # (B,T,F,W)
        return M, S_high


class ContextAwareAdaptiveSpectralIrregularity(nn.Module):
    def __init__(self, f_bins=64, h_dim=160, hidden=128, fusion="concat", eps=1e-6):
        super().__init__()
        self.masker = ContextAwareAdaptiveMask(f_bins=f_bins, h_dim=h_dim, hidden=hidden, fusion=fusion)
        self.eps = float(eps)

    def forward(self, S, h):
        M, S_high = self.masker(S, h)
        high_E = (S_high.float() * S_high.float()).sum(dim=(2, 3))  # (B,T)
        A = 1.0 + torch.log1p(S.abs().float())
        log_E = (A * A).sum(dim=(2, 3))                              # (B,T)
        I = high_E / (log_E + self.eps)                              # (B,T)
        return I.to(dtype=S.dtype), M, S_high


class ContainerFrameEncoder(nn.Module):
    """E(S_high): mag-only frame -> c_dim"""
    def __init__(self, in_ch=1, c_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d((2, 4)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d((2, 4)),

            nn.Conv2d(64, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(hidden, c_dim)

    def forward(self, x):
        # x: (B,1,F,W)
        h = self.net(x)
        h = self.pool(h).squeeze(-1).squeeze(-1)
        return self.out(h)


class ContainerPredictor(nn.Module):
    """c_hat = P(z_t, tpos, mat_emb)"""
    def __init__(self, z_dim, g_dim, c_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1 + g_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, c_dim),
        )

    def forward(self, z, tpos, g):
        # z:(N,D), tpos:(N,1), g:(N,G)
        x = torch.cat([z, tpos, g], dim=-1)
        return self.net(x)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def log_stft_loss(y_hat, y, fft_sizes):
    """
    Compute log-STFT loss.
    """
    stft_hat = torch.stft(y_hat, n_fft=fft_sizes)
    stft_y = torch.stft(y, n_fft=fft_sizes)
    return torch.mean(torch.abs(torch.log(stft_hat + 1e-6) - torch.log(stft_y + 1e-6)))

def wav_losses(y_hat, y, fft_sizes, env_kernel=129, l_wav=0.5, l_stft=1.0, l_env=0.5):
    """
    y_hat / y:
      - single: (B,L)
      - multi : (B,L,C)
    mrstft_loss/envelope_loss가 (B,L) 기준이라 채널축을 batch로 펼침
    """
    if y_hat.dim() == 3 and y.dim() == 3:
        # (B,L,C) -> (B*C,L)
        B, L, C = y_hat.shape
        y_hat_f = y_hat.permute(0, 2, 1).reshape(B * C, L)
        y_f     = y.permute(0, 2, 1).reshape(B * C, L)
    elif y_hat.dim() == 2 and y.dim() == 2:
        y_hat_f = y_hat
        y_f = y
    else:
        raise ValueError(f"y_hat/y shape mismatch: {y_hat.shape}, {y.shape}")

    loss_wav  = torch.mean(torch.abs(y_hat_f - y_f))
    loss_stft = mrstft_loss(y_hat_f, y_f, fft_sizes=fft_sizes, hop_ratio=0.25)
    loss_env  = envelope_loss(y_hat_f, y_f, kernel=env_kernel)

    loss = l_wav * loss_wav + l_stft * loss_stft + l_env * loss_env
    return loss, loss_wav, loss_stft, loss_env


@torch.no_grad()
def cache_latents_fullseq(flowits, wave_files, wave_dir, feat_root, out_dir, device,
                          batch_frames=32, channel_fusion="concat"):
    os.makedirs(out_dir, exist_ok=True)
    flowits.eval()

    mag_dir = os.path.join(feat_root, "mag")
    re_dir  = os.path.join(feat_root, "re")
    im_dir  = os.path.join(feat_root, "im")

    for wpath in tqdm(wave_files, desc="Caching z (full seq)", dynamic_ncols=True):
        rel = os.path.relpath(wpath, wave_dir)
        rel_dir = os.path.dirname(rel)
        base = os.path.splitext(os.path.basename(wpath))[0]

        z_out_dir = os.path.join(out_dir, rel_dir)
        os.makedirs(z_out_dir, exist_ok=True)
        zpath = os.path.join(z_out_dir, f"{base}_z.npy")
        if os.path.exists(zpath):
            continue

        mpath  = os.path.join(mag_dir, rel_dir, f"{base}_mag.npy")
        repath = os.path.join(re_dir,  rel_dir, f"{base}_re.npy")
        impath = os.path.join(im_dir,  rel_dir, f"{base}_im.npy")
        if not (os.path.exists(mpath) and os.path.exists(repath) and os.path.exists(impath)):
            continue

        mag = np.load(mpath).astype(np.float32)
        re  = np.load(repath).astype(np.float32)
        im  = np.load(impath).astype(np.float32)

        # normalize to (Ffrm,C,F,W)
        if mag.ndim == 3:
            mag = mag[:, None, :, :]
            re  = re[:, None, :, :]
            im  = im[:, None, :, :]

        Ffrm = mag.shape[0]
        z_list = []
        s = 0
        while s < Ffrm:
            e = min(Ffrm, s + batch_frames)

            # (1, Tchunk, C, F, W)
            mag_t = torch.from_numpy(mag[s:e][None]).to(device, non_blocking=True)
            re_t  = torch.from_numpy(re[s:e][None]).to(device, non_blocking=True)
            im_t  = torch.from_numpy(im[s:e][None]).to(device, non_blocking=True)

            z_ch, _ = encode_flowits_multichannel(flowits, mag_t, re_t, im_t, return_aux=False)  # (1,T,C,D)
            z_fused = fuse_channel_latent(z_ch, mode=channel_fusion).squeeze(0).float().cpu().numpy()  # (T,D_eff)
            z_list.append(z_fused)
            s = e

        z_full = np.concatenate(z_list, axis=0)
        np.save(zpath, z_full)


def save_ckpt(path, flowits, prior_flow, trans_flow, z_decoder, heads, opt, scaler, step, ep, stage, cfg):
    ckpt = {
        "flow_its": flowits.state_dict(),
        "prior_flow": prior_flow.state_dict(),
        "trans_flow": trans_flow.state_dict(),
        "z_decoder": z_decoder.state_dict(),
        "heads": {k: v.state_dict() for k, v in heads.items()},
        "opt": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": ep,
        "stage": stage,
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available() and (not args.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.z_cache_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "tb"))

    # -----------------------------
    # Dataset
    # -----------------------------
    ds_trip = SpecWaveWindowTripletDataset(
        wave_dir=args.wave_dir,
        feat_root=args.feat_root,
        hop=args.hop_len,
        window_k=args.window_k,
        neg_margin=args.neg_margin,
        seed=args.seed,
        recursive=True,
    )
    dl = DataLoader(
        ds_trip,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    args.materials = ds_trip.materials
    args.num_materials = len(ds_trip.materials)

    args.num_channels = int(getattr(ds_trip, "num_channels", 1))
    args.spec_f = int(getattr(ds_trip, "spec_f", args.spec_f))
    args.spec_w = int(getattr(ds_trip, "spec_w", args.spec_w))

    # concat이면 flow/decoder가 보는 latent 차원이 커짐
    args.channel_fusion = getattr(args, "channel_fusion", "concat")
    if args.channel_fusion == "concat":
        eff_z_dim = args.flowits_dim * args.num_channels
    elif args.channel_fusion in ["mean", "sum"]:
        eff_z_dim = args.flowits_dim
    else:
        raise ValueError(f"Unknown channel_fusion={args.channel_fusion}")
    args.eff_z_dim = eff_z_dim

    # -----------------------------
    # Models
    # -----------------------------
    flowits = FLOWITSModule(
        in_ch=3,
        h_dim=args.flowits_h_dim,
        m_dim=args.flowits_m_dim if args.flowits_m_dim > 0 else None,
        flowits_dim=args.flowits_dim,
        n_layers=args.flowits_layers,
        pair_hidden=args.flowits_pair_hidden,
        phi_hidden=args.flowits_phi_hidden,
        alpha=args.flowits_alpha,
        use_time_embed=True,
    ).to(device)

    # material embedding used as "global condition g" for flows
    g_dim = args.flow_mat_cond_dim if (args.flow_mat_cond_dim > 0 and args.num_materials > 0) else 0
    heads = {}
    if g_dim > 0:
        heads["mat_flow"] = nn.Embedding(args.num_materials, g_dim).to(device)

    # irregularity -> container target
    hi_dim = args.eff_z_dim + g_dim
    heads["irr"] = ContextAwareAdaptiveSpectralIrregularity(
        f_bins=args.spec_f, h_dim=hi_dim, hidden=args.irr_hidden, fusion=args.irr_fusion, eps=args.irr_eps
    ).to(device)

    heads["cont_enc"] = ContainerFrameEncoder(in_ch=1, c_dim=args.container_dim, hidden=args.container_hidden).to(device)

    heads["cont_pred"] = ContainerPredictor(
        z_dim=args.eff_z_dim, g_dim=g_dim, c_dim=args.container_dim, hidden=args.container_pred_hidden
    ).to(device)

    # prior flow (typically no cycle term)
    prior_flow = ConditionalFlowMatcher(
        x_dim=args.eff_z_dim,
        g_dim=g_dim,
        l_dim=0,
        hidden=args.flow_hidden,
        time_dim=args.flow_time_dim,
        sigma=args.flow_sigma,
        q_sigma=args.flow_q_sigma,
        q_temp=args.flow_q_temp,
        pool_chunk=args.flow_pool_chunk,
        ot_eps=args.flow_ot_eps,
        ot_iters=args.flow_ot_iters,
        ot_max_n=args.flow_ot_max_n,
        bidirectional=False,
        lambda_cyc=0.0,
        cycle_steps=args.flow_cycle_steps,
        cycle_max_n=args.flow_cycle_max_n,
    ).to(device)

    # transition flow (ablation target: bidirectional / lambda_cyc)
    trans_flow = ConditionalFlowMatcher(
        x_dim=args.eff_z_dim,
        g_dim=g_dim,
        l_dim=args.container_dim,
        hidden=args.flow_hidden,
        time_dim=args.flow_time_dim,
        sigma=args.flow_sigma,
        q_sigma=args.flow_q_sigma,
        q_temp=args.flow_q_temp,
        pool_chunk=args.flow_pool_chunk,
        ot_eps=args.flow_ot_eps,
        ot_iters=args.flow_ot_iters,
        ot_max_n=args.flow_ot_max_n,
        bidirectional=(not args.disable_bidirectional),
        lambda_cyc=args.lambda_cyc,
        cycle_steps=args.flow_cycle_steps,
        cycle_max_n=args.flow_cycle_max_n,
    ).to(device)

    z_decoder = ZOnlyDecoder1D(
        hop_len=args.hop_len,
        win_len=args.win_len,
        flowits_dim=args.eff_z_dim,
        up_ch=args.dec_up_ch,
        up_factors=tuple(args.dec_up_factors),
        num_materials=args.num_materials,
        cond_dim=args.cond_dim,
        out_channels=args.num_channels,
    ).to(device)

    # -----------------------------
    # helpers
    # -----------------------------
    def corrupt_all(mag, re, im):
        if not args.use_corrupt:
            return mag, re, im
        return spec_corrupt(
            mag, re, im,
            p_drop=args.p_drop,
            noise_std=args.mag_noise,
            ri_noise_std=args.ri_noise,
            blur_p=args.blur_p
        )

    def get_g(mat_id, dtype):
        if "mat_flow" not in heads:
            return None
        return heads["mat_flow"](mat_id).to(dtype=dtype)

    def build_hi(z_seq, g):
        # ship-like global context used only for irregularity mask conditioning
        hi = z_seq.mean(dim=1)
        if g is not None:
            hi = torch.cat([hi, g], dim=-1)
        return hi

    @torch.no_grad()
    def rollout_zseq_from_real_z0(z0, mat_id, T):
        """
        real z0를 시작점으로 sampled-style rollout을 만들어 decoder가 sampled-z 분포에도 robust해지게 함.
        Container ablation 시 trans_flow conditioning에도 동일하게 반영.
        """
        B, D = z0.shape
        g = get_g(mat_id, dtype=z0.dtype)

        z_list = [z0]
        zt = z0
        for i in range(T - 1):
            tpos = torch.full((B, 1), float(i) / max(T - 1, 1), device=z0.device, dtype=z0.dtype)

            if g is None:
                gf = torch.zeros((B, 0), device=z0.device, dtype=z0.dtype)
            else:
                gf = g

            if args.ablate_no_container_conditioning:
                c = torch.zeros((B, args.container_dim), device=z0.device, dtype=z0.dtype)
            else:
                c = heads["cont_pred"](zt, tpos, gf)  # (B,C)

            zt = trans_flow.step_euler(zt, g=g, l=c, n_steps=args.stage2_roll_steps)
            z_list.append(zt)

        return torch.stack(z_list, dim=1)  # (B,T,D)

    @torch.no_grad()
    def make_decoder_train_z(z_real, mat_id):
        """
        Stage2 decoder 입력 z를 real-z 기반으로 만들되,
        rollout mix + noise/scale jitter로 sampled-z OOD를 완화.
        """
        z_in = z_real
        B, T, D = z_real.shape

        # 1) rollout mix (real-z와 sampled-style z를 혼합)
        if args.stage2_roll_mix > 0.0 and T > 1:
            z_roll = rollout_zseq_from_real_z0(z_real[:, 0, :], mat_id, T)  # (B,T,D)
            alpha = torch.rand((B, 1, 1), device=z_real.device, dtype=z_real.dtype) * float(args.stage2_roll_mix)
            z_in = (1.0 - alpha) * z_in + alpha * z_roll

        # 2) latent scale jitter (sampled-z scale mismatch 완화)
        if args.stage2_z_scale_jitter > 0:
            s = 1.0 + float(args.stage2_z_scale_jitter) * torch.randn((B, 1, 1), device=z_real.device, dtype=z_real.dtype)
            s = s.clamp(0.7, 1.3)
            z_in = z_in * s

        # 3) small additive noise
        if args.stage2_z_noise_std > 0:
            z_in = z_in + float(args.stage2_z_noise_std) * torch.randn_like(z_in)

        return z_in

    # -----------------------------
    # Stage 1: Train FLOWITS + flows + container distill
    # -----------------------------
    print(f"\n=== Stage 1: FLOWITS + prior/trans flow + container distill ({args.stage1_epochs} epochs) ===")

    set_requires_grad(flowits, True)
    set_requires_grad(prior_flow, True)
    set_requires_grad(trans_flow, True)
    set_requires_grad(z_decoder, False)
    for h in heads.values():
        set_requires_grad(h, True)

    params1 = (
        list(flowits.parameters())
        + list(prior_flow.parameters())
        + list(trans_flow.parameters())
        + [p for h in heads.values() for p in h.parameters()]
    )
    opt1 = torch.optim.AdamW(params1, lr=args.lr_stage1, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scaler1 = torch.cuda.amp.GradScaler(enabled=args.amp and use_cuda)

    step = 0
    t0 = time.time()

    for ep in range(1, args.stage1_epochs + 1):
        flowits.train(); prior_flow.train(); trans_flow.train()
        for h in heads.values():
            h.train()

        pbar = tqdm(dl, desc=f"[S1] Ep {ep}/{args.stage1_epochs}", dynamic_ncols=True)
        for batch in pbar:
            step += 1
            (mag_a, re_a, im_a,
             _mag_p, _re_p, _im_p,
             _mag_n, _re_n, _im_n,
             y_a,
             mat_id) = batch

            mag_a = mag_a.to(device, non_blocking=True)
            re_a  = re_a.to(device, non_blocking=True)
            im_a  = im_a.to(device, non_blocking=True)
            y_a   = y_a.to(device, non_blocking=True)
            mat_id = mat_id.to(device, non_blocking=True)

            mag_raw = mag_a.detach()
            mag_a, re_a, im_a = corrupt_all(mag_a, re_a, im_a)

            with torch.autocast(device_type=("cuda" if use_cuda else "cpu"),
                                dtype=(torch.float16 if use_cuda else torch.bfloat16),
                                enabled=args.amp):

                # mag_a/re_a/im_a shape from dataset: (B,K,C,F,W)
                z_ch, aux = encode_flowits_multichannel(flowits, mag_a, re_a, im_a, return_aux=True)  # (B,T,C,D_base)
                z_seq = fuse_channel_latent(z_ch, mode=args.channel_fusion)  # (B,T,D_eff)
                B, T, D = z_seq.shape
                K = max(T - 1, 0)

                g = get_g(mat_id, dtype=z_seq.dtype)  # (B,G) or None

                # ---- container target from mag_raw (teacher) ----
                hi = build_hi(z_seq, g)  # (B, hi_dim)
                # irregularity teacher는 멀티채널 스펙을 평균해서 사용 (global teacher)
                if mag_raw.dim() == 5:
                    mag_for_irr = mag_raw.mean(dim=2)   # (B,T,F,W)
                else:
                    mag_for_irr = mag_raw
                I_frame, _M, S_high = heads["irr"](mag_for_irr, hi)  # I:(B,T), S_high:(B,T,F,W)

                Sh = S_high.reshape(B * T, 1, S_high.size(2), S_high.size(3))
                c_frame = heads["cont_enc"](Sh).reshape(B, T, -1)    # (B,T,C)

                # Ablation: irregularity weighting on/off
                if args.ablate_no_irr_weight:
                    c_tgt = c_frame.detach()
                else:
                    c_tgt = (I_frame.unsqueeze(-1) * c_frame).detach()

                l_tgt = c_tgt[:, :-1, :] if T > 1 else torch.zeros((B, 0, args.container_dim), device=device, dtype=z_seq.dtype)

                # ---- container predictor (student) ----
                tpos = aux["tpos"].to(dtype=z_seq.dtype)  # (1,T,1)
                t_edge = tpos[:, :-1, :].expand(B, K, 1) if T > 1 else torch.zeros((B, 0, 1), device=device, dtype=z_seq.dtype)
                z_edge = z_seq[:, :-1, :] if T > 1 else torch.zeros((B, 0, D), device=device, dtype=z_seq.dtype)

                if g is None:
                    g_edge = torch.zeros((B, K, 0), device=device, dtype=z_seq.dtype)
                else:
                    g_edge = g[:, None, :].expand(B, K, g.size(-1))

                zf = z_edge.reshape(B * K, D) if K > 0 else z_edge.reshape(0, D)
                tf = t_edge.reshape(B * K, 1) if K > 0 else t_edge.reshape(0, 1)
                gf = (
                    g_edge.reshape(B * K, g_edge.size(-1))
                    if (K > 0 and g is not None)
                    else torch.zeros((B * K, 0), device=device, dtype=z_seq.dtype)
                )

                c_pred = (
                    heads["cont_pred"](zf, tf, gf).reshape(B, K, args.container_dim)
                    if K > 0 else torch.zeros((B, 0, args.container_dim), device=device, dtype=z_seq.dtype)
                )

                # ---- choose container conditioning (ablation-aware) ----
                if args.ablate_no_container_conditioning:
                    l_mix_used = torch.zeros_like(l_tgt)
                else:
                    use_teacher = not args.ablate_no_container_teacher
                    use_pred = not args.ablate_no_container_pred

                    if use_teacher and use_pred:
                        l_mix_used = (1.0 - args.container_mix) * l_tgt + args.container_mix * c_pred
                    elif use_teacher:
                        l_mix_used = l_tgt
                    elif use_pred:
                        l_mix_used = c_pred
                    else:
                        # guarded by sanity check; fallback
                        l_mix_used = torch.zeros_like(l_tgt)

                # ---- transition flow loss (LG + lambda_cyc * Lcyc if enabled) ----
                loss_trans_flow, flow_parts = trans_flow.loss_expectation_with_cycle(
                    z_seq=z_seq, g=g, l=l_mix_used, return_parts=True
                )

                # ---- prior flow loss (noise -> z0) ----
                z0 = z_seq[:, 0, :]                        # (B,D)
                x0 = torch.randn_like(z0)                  # noise
                loss_prior = prior_flow.loss(x0=x0, x1=z0, g=g, l=None, use_ot=False)

                # ---- container distill supervised ----
                loss_cont = (
                    torch.mean((c_pred - l_tgt) ** 2)
                    if K > 0 else torch.zeros((), device=device, dtype=z_seq.dtype)
                )

                # total
                loss = (
                    args.l_trans_flow * loss_trans_flow +
                    args.l_prior      * loss_prior +
                    args.l_container  * loss_cont
                )

            opt1.zero_grad(set_to_none=True)
            scaler1.scale(loss).backward()
            scaler1.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(params1, args.grad_clip)
            scaler1.step(opt1)
            scaler1.update()

            if step % args.log_every == 0:
                elapsed_min = (time.time() - t0) / 60.0
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    tflow=f"{loss_trans_flow.item():.4f}",
                    lg=f"{flow_parts['loss_g'].item():.4f}",
                    lcyc=f"{flow_parts['loss_cyc'].item():.4f}",
                    prior=f"{loss_prior.item():.4f}",
                    cont=f"{loss_cont.item():.4f}",
                    t=f"{elapsed_min:.1f}m"
                )
                writer.add_scalar("S1/loss", float(loss.item()), step)
                writer.add_scalar("S1/loss_trans_flow", float(loss_trans_flow.item()), step)
                writer.add_scalar("S1/loss_trans_flow_g", float(flow_parts["loss_g"].item()), step)
                writer.add_scalar("S1/loss_trans_flow_cyc", float(flow_parts["loss_cyc"].item()), step)
                writer.add_scalar("S1/loss_prior", float(loss_prior.item()), step)
                writer.add_scalar("S1/loss_cont", float(loss_cont.item()), step)

        if ep % args.save_every == 0 or ep == args.stage1_epochs:
            save_ckpt(
                os.path.join(args.ckpt_dir, f"s1_ep{ep:04d}.pt"),
                flowits, prior_flow, trans_flow, z_decoder, heads, opt1, scaler1, step, ep, "stage1", vars(args)
            )

    # -----------------------------
    # Stage 2: Freeze FLOWITS + cache z + train z-only decoder
    # -----------------------------
    print(f"\n=== Stage 2: cache z + train ZOnlyDecoder ({args.stage2_epochs} epochs) ===")

    set_requires_grad(flowits, False); flowits.eval()
    set_requires_grad(prior_flow, False); prior_flow.eval()
    set_requires_grad(trans_flow, False); trans_flow.eval()
    for h in heads.values():
        set_requires_grad(h, False)
        h.eval()

    cache_latents_fullseq(
        flowits=flowits,
        wave_files=ds_trip.wave_files,
        wave_dir=args.wave_dir,
        feat_root=args.feat_root,
        out_dir=args.z_cache_dir,
        device=device,
        batch_frames=args.cache_batch_frames,
        channel_fusion=args.channel_fusion,
    )

    ds_z = SpecZWaveWindowDataset(
        wave_dir=args.wave_dir,
        feat_root=args.feat_root,
        z_dir=args.z_cache_dir,
        hop=args.hop_len,
        window_k=args.window_k,
        til_dim=args.eff_z_dim,
        seed=args.seed + 777,
        recursive=True,
    )
    dl_z = DataLoader(
        ds_z,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    set_requires_grad(z_decoder, True)
    opt2 = torch.optim.AdamW(z_decoder.parameters(), lr=args.lr_stage2, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scaler2 = torch.cuda.amp.GradScaler(enabled=args.amp and use_cuda)

    for ep in range(1, args.stage2_epochs + 1):
        z_decoder.train()
        pbar = tqdm(dl_z, desc=f"[S2] Ep {ep}/{args.stage2_epochs}", dynamic_ncols=True)

        for mag_a, re_a, im_a, z_a, y_a, mat_id in pbar:
            step += 1
            z_a   = z_a.to(device, non_blocking=True)
            y_a   = y_a.to(device, non_blocking=True)
            mat_id = mat_id.to(device, non_blocking=True)

            with torch.autocast(device_type=("cuda" if use_cuda else "cpu"),
                                dtype=(torch.float16 if use_cuda else torch.bfloat16),
                                enabled=args.amp):

                # real-z + sampled-style latent augmentation
                z_in = make_decoder_train_z(z_a, mat_id)

                # clean pass (in-dist)
                y_hat_clean = z_decoder(z_a, mat_id=mat_id)
                loss_clean, loss_wav_c, loss_stft_c, loss_env_c = wav_losses(
                    y_hat_clean, y_a,
                    fft_sizes=args.fft_sizes,
                    env_kernel=args.env_kernel,
                    l_wav=args.l_wav, l_stft=args.l_stft, l_env=args.l_env
                )

                # aug pass (sampled-z robustness)
                if args.l_rec_aug > 0:
                    y_hat_aug = z_decoder(z_in, mat_id=mat_id)
                    loss_aug, loss_wav_a, loss_stft_a, loss_env_a = wav_losses(
                        y_hat_aug, y_a,
                        fft_sizes=args.fft_sizes,
                        env_kernel=args.env_kernel,
                        l_wav=args.l_wav, l_stft=args.l_stft, l_env=args.l_env
                    )
                else:
                    loss_aug = torch.zeros((), device=device, dtype=loss_clean.dtype)
                    loss_wav_a = torch.zeros_like(loss_aug)
                    loss_stft_a = torch.zeros_like(loss_aug)
                    loss_env_a = torch.zeros_like(loss_aug)

                # derivative loss (anti-averaging)
                if args.l_diff > 0:
                    dy_hat = y_hat_clean[:, 1:] - y_hat_clean[:, :-1]
                    dy_gt  = y_a[:, 1:] - y_a[:, :-1]
                    loss_diff = torch.mean(torch.abs(dy_hat - dy_gt))
                else:
                    loss_diff = torch.zeros((), device=device, dtype=loss_clean.dtype)

                loss = loss_clean + args.l_rec_aug * loss_aug + args.l_diff * loss_diff

            opt2.zero_grad(set_to_none=True)
            scaler2.scale(loss).backward()
            scaler2.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(z_decoder.parameters(), args.grad_clip)
            scaler2.step(opt2)
            scaler2.update()

            if step % args.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    clean=f"{loss_clean.item():.4f}",
                    aug=f"{loss_aug.item():.4f}",
                    diff=f"{loss_diff.item():.4f}",
                )
                writer.add_scalar("S2/loss", float(loss.item()), step)
                writer.add_scalar("S2/loss_clean", float(loss_clean.item()), step)
                writer.add_scalar("S2/loss_aug", float(loss_aug.item()), step)
                writer.add_scalar("S2/loss_diff", float(loss_diff.item()), step)

                writer.add_scalar("S2/clean_wav", float(loss_wav_c.item()), step)
                writer.add_scalar("S2/clean_stft", float(loss_stft_c.item()), step)
                writer.add_scalar("S2/clean_env", float(loss_env_c.item()), step)

                if args.l_rec_aug > 0:
                    writer.add_scalar("S2/aug_wav", float(loss_wav_a.item()), step)
                    writer.add_scalar("S2/aug_stft", float(loss_stft_a.item()), step)
                    writer.add_scalar("S2/aug_env", float(loss_env_a.item()), step)

        if ep % args.save_every == 0 or ep == args.stage2_epochs:
            save_ckpt(
                os.path.join(args.ckpt_dir, f"s2_ep{ep:04d}.pt"),
                flowits, prior_flow, trans_flow, z_decoder, heads, opt2, scaler2, step, ep, "stage2", vars(args)
            )

    writer.close()
    print("✅ Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")

    # data
    p.add_argument("--wave_dir", type=str, required=True)
    p.add_argument("--feat_root", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, default="ckpts_strategyA")
    p.add_argument("--z_cache_dir", type=str, default="z_cache")

    # framing (128 total = 8 frames * 16 hop)
    p.add_argument("--hop_len", type=int, default=16)
    p.add_argument("--win_len", type=int, default=16)

    # window sampling
    p.add_argument("--window_k", type=int, default=8)
    p.add_argument("--neg_margin", type=int, default=16)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--stage1_epochs", type=int, default=50)
    p.add_argument("--stage2_epochs", type=int, default=200)
    p.add_argument("--lr_stage1", type=float, default=2e-4)
    p.add_argument("--lr_stage2", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # decoder (hop_len=16 -> product must be 16)
    p.add_argument("--dec_up_ch", type=int, default=128)
    p.add_argument("--dec_up_factors", type=int, nargs="+", default=[4, 4])

    # waveform losses
    p.add_argument("--l_wav", type=float, default=0.5)
    p.add_argument("--l_stft", type=float, default=1.0)
    p.add_argument("--l_env", type=float, default=0.5)
    p.add_argument("--env_kernel", type=int, default=129)
    p.add_argument("--fft_sizes", type=int, nargs="+", default=[16, 32, 64])

    # corruption (optional)
    p.add_argument("--use_corrupt", action="store_true")
    p.add_argument("--p_drop", type=float, default=0.15)
    p.add_argument("--mag_noise", type=float, default=0.03)
    p.add_argument("--ri_noise", type=float, default=0.02)
    p.add_argument("--blur_p", type=float, default=0.10)

    # FLOW-ITS
    p.add_argument("--flowits_dim", type=int, default=128)
    p.add_argument("--flowits_h_dim", type=int, default=192)
    p.add_argument("--flowits_m_dim", type=int, default=0)
    p.add_argument("--flowits_layers", type=int, default=8)
    p.add_argument("--flowits_pair_hidden", type=int, default=256)
    p.add_argument("--flowits_phi_hidden", type=int, default=256)
    p.add_argument("--flowits_alpha", type=float, default=1.0)

    # Flow Matching
    p.add_argument("--flow_sigma", type=float, default=0.01)
    p.add_argument("--flow_hidden", type=int, default=256)
    p.add_argument("--flow_time_dim", type=int, default=64)
    p.add_argument("--flow_ot_eps", type=float, default=0.05)
    p.add_argument("--flow_ot_iters", type=int, default=50)
    p.add_argument("--flow_ot_max_n", type=int, default=512)
    p.add_argument("--flow_q_sigma", type=float, default=0.10)
    p.add_argument("--flow_q_temp", type=float, default=1.0)
    p.add_argument("--flow_pool_chunk", type=int, default=2048)

    # --- NEW: bidirectional / cycle ablation controls ---
    p.add_argument("--disable_bidirectional", action="store_true",
                   help="Disable bidirectional cycle consistency in transition flow (use LG only).")
    p.add_argument("--lambda_cyc", type=float, default=0.6,
                   help="Cycle-consistency loss weight for transition flow.")
    p.add_argument("--flow_cycle_steps", type=int, default=16,
                   help="Euler steps used in cycle-consistency integration.")
    p.add_argument("--flow_cycle_max_n", type=int, default=4096,
                   help="Max number of edges used for cycle loss per batch (speed/memory control). 0 disables subsampling.")

    # conditions
    p.add_argument("--flow_mat_cond_dim", type=int, default=32)
    p.add_argument("--cond_dim", type=int, default=32)

    # irregularity / container
    p.add_argument("--spec_f", type=int, default=16)
    p.add_argument("--spec_w", type=int, default=16)
    p.add_argument("--irr_hidden", type=int, default=128)
    p.add_argument("--irr_eps", type=float, default=1e-6)
    p.add_argument("--irr_fusion", type=str, default="concat", choices=["concat", "add"])

    p.add_argument("--container_dim", type=int, default=64)
    p.add_argument("--container_hidden", type=int, default=128)
    p.add_argument("--container_pred_hidden", type=int, default=256)
    p.add_argument("--container_mix", type=float, default=0.5, help="0: teacher only, 1: pred only")

    # --- NEW: container ablation flags ---
    p.add_argument("--ablate_no_container_conditioning", action="store_true",
                   help="Disable container conditioning to trans_flow and Stage2 rollout.")
    p.add_argument("--ablate_no_irr_weight", action="store_true",
                   help="Disable irregularity weighting when forming container target c_tgt.")
    p.add_argument("--ablate_no_container_teacher", action="store_true",
                   help="Do not use teacher container target in trans_flow conditioning (pred-only if enabled).")
    p.add_argument("--ablate_no_container_pred", action="store_true",
                   help="Do not use predicted container in trans_flow conditioning (teacher-only if enabled).")

    # loss weights (Stage1)
    p.add_argument("--l_trans_flow", type=float, default=0.20)
    p.add_argument("--l_prior", type=float, default=0.20)
    p.add_argument("--l_container", type=float, default=0.20)

    # Stage2 decoder robustness / anti-averaging
    p.add_argument("--stage2_roll_steps", type=int, default=8,
                   help="trans_flow rollout Euler steps for latent augmentation")
    p.add_argument("--stage2_roll_mix", type=float, default=0.20,
                   help="max mix ratio with rollout z (0 disables)")
    p.add_argument("--stage2_z_noise_std", type=float, default=0.005,
                   help="small additive latent noise in stage2 decoder training")
    p.add_argument("--stage2_z_scale_jitter", type=float, default=0.10,
                   help="latent scale jitter std in stage2 decoder training")
    p.add_argument("--l_rec_aug", type=float, default=0.3,
                   help="weight for augmented-z reconstruction loss")
    p.add_argument("--l_diff", type=float, default=0.1,
                   help="waveform first-difference L1 loss weight (anti-averaging)")

    # caching
    p.add_argument("--cache_batch_frames", type=int, default=16)

    # log/save
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=10)

    # channel fusion for multichannel latent
    p.add_argument("--channel_fusion", type=str, default="concat", choices=["concat", "mean", "sum"],
                   help="FLOWITS channel latent fusion. concat is recommended for multichannel preservation.")

    args = p.parse_args()

    # sanity checks
    if args.ablate_no_container_teacher and args.ablate_no_container_pred and (not args.ablate_no_container_conditioning):
        raise ValueError("Invalid config: both teacher and pred container are disabled while container conditioning is enabled.")

    train(args)