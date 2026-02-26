# flow_its_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_btcfw(x):
    # x: (B,T,F,W) or (B,T,C,F,W)
    if x.ndim == 4:
        return x.unsqueeze(2)  # (B,T,1,F,W)
    elif x.ndim == 5:
        return x
    else:
        raise ValueError(f"Expected (B,T,F,W) or (B,T,C,F,W), got {x.shape}")

class DynamicAvgPool2d(nn.Module):
    def __init__(self, kh: int, kw: int):
        super().__init__()
        self.kh = int(kh)
        self.kw = int(kw)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        kh = min(self.kh, H)
        kw = min(self.kw, W)
        sh = kh
        sw = kw
        return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw))


class FrameEncoder2D(nn.Module):
    def __init__(self, in_ch=3, h_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            DynamicAvgPool2d(2, 4),  # safe

            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            DynamicAvgPool2d(2, 4),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            DynamicAvgPool2d(2, 4),

            nn.Conv2d(128, h_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.net(x)
        return self.pool(h).squeeze(-1).squeeze(-1)


class GMessage(nn.Module):
    def __init__(self, h_dim, m_dim=None, hidden=256):
        super().__init__()
        self.h_dim = h_dim
        self.m_dim = m_dim if m_dim is not None else h_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * h_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.m_dim),
        )

    def forward(self, h_prev, h_curr):
        return self.mlp(torch.cat([h_prev, h_curr], dim=-1))


class PhiUpdate(nn.Module):
    """Δh_t = Φ(h_{t-1}, m_{t-1->t})"""
    def __init__(self, h_dim, m_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + m_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, h_dim),
        )

    def forward(self, h_prev, m):
        return self.net(torch.cat([h_prev, m], dim=-1))


class FLOWITSBlock(nn.Module):
    """
    FLOW-ITS block:
      m_{t-1->t} = G(h_{t-1}, h_t)
      Δh_t       = Φ(h_{t-1}, m_{t-1->t})
      h_t <- h_t + alpha * Δh_t     (t>=1)
    """
    def __init__(self, h_dim, m_dim=None, pair_hidden=256, phi_hidden=256, alpha=1.0):
        super().__init__()
        self.h_dim = h_dim
        self.m_dim = m_dim if m_dim is not None else h_dim
        self.alpha = float(alpha)

        self.ln = nn.LayerNorm(h_dim)
        self.G = GMessage(h_dim=h_dim, m_dim=self.m_dim, hidden=pair_hidden)
        self.Phi = PhiUpdate(h_dim=h_dim, m_dim=self.m_dim, hidden=phi_hidden)

    def edge_messages(self, h):
        """
        h: (B,T,h_dim)
        returns:
          m_fwd: (B,T-1,m_dim) messages for t->t+1
          m_bwd: (B,T-1,m_dim) messages for (t+1)->t
        """
        B, T, _ = h.shape
        if T <= 1:
            m0 = torch.zeros((B, 0, self.m_dim), device=h.device, dtype=h.dtype)
            return m0, m0

        hn = self.ln(h)
        h_prev = hn[:, :-1, :]
        h_curr = hn[:,  1:, :]
        m_fwd = self.G(h_prev, h_curr)
        m_bwd = self.G(h_curr, h_prev)
        return m_fwd, m_bwd

    def forward(self, h):
        B, T, _ = h.shape
        if T <= 1:
            return h

        m_fwd, _ = self.edge_messages(h)     # (B,T-1,m)
        hn = self.ln(h)
        h_prev = hn[:, :-1, :]               # (B,T-1,h)
        dh = self.Phi(h_prev, m_fwd)         # (B,T-1,h)

        h_out = h.clone()
        h_out[:, 1:, :] = h_out[:, 1:, :] + self.alpha * dh
        return h_out


class FLOWITSModule(nn.Module):
    """
    FLOW-ITS module:
      - encodes each frame (mag,re,im) -> hE
      - optional time embedding with uniform tpos in [0,1]
      - applies FLOWITSBlocks
      - outputs per-frame latent z_t (flowits_dim) using [h_t, m_{t->t+1}] fusion
    """
    def __init__(
        self,
        in_ch=3,
        h_dim=192,
        m_dim=None,
        flowits_dim=128,
        n_layers=8,
        pair_hidden=256,
        phi_hidden=256,
        alpha=1.0,
        use_time_embed=True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.h_dim = int(h_dim)
        self.m_dim = int(m_dim) if m_dim is not None else int(h_dim)
        self.flowits_dim = int(flowits_dim)
        self.n_layers = int(n_layers)

        self.Es = FrameEncoder2D(in_ch=in_ch, h_dim=self.h_dim)

        self.use_time_embed = bool(use_time_embed)
        if self.use_time_embed:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, self.h_dim),
                nn.SiLU(),
                nn.Linear(self.h_dim, self.h_dim),
            )
        else:
            self.time_mlp = None

        self.blocks = nn.ModuleList([
            FLOWITSBlock(
                h_dim=self.h_dim,
                m_dim=self.m_dim,
                pair_hidden=pair_hidden,
                phi_hidden=phi_hidden,
                alpha=alpha,
            )
            for _ in range(self.n_layers)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(self.h_dim + self.m_dim),
            nn.Linear(self.h_dim + self.m_dim, self.flowits_dim),
        )

    def edge_messages(self, h, block_idx=-1):
        blk = self.blocks[block_idx]
        return blk.edge_messages(h)

    def forward(self, mag, re, im, return_aux=False):
        """
        mag/re/im: (B,T,64,512)
        returns:
          z: (B,T,flowits_dim)
          aux(optional):
            {
              "hE": (B,T,h_dim),   # encoder output
              "h":  (B,T,h_dim),   # final hidden after blocks
              "m_fwd": (B,T,m_dim),# padded edge messages (last is 0)
              "tpos": (1,T,1)      # uniform positions
            }
        """
        B, T, S, W = mag.shape
        x = torch.stack([mag, re, im], dim=2).reshape(B * T, 3, S, W)

        hE = self.Es(x).view(B, T, self.h_dim)  # (B,T,h_dim)

        if self.time_mlp is not None and T > 0:
            tpos = torch.linspace(0.0, 1.0, T, device=hE.device, dtype=hE.dtype)[None, :, None]  # (1,T,1)
            h = hE + self.time_mlp(tpos)
        else:
            tpos = torch.linspace(0.0, 1.0, T, device=hE.device, dtype=hE.dtype)[None, :, None]
            h = hE

        for blk in self.blocks:
            h = blk(h)

        # last-block edge messages, padded to length T
        if T <= 1:
            m_fwd_pad = torch.zeros((B, T, self.m_dim), device=h.device, dtype=h.dtype)
        else:
            m_fwd, _ = self.edge_messages(h, block_idx=-1)  # (B,T-1,m)
            m_fwd_pad = torch.cat(
                [m_fwd, torch.zeros((B, 1, self.m_dim), device=h.device, dtype=h.dtype)],
                dim=1
            )  # (B,T,m)

        z = self.out(torch.cat([h, m_fwd_pad], dim=-1))  # (B,T,flowits_dim)

        if return_aux:
            return z, {"hE": hE, "h": h, "m_fwd": m_fwd_pad, "tpos": tpos}
        return z
