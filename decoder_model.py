# decoder_model.py (MULTI-CHANNEL OUTPUT)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=p),
            nn.SiLU(),
            nn.Conv1d(ch, ch, k, padding=p),
        )

    def forward(self, x):
        return F.silu(x + self.net(x))


class ZOnlyDecoder1D(nn.Module):
    """
    Input:
      z_seq: (B,T,D)
      mat_id: (B,) optional
    Output:
      if out_channels == 1 -> (B, L)
      else                 -> (B, L, C)
    """
    def __init__(
        self,
        hop_len: int,
        win_len: int,
        flowits_dim: int,
        up_ch: int = 128,
        up_factors=(4, 4),
        num_materials: int = 1,
        cond_dim: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)
        self.flowits_dim = int(flowits_dim)
        self.up_ch = int(up_ch)
        self.up_factors = tuple(int(x) for x in up_factors)
        self.num_materials = int(max(1, num_materials))
        self.cond_dim = int(cond_dim)
        self.out_channels = int(max(1, out_channels))

        self.mat_emb = nn.Embedding(self.num_materials, self.cond_dim) if self.cond_dim > 0 else None

        in_dim = self.flowits_dim + (self.cond_dim if self.cond_dim > 0 else 0)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, self.up_ch),
            nn.SiLU(),
            nn.Linear(self.up_ch, self.up_ch),
        )

        blocks = []
        ch = self.up_ch
        for uf in self.up_factors:
            blocks += [
                ResBlock1D(ch),
                nn.Upsample(scale_factor=uf, mode="nearest"),
                nn.Conv1d(ch, ch, kernel_size=5, padding=2),
                nn.SiLU(),
                ResBlock1D(ch),
            ]
        self.net = nn.Sequential(*blocks)

        self.out_head = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(ch, self.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_seq: torch.Tensor, mat_id: torch.Tensor | None = None):
        if z_seq.dim() != 3:
            raise ValueError(f"z_seq must be (B,T,D), got {z_seq.shape}")

        B, T, D = z_seq.shape
        if D != self.flowits_dim:
            raise ValueError(f"z_seq dim mismatch: got {D}, expected {self.flowits_dim}")

        if self.cond_dim > 0:
            if mat_id is None:
                mat_id = torch.zeros((B,), device=z_seq.device, dtype=torch.long)
            m = self.mat_emb(mat_id)                 # (B,cond)
            m = m[:, None, :].expand(B, T, m.size(-1))
            x = torch.cat([z_seq, m], dim=-1)        # (B,T,D+cond)
        else:
            x = z_seq

        x = self.in_proj(x)                          # (B,T,up_ch)
        x = x.transpose(1, 2).contiguous()           # (B,up_ch,T)

        x = self.net(x)
        y = self.out_head(x)                         # (B,C_out,L_raw)

        # 정확히 T * hop_len로 맞추기
        target_len = int(T * self.hop_len)
        if y.size(-1) != target_len:
            y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)

        # output layout: (B,L,C)
        y = y.transpose(1, 2).contiguous()           # (B,L,C)

        return y