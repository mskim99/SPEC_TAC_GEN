import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic efficient blocks
# -----------------------------
class DWConvBlock(nn.Module):
    """Depthwise separable ConvBlock: lighter than 2 full convs"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class ConvDownLite(nn.Module):
    """Single stride-2 ConvDown (anti-alias learnable)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DWConvBlock(in_ch, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UpConvLite(nn.Module):
    """Upsample + depthwise conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DWConvBlock(in_ch, out_ch)
    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)

# -----------------------------
# LiteCBAM (memory-friendly)
# -----------------------------
class LiteChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        hidden = max(1, in_ch // reduction)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1, bias=False)
    def forward(self, x):
        w = torch.mean(x, dim=(2,3), keepdim=True)
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(w))))
        return x * w

class LiteSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    def forward(self, x):
        attn = torch.sigmoid(self.conv(torch.mean(x, dim=1, keepdim=True)))
        return x * attn

class LiteCBAM(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.ca = LiteChannelAttention(in_ch, reduction)
        self.sa = LiteSpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

# -----------------------------
# Sinusoidal timestep embedding
# -----------------------------
def sinusoidal_embedding(t, dim=128):
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-torch.log(torch.tensor(10000.0)) / (half - 1)))
    emb = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# -----------------------------
# Lightweight UNet with LiteCBAM
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_ch=64, ch_mult=(1,2,4,8),
                 conditional=False, num_classes=0, emb_dim=128):
        super().__init__()
        self.conditional = conditional
        self.depth = len(ch_mult) - 1
        self.ch_mult = ch_mult

        # Encoder
        self.enc_blocks = nn.ModuleList()
        prev_ch = in_ch
        for i, m in enumerate(ch_mult):
            out_ch_enc = base_ch * m
            if i == 0:
                self.enc_blocks.append(DWConvBlock(prev_ch, out_ch_enc))
            else:
                self.enc_blocks.append(ConvDownLite(prev_ch, out_ch_enc))
            prev_ch = out_ch_enc

        # Bottleneck
        bottleneck_ch = base_ch * ch_mult[-1]
        self.bottleneck = nn.Sequential(
            DWConvBlock(bottleneck_ch, bottleneck_ch),
            LiteCBAM(bottleneck_ch)
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(self.depth)):
            in_c = base_ch * ch_mult[i+1]
            out_c = base_ch * ch_mult[i]
            self.upconvs.append(UpConvLite(in_c, out_c))
            self.dec_blocks.append(DWConvBlock(out_c + out_c, out_c))

        self.out_conv = nn.Conv2d(base_ch * ch_mult[0], out_ch, 1)

        if self.conditional:
            self.cond_emb = nn.Embedding(num_classes, emb_dim)
            self.t_mlp = nn.Linear(emb_dim, bottleneck_ch)

    def forward(self, x, t=None, cond=None):
        input_spatial = x.shape[2:]
        feats = []

        h = x
        for i, enc in enumerate(self.enc_blocks):
            h = enc(h)
            # detach early encoder skip to save gradient graph
            feats.append(h.detach() if i == 0 else h)

        # bottleneck
        h = self.bottleneck(h)

        # optional conditional embedding
        if self.conditional and t is not None and cond is not None:
            t_emb = sinusoidal_embedding(t, dim=self.cond_emb.embedding_dim)
            c_emb = self.cond_emb(cond)
            h = h + self.t_mlp(t_emb + c_emb)[:, :, None, None]

        # decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.dec_blocks)):
            skip = feats[-(i+2)]
            h = up(h, skip.shape[2:])
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        if h.shape[2:] != input_spatial:
            h = F.interpolate(h, size=input_spatial, mode="bilinear", align_corners=False)
        return self.out_conv(h)