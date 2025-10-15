# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# Building Blocks
# ===============================

class ConvBlock(nn.Module):
    """기본 CNN 블록 (Conv → BN → ReLU) x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def sinusoidal_embedding(t, dim=128):
    device = t.device
    half = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


# ===============================
# Flexible UNet
# ===============================

class UNet(nn.Module):
    """
    UNet(depth=N, ch_mult=[1,2,4,8]) 형태로 encoder/decoder 블록 수를 조절 가능
    """
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        base_ch=32,
        ch_mult=(1, 2, 4, 8),
        conditional=False,
        num_classes=0,
        emb_dim=128,
    ):
        super().__init__()
        self.conditional = conditional
        self.depth = len(ch_mult) - 1  # encoder 단계 수

        # ---------- Encoder ----------
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_ch
        for mult in ch_mult[:-1]:
            out_ch_enc = base_ch * mult
            self.enc_blocks.append(ConvBlock(prev_ch, out_ch_enc))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = out_ch_enc

        # ---------- Bottleneck ----------
        self.bottleneck = ConvBlock(base_ch * ch_mult[-2], base_ch * ch_mult[-1])

        # ---------- Decoder ----------
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(self.depth)):
            in_channels = base_ch * ch_mult[i + 1]
            out_channels = base_ch * ch_mult[i]
            self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, 2, 2))
            self.dec_blocks.append(ConvBlock(out_channels * 2, out_channels))

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(base_ch * ch_mult[0], out_ch, 1)

        # ---------- Conditional Embedding ----------
        if self.conditional:
            self.cond_emb = nn.Embedding(num_classes, emb_dim)
            self.fc = nn.Linear(emb_dim * 2, base_ch * ch_mult[-1])

    def forward(self, x, t=None, cond=None):
        # Encoder
        enc_feats = []
        out = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            out = enc(out)
            enc_feats.append(out)
            out = pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Conditional Embedding (if applicable)
        if self.conditional and t is not None and cond is not None:
            t_emb = sinusoidal_embedding(t, dim=self.cond_emb.embedding_dim)
            c_emb = self.cond_emb(cond)
            joint = torch.cat([t_emb, c_emb], dim=1)
            out = out + self.fc(joint)[:, :, None, None]

        # Decoder
        for i in range(self.depth):
            out = self.upconvs[i](out)
            skip = enc_feats[-(i + 1)]
            out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = self.dec_blocks[i](torch.cat([out, skip], dim=1))

        return self.out_conv(out)