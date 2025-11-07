# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# [NEW] Complex-valued helpers
# -----------------------------
class ComplexConv2d(nn.Module):  # 복소 컨볼루션: (xr, xi) -> (yr, yi)
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False, dilation=1):
        super().__init__()
        self.wr = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias, dilation=dilation)
        self.wi = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias, dilation=dilation)

    def forward(self, xr, xi):
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return yr, yi


class ComplexGroupNorm(nn.Module):  # [NEW] 채널쌍 공유 GroupNorm
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        # 실수/허수 동일 파라미터 공유를 위해 하나만 학습하고 양쪽에 적용
        self.gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels, eps=eps,
                               affine=True)

    def forward(self, xr, xi):
        # magnitude 정규화가 아니고, 채널별 통계를 공유 적용
        xr = self.gn(xr)
        xi = self.gn(xi)
        return xr, xi


class ModReLU(nn.Module):  # [NEW] 복소 활성화
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, xr, xi):
        mag = torch.sqrt(xr ** 2 + xi ** 2 + self.eps)
        scale = F.relu(mag) / (mag + self.eps)
        return xr * scale, xi * scale


class ComplexBlock(nn.Module):  # [NEW] Conv -> GN -> Act
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8, dilation=1):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, k=k, s=s, p=p, bias=False, dilation=dilation)
        self.norm = ComplexGroupNorm(groups, out_ch)
        self.act = ModReLU()

    def forward(self, xr, xi):
        xr, xi = self.conv(xr, xi)
        xr, xi = self.norm(xr, xi)
        xr, xi = self.act(xr, xi)
        return xr, xi


# -----------------------------
# [NEW] Complex down / up
# -----------------------------
class ComplexDown(nn.Module):  # stride-2 복소 다운샘플
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.block = ComplexBlock(in_ch, out_ch, k=3, s=2, p=1, groups=groups)

    def forward(self, xr, xi): return self.block(xr, xi)


class ComplexUp(nn.Module):  # 업샘플 + 복소 컨볼루션
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv = ComplexBlock(in_ch, out_ch, k=3, s=1, p=1, groups=groups)

    def forward(self, xr, xi, size):
        xr = F.interpolate(xr, size=size, mode="bilinear", align_corners=False)
        xi = F.interpolate(xi, size=size, mode="bilinear", align_corners=False)
        return self.conv(xr, xi)


# -----------------------------
# [NEW] Complex ASPP (dilated context)
# -----------------------------
class ComplexASPP(nn.Module):
    def __init__(self, ch, rates=(1, 2, 4), groups=8):
        super().__init__()
        self.branches = nn.ModuleList([
            ComplexBlock(ch, ch, k=3, s=1, p=r, groups=groups, dilation=r) for r in rates
        ])
        # 1x1 복소 축소
        self.proj = ComplexBlock(ch * len(rates), ch, k=1, s=1, p=0, groups=groups)
        self.rates = rates

    def forward(self, xr, xi):
        outs_r, outs_i = [], []
        for r, b in zip(self.rates, self.branches):
            # dilated conv 효과: padding=r, kernel=3 으로 receptive field 증가
            yr, yi = b.conv(xr, xi)  # Conv만 먼저
            yr, yi = b.norm(yr, yi)
            yr, yi = b.act(yr, yi)
            outs_r.append(yr);
            outs_i.append(yi)
        xr = torch.cat(outs_r, dim=1);
        xi = torch.cat(outs_i, dim=1)
        xr, xi = self.proj(xr, xi)
        return xr, xi


# -----------------------------
# [NEW] Align head (a = ar + j ai)
# -----------------------------
class ComplexAlignHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # 입력 채널을 ch (mag)에서 ch*2 (xr, xi concat)로 변경
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, ch // 2, 1), nn.SiLU(),  # ch -> ch*2, ch//4 -> ch//2
            nn.Conv2d(ch // 2, 2, 1)  # (ar, ai)
        )

    def forward(self, yr, yi):
        # (yr, yi)를 concat하여 입력으로 사용
        y_cat = torch.cat([yr, yi], dim=1)  # (B, 2*ch, H, W)
        coeff = self.head(y_cat)  # (B, 2, 1, 1)

        ar = coeff[:, 0:1];
        ai = coeff[:, 1:2]
        y_r = ar * yr - ai * yi
        y_i = ar * yi + ai * yr
        return y_r, y_i, ar, ai


# -----------------------------
# [NEW] Sinusoidal timestep embedding (유지)
# -----------------------------
def sinusoidal_embedding(t, dim=128):
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-torch.log(torch.tensor(10000.0)) / (half - 1)))
    emb = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


# -----------------------------
# [CHANGED] UNet -> Complex UNet with ASPP + Align Head
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_ch=64, ch_mult=(1, 2, 4, 8),
                 conditional=False, num_classes=0, emb_dim=128, groups=8):
        super().__init__()
        assert in_ch == 2 and out_ch == 2, "Complex UNet expects 2-channel (Real, Imag) I/O"
        self.conditional = conditional
        self.depth = len(ch_mult) - 1
        self.ch_mult = ch_mult
        self.groups = groups

        # [NEW] Encoder (complex)
        self.enc_blocks = nn.ModuleList()
        prev = base_ch * ch_mult[0]

        # 2ch 입력을 2*prev ch로 변환하여 xr, xi를 생성
        self.in_proj = nn.Conv2d(in_ch, prev * 2, 1, bias=False)  # 2 -> 2*prev

        self.enc_blocks.append(ComplexBlock(prev, prev, k=3, s=1, p=1, groups=groups))  # stage 0

        for i in range(1, len(ch_mult)):
            in_c = base_ch * ch_mult[i - 1]
            out_c = base_ch * ch_mult[i]
            self.enc_blocks.append(ComplexDown(in_c, out_c, groups=groups))

        # [NEW] Bottleneck: Complex ASPP
        bottleneck_ch = base_ch * ch_mult[-1]
        self.aspp = ComplexASPP(bottleneck_ch, rates=(1, 2, 4), groups=groups)

        # [NEW] Decoder (complex up + skip concat)
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(self.depth)):
            in_c = base_ch * ch_mult[i + 1]
            out_c = base_ch * ch_mult[i]
            self.upconvs.append(ComplexUp(in_c, out_c, groups=groups))
            # 업샘플 결과(out_c)와 skip(out_c)을 concat → 2*out_c -> out_c 축소
            self.dec_blocks.append(ComplexBlock(out_c * 2, out_c, k=1, s=1, p=0, groups=groups))

        # [NEW] Align head (전역 위상/스케일 보정)
        # self.align = ComplexAlignHead(base_ch * ch_mult[0])

        # 독립적인 r, i conv 대신 ComplexConv2d 사용
        self.out_conv = ComplexConv2d(base_ch * ch_mult[0], 1, k=1, s=1, p=0, bias=False)

        # [CHANGED] Optional conditional embedding (magnitude로 주입)
        if self.conditional:
            self.cond_emb = nn.Embedding(num_classes, emb_dim)
            self.t_mlp = nn.Linear(emb_dim, bottleneck_ch * 2)

    def forward(self, x, t=None, cond=None):
        B, C, H, W = x.shape
        assert C == 2, "Expect input as (B,2,H,W) -> (Real, Imag)"

        # 문제 1 수정: in_proj를 사용하여 xr, xi 생성 (repeat 제거)
        h = self.in_proj(x)  # (B, 2*C0, H, W)
        xr, xi = torch.chunk(h, 2, dim=1)  # (B, C0, H, W) each

        # 입력 x를 (xr, xi)로 분리 → 임베딩은 magnitude 가중치로 본문에 섞이지 않게 바로 사용하지 않음
        xr, xi = self.enc_blocks[0](xr, xi)  # stage 0

        feats = [(xr, xi)]

        # Encoder stages
        for i in range(1, len(self.enc_blocks)):
            xr, xi = self.enc_blocks[i](xr, xi)
            feats.append((xr, xi))

        # Bottleneck ASPP
        xr, xi = self.aspp(xr, xi)

        # Optional conditional
        if self.conditional and t is not None and cond is not None:
            t_emb = sinusoidal_embedding(t, dim=self.cond_emb.embedding_dim)  # (B,emb_dim)
            c_emb = self.cond_emb(cond)  # (B,emb_dim)

            # add_r, add_i를 분리하여 xr, xi에 각각 더함
            emb = self.t_mlp(t_emb + c_emb).view(B, -1, 1, 1)  # (B, bottleneck_ch * 2, 1, 1)
            add_r, add_i = torch.chunk(emb, 2, dim=1)  # (B, bottleneck_ch, 1, 1) each

            xr = xr + add_r
            xi = xi + add_i

        # Decoder with complex skip concat
        for i, up in enumerate(self.upconvs):
            skip_r, skip_i = feats[-(i + 2)]
            xr, xi = up(xr, xi, size=skip_r.shape[2:])
            # concat real & imag 각각 동일 채널 수 기준
            xr = torch.cat([xr, skip_r], dim=1)
            xi = torch.cat([xi, skip_i], dim=1)
            xr, xi = self.dec_blocks[i](xr, xi)

        # Align head (전역 위상/스케일 보정)
        # xr, xi, ar, ai = self.align(xr, xi)

        # out_conv (ComplexConv2d)를 통해 yr, yi 계산
        yr, yi = self.out_conv(xr, xi)
        y = torch.cat([yr, yi], dim=1)

        # 출력 크기 보정(필요 시)
        if y.shape[2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        y = torch.tanh(y)

        return y