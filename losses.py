# losses.py
import torch
import torch.nn.functional as F

def stft_mag(x, n_fft, hop, win):
    w = torch.hann_window(win, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=w, center=True, return_complex=True
    )
    return torch.abs(X)

def mrstft_loss(y_hat, y, fft_sizes=(256, 512, 1024), hop_ratio=0.25, log_eps=1e-5):
    loss = 0.0
    for n_fft in fft_sizes:
        hop = int(n_fft * hop_ratio)
        win = n_fft

        Yh = stft_mag(y_hat, n_fft, hop, win)  # (B,F,T)
        Y  = stft_mag(y,     n_fft, hop, win)

        # spectral convergence (per-sample)
        diff = (Y - Yh).reshape(Y.shape[0], -1)
        ref  = Y.reshape(Y.shape[0], -1)
        sc = (torch.norm(diff, dim=1) / (torch.norm(ref, dim=1) + 1e-8)).mean()

        # log-mag L1
        lm = F.l1_loss(torch.log(Yh.clamp_min(log_eps)), torch.log(Y.clamp_min(log_eps)))

        loss = loss + sc + lm
    return loss / len(fft_sizes)

def envelope(x, kernel=129):
    x = torch.abs(x).unsqueeze(1)  # (B,1,N)
    pad = kernel // 2
    if x.shape[-1] <= pad:
        x = F.pad(x, (pad, pad), mode="replicate")
    else:
        x = F.pad(x, (pad, pad), mode="reflect")
    x = F.avg_pool1d(x, kernel_size=kernel, stride=1)
    return x.squeeze(1)

def envelope_loss(y_hat, y, kernel=129):
    return F.l1_loss(envelope(y_hat, kernel=kernel), envelope(y, kernel=kernel))
