# flow_matching.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N,1) or (N,)
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        if half <= 0:
            return torch.zeros((t.size(0), self.dim), device=t.device, dtype=t.dtype)

        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def sinkhorn_coupling(cost, eps=0.05, iters=50):
    """
    cost: (N,N)  non-negative
    returns P: (N,N) approx OT coupling (rows/cols ~ uniform)
    """
    N = cost.size(0)
    a = torch.full((N,), 1.0 / N, device=cost.device, dtype=cost.dtype)
    b = torch.full((N,), 1.0 / N, device=cost.device, dtype=cost.dtype)

    K = torch.exp(-cost / eps).clamp_min(1e-12)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.t() @ u + 1e-12)

    P = (u[:, None] * K) * v[None, :]
    return P


class GlobalLocalVelocityField(nn.Module):
    """
    v(x,t,g,l) = vG(x,t,g) + vL(x,t,l)
    - x: (N,D)
    - t: (N,1)
    - g: (N,G)
    - l: (N,L)
    """
    def __init__(self, x_dim, g_dim, l_dim, hidden=256, time_dim=64):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.x_proj = nn.Linear(x_dim, hidden)

        self.g_proj = nn.Linear(g_dim, hidden) if g_dim > 0 else None
        self.l_proj = nn.Linear(l_dim, hidden) if l_dim > 0 else None

        self.head_g = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )
        self.head_l = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t, g=None, l=None):
        if t.dim() == 1:
            t = t[:, None]
        ht = self.time_mlp(self.time_emb(t))

        hx = self.x_proj(x)
        h = hx + ht

        if self.g_proj is not None and g is not None:
            h_g = h + self.g_proj(g)
        else:
            h_g = h

        if self.l_proj is not None and l is not None:
            h_l = h + self.l_proj(l)
        else:
            h_l = h

        v_g = self.head_g(h_g)
        v_l = self.head_l(h_l)
        return v_g + v_l


class ConditionalFlowMatcher(nn.Module):
    """
    Conditional Flow Matching (CFM) with:
      - baseline pairwise CFM loss()
      - expectation-based global velocity field loss_expectation()
      - optional bidirectional cycle consistency via loss_cycle()

    Linear probability path:
      x_t = (1-t) x0 + t x1 + sigma * eps
      v*  = x1 - x0
    """
    def __init__(
        self,
        x_dim,
        g_dim,
        l_dim,
        hidden=256,
        time_dim=64,
        sigma=0.0,
        q_sigma=0.1,
        q_temp=1.0,
        pool_chunk=2048,
        ot_eps=0.05,
        ot_iters=50,
        ot_max_n=512,
        # --- bidirectional / cycle controls ---
        bidirectional=True,
        lambda_cyc=0.0,
        cycle_steps=16,
        cycle_max_n=4096,
    ):
        super().__init__()
        self.net = GlobalLocalVelocityField(
            x_dim=x_dim, g_dim=g_dim, l_dim=l_dim,
            hidden=hidden, time_dim=time_dim
        )
        self.sigma = float(sigma)

        # expectation posterior controls
        self.q_sigma = float(q_sigma)
        self.q_temp = float(q_temp)
        self.pool_chunk = int(pool_chunk)

        # OT (optional for baseline loss)
        self.ot_eps = float(ot_eps)
        self.ot_iters = int(ot_iters)
        self.ot_max_n = int(ot_max_n)

        # bidirectional / cycle ablation controls
        self.bidirectional = bool(bidirectional)
        self.lambda_cyc = float(lambda_cyc)
        self.cycle_steps = int(cycle_steps)
        self.cycle_max_n = int(cycle_max_n)

    @torch.no_grad()
    def ot_barycentric_map(self, x0, x1):
        C = torch.cdist(x0, x1, p=2.0) ** 2
        C = C.float()
        P = sinkhorn_coupling(C, eps=self.ot_eps, iters=self.ot_iters).to(dtype=x1.dtype)
        row = P.sum(dim=1, keepdim=True).clamp_min(1e-12)
        x1_bar = (P @ x1) / row
        return x1_bar

    def loss(self, x0, x1, g=None, l=None, use_ot=False):
        """
        baseline pairwise CFM loss
        """
        N, _ = x0.shape

        if use_ot and (N > 1) and (N <= self.ot_max_n):
            x1_use = self.ot_barycentric_map(x0.detach(), x1.detach())
        else:
            x1_use = x1

        t = torch.rand((N, 1), device=x0.device, dtype=x0.dtype)
        xt = (1.0 - t) * x0 + t * x1_use
        if self.sigma > 0:
            xt = xt + self.sigma * torch.randn_like(xt)

        v_target = (x1_use - x0).detach()
        v_pred = self.net(xt, t, g=g, l=l)
        return torch.mean((v_pred - v_target) ** 2)

    def _expand_cond(self, cond, B, K):
        if cond is None:
            return None
        if cond.dim() == 2:
            return cond[:, None, :].expand(B, K, cond.size(-1))
        if cond.dim() == 3:
            assert cond.size(0) == B and cond.size(1) == K
            return cond
        raise ValueError(f"Unsupported cond dim: {cond.dim()}")

    def loss_expectation(self, z_seq: torch.Tensor, g=None, l=None):
        """
        Expectation-based global velocity field:
          v_G(z,t) = E_{j~q_t(j|z)} [x_j - x_{j-1}]

        z_seq: (B,T,D)
        candidates: all edges in batch => M = B*(T-1)
        q_t uses distances between z_t and P_j(t)=(1-t)x_{j-1}+t x_j

        NOTE: no top-k, no memory bank. Streaming over candidate edges by chunks.
        """
        assert z_seq.dim() == 3, f"z_seq must be (B,T,D), got {z_seq.shape}"
        B, T, D = z_seq.shape
        if T <= 1:
            return torch.zeros((), device=z_seq.device, dtype=z_seq.dtype)

        K = T - 1
        z0 = z_seq[:, :-1, :]
        z1 = z_seq[:,  1:, :]

        # sample t per edge
        t = torch.rand((B, K, 1), device=z_seq.device, dtype=z_seq.dtype)
        zt = (1.0 - t) * z0 + t * z1
        if self.sigma > 0:
            zt = zt + self.sigma * torch.randn_like(zt)

        # flatten query edges
        Nq = B * K
        zt_q = zt.reshape(Nq, D).detach().float()
        t_q  = t.reshape(Nq, 1).detach().float()

        # candidate edges (detached)
        c0 = z0.reshape(-1, D).detach().float()
        c1 = z1.reshape(-1, D).detach().float()
        dx = (c1 - c0)  # (M,D)
        M = c0.size(0)

        # posterior params
        sigma_q = max(self.q_sigma, 1e-4)
        inv_2s2 = 1.0 / (2.0 * (sigma_q ** 2))
        temp = max(self.q_temp, 1e-6)
        chunk = max(1, self.pool_chunk)

        # 1) pass: max logw for log-sum-exp stability
        max_logw = torch.full((Nq,), -float("inf"), device=zt_q.device, dtype=torch.float32)
        for s in range(0, M, chunk):
            e = min(M, s + chunk)
            c0c = c0[s:e]  # (C,D)
            c1c = c1[s:e]

            tt = t_q.view(Nq, 1, 1)
            mu = (1.0 - tt) * c0c.view(1, -1, D) + tt * c1c.view(1, -1, D)  # (Nq,C,D)
            diff = zt_q.view(Nq, 1, D) - mu
            dist2 = (diff * diff).sum(dim=-1)  # (Nq,C)

            logw = -(dist2 * inv_2s2) / temp
            max_logw = torch.maximum(max_logw, logw.max(dim=1).values)

        # 2) pass: denom + numer
        denom = torch.zeros((Nq,), device=zt_q.device, dtype=torch.float32)
        numer = torch.zeros((Nq, D), device=zt_q.device, dtype=torch.float32)

        for s in range(0, M, chunk):
            e = min(M, s + chunk)
            c0c = c0[s:e]
            c1c = c1[s:e]
            dxc = dx[s:e]  # (C,D)

            tt = t_q.view(Nq, 1, 1)
            mu = (1.0 - tt) * c0c.view(1, -1, D) + tt * c1c.view(1, -1, D)
            diff = zt_q.view(Nq, 1, D) - mu
            dist2 = (diff * diff).sum(dim=-1)
            logw = -(dist2 * inv_2s2) / temp

            w = torch.exp(logw - max_logw.view(Nq, 1))
            denom = denom + w.sum(dim=1)
            numer = numer + (w @ dxc)

        denom = denom.clamp_min(1e-12)
        v_target = (numer / denom.view(Nq, 1)).to(dtype=z_seq.dtype).detach()

        # expand cond to per-edge
        g_k = self._expand_cond(g, B, K)
        l_k = self._expand_cond(l, B, K)
        g_f = g_k.reshape(Nq, g_k.size(-1)) if g_k is not None else None
        l_f = l_k.reshape(Nq, l_k.size(-1)) if l_k is not None else None

        v_pred = self.net(zt.reshape(Nq, D), t.reshape(Nq, 1), g=g_f, l=l_f)
        return torch.mean((v_pred - v_target) ** 2)

    def loss_cycle(self, x0: torch.Tensor, g=None, l=None, n_steps=None):
        """
        Differentiable bidirectional cycle consistency:
          x0 -> integrate forward 0->1
          then integrate backward 1->0 using sign flip (-v) with reversed time
        """
        if n_steps is None:
            n_steps = self.cycle_steps

        B, D = x0.shape
        dt = 1.0 / float(n_steps)

        # forward
        x = x0
        t = torch.zeros((B, 1), device=x0.device, dtype=x0.dtype)
        for _ in range(n_steps):
            v = self.net(x, t, g=g, l=l)
            x = x + dt * v
            t = t + dt

        # backward: reverse time and flip sign
        t = torch.ones((B, 1), device=x0.device, dtype=x0.dtype)
        for _ in range(n_steps):
            v = self.net(x, t, g=g, l=l)
            x = x - dt * v
            t = t - dt

        return torch.mean((x - x0) ** 2)

    def loss_expectation_with_cycle(self, z_seq: torch.Tensor, g=None, l=None, return_parts=False):
        """
        Total transition-flow loss:
            L_total = L_G + lambda_cyc * L_cyc   (if bidirectional enabled)
                    = L_G                         (otherwise)
        """
        loss_g = self.loss_expectation(z_seq=z_seq, g=g, l=l)

        if (not self.bidirectional) or (self.lambda_cyc <= 0.0):
            loss_cyc = torch.zeros((), device=z_seq.device, dtype=z_seq.dtype)
            loss_total = loss_g
        else:
            B, T, D = z_seq.shape
            if T <= 1:
                loss_cyc = torch.zeros((), device=z_seq.device, dtype=z_seq.dtype)
                loss_total = loss_g
            else:
                K = T - 1
                x0 = z_seq[:, :-1, :].reshape(B * K, D)

                g_k = self._expand_cond(g, B, K)
                l_k = self._expand_cond(l, B, K)
                g_f = g_k.reshape(B * K, g_k.size(-1)) if g_k is not None else None
                l_f = l_k.reshape(B * K, l_k.size(-1)) if l_k is not None else None

                # optional subsampling for speed
                N = x0.size(0)
                if self.cycle_max_n > 0 and N > self.cycle_max_n:
                    idx = torch.randperm(N, device=x0.device)[:self.cycle_max_n]
                    x0_c = x0[idx]
                    g_c = g_f[idx] if g_f is not None else None
                    l_c = l_f[idx] if l_f is not None else None
                else:
                    x0_c, g_c, l_c = x0, g_f, l_f

                loss_cyc = self.loss_cycle(x0_c, g=g_c, l=l_c, n_steps=self.cycle_steps)
                loss_total = loss_g + self.lambda_cyc * loss_cyc

        if return_parts:
            return loss_total, {"loss_g": loss_g, "loss_cyc": loss_cyc}
        return loss_total

    @torch.no_grad()
    def step_euler(self, x, g=None, l=None, n_steps=16):
        """
        Simple Euler ODE solve from t=0 -> 1:
          dx/dt = v(x,t,g,l)
        """
        dt = 1.0 / float(n_steps)
        t = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        for _ in range(n_steps):
            v = self.net(x, t, g=g, l=l)
            x = x + dt * v
            t = t + dt
        return x