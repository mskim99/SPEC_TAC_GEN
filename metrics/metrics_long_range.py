# metrics_long_range.py
from torch import nn
import torch
import torch.optim as optim
import numpy as np
from types import SimpleNamespace

from tqdm.auto import tqdm
from models.testing_models.s4d import S4D, dropout_fn

from metrics.discriminative_torch import discriminative_score_metrics


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_state,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        bidirectional=False,
        seq2seq=False,
        lr=0.001,
        activation=nn.Identity()
    ):
        super().__init__()
        self.prenorm = prenorm
        self.seq2seq = seq2seq

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
                    lr=min(0.001, lr)
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout) if dropout > 0 else nn.Identity())

        self.decoder = nn.Linear(d_model, d_output)
        self.act = activation

    def forward(self, x, aux=None, t=None, **kwargs):
        x = self.encoder(x)      # (B,L,d_model)
        x = x.transpose(-1, -2)  # (B,d_model,L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)
            x = z + x

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B,L,d_model)

        if not self.seq2seq:
            x = x.mean(1)        # (B,d_model)

        x = self.decoder(x)
        x = self.act(x)
        return x, None


def compute_discriminative_score(x_real, x_fake, device, n_runs=10, iters=2000, batch_size=32):
    """
    x_real/x_fake: torch (N,L,C)
    returns mean/std of TimeGAN-style discriminative score: |0.5-acc|
    """
    xr = x_real.detach().cpu().numpy()
    xf = x_fake.detach().cpu().numpy()

    args = SimpleNamespace(
        input_size=xr.shape[-1],
        device=str(device),
    )

    vals = []
    for _ in range(n_runs):
        v = discriminative_score_metrics(xr, xf, args=args)
        vals.append(v)

    return float(np.mean(vals)), float(np.std(vals))


def compute_predictive_score(x_real, x_fake, pred_step, get_optim_func, device, pred_activation):
    x_fake = x_fake.detach().cpu()
    x_real = x_real.detach().cpu()

    X = x_fake[:, :-1]
    Y = x_fake[:, 1:]

    masks = torch.ones_like(X, dtype=torch.bool)
    masks[:, :-pred_step] = 0

    X_test = x_real[:, :-1]
    Y_test = x_real[:, 1:]

    C = X.shape[-1]  # number of channels/features

    model = S4Model(
        d_input=C, d_state=16, d_output=C, d_model=16, n_layers=1,
        dropout=0.0, seq2seq=True, activation=pred_activation
    ).to(device)

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y, masks), shuffle=True, batch_size=128
    )
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test, masks), batch_size=128
    )

    optimizer, _ = get_optim_func(model, lr=0.01, weight_decay=0.0, epochs=100)

    pbar = tqdm(range(100))
    for i in range(100):
        for data, target, mask in trainloader:
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(data.to(device))
            loss = torch.nn.MSELoss()(pred[mask], target.to(device)[mask])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            for ind, (data, target, mask) in enumerate(testloader):
                pred, _ = model(data.to(device))
                loss = torch.nn.MSELoss()(pred[mask], target.to(device)[mask]).detach().cpu()
                test_loss += loss
            pbar.set_description(f'Epoch {i} Test loss: {test_loss / (ind + 1)}')

    return test_loss


def compute_all_metrics(x_real, gens, get_optim_func, pred_activation, device,
                        n_runs=10, pred_step=10,
                        disc_iters=2000, disc_batch=32):
    """
    Returns ONLY:
      - discriminative_score_mean/std (TimeGAN style |0.5-acc|)
      - predictive_score_mean/std
    """
    disc_mean, disc_std = compute_discriminative_score(
        x_real=x_real, x_fake=gens, device=device,
        n_runs=n_runs, iters=disc_iters, batch_size=disc_batch
    )

    pred = []
    for _ in range(n_runs):
        predscore = compute_predictive_score(x_real, gens, pred_step, get_optim_func, device, pred_activation)
        pred.append(predscore)

    pred_mean, pred_std = float(np.mean(pred)), float(np.std(pred))

    return {
        "discriminative_score_mean": disc_mean,
        "discriminative_score_std":  disc_std,
        "predictive_score_mean":     pred_mean,
        "predictive_score_std":      pred_std,
    }


def setup_optimizer(model, lr, weight_decay, epochs):
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler
