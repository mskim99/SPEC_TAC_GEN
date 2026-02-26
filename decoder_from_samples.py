# sample_strategyA.py (MULTI-CHANNEL FINAL + random material + total/microbatch + png)
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_matching import ConditionalFlowMatcher
from decoder_model import ZOnlyDecoder1D


class ContainerPredictor(nn.Module):
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
        x = torch.cat([z, tpos, g], dim=-1)
        return self.net(x)


def save_wave_png(y: np.ndarray, path: str, title: str = None, dpi: int = 180, max_points: int = 200000):
    """
    y: (L,) or (L,C)
    멀티채널이면 채널별 subplot 저장
    """
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        y = y[:, None]  # (L,1)
    elif y.ndim != 2:
        raise ValueError(f"Waveform must be 1D or 2D. got {y.shape}")

    L, C = y.shape

    if L > max_points:
        step = int(np.ceil(L / max_points))
        x = np.arange(0, L, step)
        y_plot = y[::step]
    else:
        x = np.arange(L)
        y_plot = y

    fig_h = max(2.5, 2.2 * C)
    fig, axes = plt.subplots(C, 1, figsize=(12, fig_h), squeeze=False)
    axes = axes[:, 0]

    for c in range(C):
        axes[c].plot(x, y_plot[:, c], linewidth=0.8)
        axes[c].set_ylabel(f"ch{c}")
        if c != C - 1:
            axes[c].tick_params(labelbottom=False)

    axes[-1].set_xlabel("sample_idx")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


def _infer_dims_from_cfg(cfg: dict):
    flowits_dim = int(cfg.get("flowits_dim", 128))
    num_channels = int(cfg.get("num_channels", 1))
    channel_fusion = str(cfg.get("channel_fusion", "concat"))

    if "eff_z_dim" in cfg:
        eff_z_dim = int(cfg["eff_z_dim"])
    else:
        if channel_fusion == "concat":
            eff_z_dim = flowits_dim * num_channels
        elif channel_fusion in ["mean", "sum"]:
            eff_z_dim = flowits_dim
        else:
            raise ValueError(f"Unknown channel_fusion={channel_fusion}")

    return flowits_dim, num_channels, channel_fusion, eff_z_dim


def _resolve_material_ids(
    B: int,
    materials: list[str],
    material_to_id: dict,
    args,
    device,
):
    """
    returns:
      mat_t: (B,) long
      mat_names: list[str] len B
    """
    n_mat = len(materials)
    if n_mat <= 0:
        mat_t = torch.zeros((B,), device=device, dtype=torch.long)
        return mat_t, ["unknown"] * B

    # random mode 우선
    if args.random_material:
        mat_np = np.random.randint(0, n_mat, size=(B,), dtype=np.int64)
        mat_t = torch.from_numpy(mat_np).to(device=device, dtype=torch.long)
        mat_names = [materials[int(i)] for i in mat_np.tolist()]
        return mat_t, mat_names

    # fixed mode
    if args.material is None:
        mid = 0
        mname = materials[0]
    else:
        mid = int(material_to_id.get(args.material, 0))
        mname = materials[mid]

    mat_t = torch.full((B,), mid, device=device, dtype=torch.long)
    return mat_t, [mname] * B


@torch.no_grad()
def main(args):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})

    materials = cfg.get("materials", ["unknown"])
    if not isinstance(materials, list) or len(materials) == 0:
        materials = ["unknown"]
    material_to_id = {m: i for i, m in enumerate(materials)}

    flowits_dim, num_channels, channel_fusion, eff_z_dim = _infer_dims_from_cfg(cfg)
    g_dim = int(cfg.get("flow_mat_cond_dim", 0))
    c_dim = int(cfg.get("container_dim", 64))

    # -----------------------------
    # Build models
    # -----------------------------
    prior_flow = ConditionalFlowMatcher(
        x_dim=eff_z_dim, g_dim=g_dim, l_dim=0,
        hidden=int(cfg.get("flow_hidden", 256)),
        time_dim=int(cfg.get("flow_time_dim", 64)),
        sigma=float(cfg.get("flow_sigma", 0.01)),
        q_sigma=float(cfg.get("flow_q_sigma", 0.10)),
        q_temp=float(cfg.get("flow_q_temp", 1.0)),
        pool_chunk=int(cfg.get("flow_pool_chunk", 2048)),
        ot_eps=float(cfg.get("flow_ot_eps", 0.05)),
        ot_iters=int(cfg.get("flow_ot_iters", 50)),
        ot_max_n=int(cfg.get("flow_ot_max_n", 512)),
    ).to(device)

    trans_flow = ConditionalFlowMatcher(
        x_dim=eff_z_dim, g_dim=g_dim, l_dim=c_dim,
        hidden=int(cfg.get("flow_hidden", 256)),
        time_dim=int(cfg.get("flow_time_dim", 64)),
        sigma=float(cfg.get("flow_sigma", 0.01)),
        q_sigma=float(cfg.get("flow_q_sigma", 0.10)),
        q_temp=float(cfg.get("flow_q_temp", 1.0)),
        pool_chunk=int(cfg.get("flow_pool_chunk", 2048)),
        ot_eps=float(cfg.get("flow_ot_eps", 0.05)),
        ot_iters=int(cfg.get("flow_ot_iters", 50)),
        ot_max_n=int(cfg.get("flow_ot_max_n", 512)),
    ).to(device)

    zdec = ZOnlyDecoder1D(
        hop_len=int(cfg.get("hop_len", 16)),
        win_len=int(cfg.get("win_len", 16)),
        flowits_dim=eff_z_dim,
        up_ch=int(cfg.get("dec_up_ch", 128)),
        up_factors=tuple(cfg.get("dec_up_factors", [4, 4])),
        num_materials=int(cfg.get("num_materials", len(materials))),
        cond_dim=int(cfg.get("cond_dim", 32)),
        out_channels=num_channels,   # 멀티채널 출력
    ).to(device)

    # heads
    mat_flow = None
    if g_dim > 0:
        mat_flow = nn.Embedding(int(cfg.get("num_materials", len(materials))), g_dim).to(device)

    cont_pred = ContainerPredictor(
        z_dim=eff_z_dim, g_dim=g_dim, c_dim=c_dim,
        hidden=int(cfg.get("container_pred_hidden", 256))
    ).to(device)

    # load states
    prior_flow.load_state_dict(ckpt["prior_flow"], strict=True)
    trans_flow.load_state_dict(ckpt["trans_flow"], strict=True)
    zdec.load_state_dict(ckpt["z_decoder"], strict=True)

    heads = ckpt.get("heads", {})
    if mat_flow is not None and "mat_flow" in heads:
        mat_flow.load_state_dict(heads["mat_flow"], strict=True)
    if "cont_pred" in heads:
        cont_pred.load_state_dict(heads["cont_pred"], strict=True)

    prior_flow.eval()
    trans_flow.eval()
    zdec.eval()
    cont_pred.eval()
    if mat_flow is not None:
        mat_flow.eval()

    # sampling config
    T = int(args.frames)
    micro_B = max(1, int(args.micro_batch))
    total = int(args.total if args.total is not None else micro_B)
    total = max(1, total)

    # save run meta
    run_meta = {
        "ckpt": args.ckpt,
        "frames": T,
        "prior_steps": int(args.prior_steps),
        "trans_steps": int(args.trans_steps),
        "micro_batch": micro_B,
        "total": total,
        "random_material": bool(args.random_material),
        "material_arg": args.material,
        "materials": materials,
        "num_channels": num_channels,
        "flowits_dim": flowits_dim,
        "eff_z_dim": eff_z_dim,
        "channel_fusion": channel_fusion,
    }
    with open(os.path.join(args.out_dir, "sample_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # per-sample metadata log
    sample_log_path = os.path.join(args.out_dir, "sample_index.jsonl")

    global_idx = 0
    with open(sample_log_path, "w", encoding="utf-8") as flog:
        while global_idx < total:
            B = min(micro_B, total - global_idx)

            # material ids (fixed or random)
            mat_t, mat_names = _resolve_material_ids(
                B=B,
                materials=materials,
                material_to_id=material_to_id,
                args=args,
                device=device,
            )

            # global cond g
            if g_dim > 0 and mat_flow is not None:
                g = mat_flow(mat_t)  # (B,G)
            else:
                g = None

            # ---- sample z0 via prior flow ----
            x0 = torch.randn((B, eff_z_dim), device=device)
            z0 = prior_flow.step_euler(x0, g=g, l=None, n_steps=int(args.prior_steps))  # (B,D_eff)

            # ---- rollout z_seq via trans flow + predicted containers ----
            z_list = [z0]
            zt = z0
            for i in range(T - 1):
                tpos = torch.full((B, 1), float(i) / max(T - 1, 1), device=device, dtype=zt.dtype)

                if g is None:
                    gf = torch.zeros((B, 0), device=device, dtype=zt.dtype)
                else:
                    gf = g.to(dtype=zt.dtype)

                c = cont_pred(zt, tpos, gf)  # (B,Cdim)
                zt = trans_flow.step_euler(zt, g=g, l=c, n_steps=int(args.trans_steps))
                z_list.append(zt)

            z_seq = torch.stack(z_list, dim=1)  # (B,T,D_eff)

            # ---- decode waveform ----
            y = zdec(z_seq, mat_id=mat_t).detach().cpu().numpy()
            # y shape:
            # - single: (B,L)
            # - multi : (B,L,C)

            # save outputs
            for i in range(B):
                idx = global_idx + i
                tag = f"{args.tag_prefix}_{idx:06d}"

                yi = y[i]
                # single-channel도 저장 형식 통일하고 싶으면 (L,1)로 바꿔도 되지만,
                # 여기선 decoder 출력 그대로 저장
                np.save(os.path.join(args.out_dir, f"{tag}_wave.npy"), yi)

                if args.save_png:
                    # plot helper는 (L,) / (L,C) 모두 지원
                    save_wave_png(
                        yi,
                        os.path.join(args.out_dir, f"{tag}_wave.png"),
                        title=f"{tag} | mat={mat_names[i]}(id={int(mat_t[i].item())}) | T={T}",
                        dpi=int(args.png_dpi),
                        max_points=int(args.png_max_points),
                    )

                rec = {
                    "index": int(idx),
                    "tag": tag,
                    "material_name": mat_names[i],
                    "material_id": int(mat_t[i].item()),
                    "frames": int(T),
                    "shape": list(np.asarray(yi).shape),
                }
                flog.write(json.dumps(rec, ensure_ascii=False) + "\n")

            global_idx += B
            if device.type == "cuda" and args.empty_cache_each_iter:
                torch.cuda.empty_cache()

    print("✅ Done")
    print(f"  out_dir      : {args.out_dir}")
    print(f"  total        : {total}")
    print(f"  micro_batch  : {micro_B}")
    print(f"  frames       : {T}")
    print(f"  num_channels : {num_channels}")
    print(f"  material mode: {'random' if args.random_material else 'fixed'}")
    if not args.random_material:
        print(f"  material     : {args.material if args.material is not None else materials[0]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="gen_out")
    p.add_argument("--tag_prefix", type=str, default="sample")

    # sampling count controls
    p.add_argument("--micro_batch", type=int, default=16, help="number of samples to generate per iteration")
    p.add_argument("--total", type=int, default=3000, help="total number of samples to generate (overrides micro_batch if set)")

    # sequence rollout
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--prior_steps", type=int, default=50)
    p.add_argument("--trans_steps", type=int, default=50)

    # material controls
    p.add_argument("--material", type=str, default=None, help="fixed material name (when, random_material=False)")
    p.add_argument("--random_material", action="store_true", help="random selection of material for each sample")

    # output visualization
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--png_dpi", type=int, default=180)
    p.add_argument("--png_max_points", type=int, default=200000)

    # runtime
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--empty_cache_each_iter", action="store_true")

    args = p.parse_args()
    main(args)