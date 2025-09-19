# multi_adapt.py
# Adaptive training for 4-subdomain Nitsche problem (flower interfaces)

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# from Adaptive.visualize_adapt import abs_error_adapt
from Fc_Net import FCNet
from multi_loss import *
from multi_indicator import *  # uses gradients() internally
from multi_3flowers_pde import *
from visualize_multi import *

# ---------------- Global config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def setup_seed(seed: int = 888888):
    torch.manual_seed(seed)


def l2_error(u: torch.Tensor, uh: torch.Tensor) -> torch.Tensor:
    num = torch.sqrt(torch.mean((u - uh) ** 2))
    den = torch.sqrt(torch.mean(u ** 2)) + 1e-12
    return num / den


# ---------------- Geometry / sampling utilities ----------------
def generate_interface_points(interface_id: int, N_f: int) -> torch.Tensor:
    """Interface samples (N_f, 3) with dummy last col (0)."""
    θ = torch.linspace(0, 2 * torch.pi, N_f, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    if interface_id == 0:
        r = 0.2 - 0.1 * torch.cos(4 * θ)
    elif interface_id == 1:
        r = 0.5 - 0.1 * torch.cos(5 * θ)
    elif interface_id == 2:
        r = 0.8 - 0.1 * torch.cos(6 * θ)
    else:
        raise ValueError("interface_id must be 0, 1, or 2")
    x = r * torch.cos(θ)
    y = r * torch.sin(θ)
    xy = torch.cat([x, y], dim=1)
    return torch.cat([xy, torch.zeros(xy.size(0), 1, dtype=DTYPE, device=DEVICE)], dim=1)


def refine_interface_points(xf_2d: torch.Tensor, interface_id: int, n_insert: int = 1) -> torch.Tensor:
    """Densify interface by inserting n_insert points along each edge; returns (M,3)."""
    xf = xf_2d.to(DEVICE, DTYPE)
    N = xf.shape[0]
    segs = []
    for i in range(N):
        x0, y0 = xf[i]
        x1, y1 = xf[(i + 1) % N]
        t0 = np.arctan2(float(y0), float(x0))
        t1 = np.arctan2(float(y1), float(x1))
        if t1 < t0:
            t1 += 2 * np.pi
        t = torch.linspace(t0, t1, n_insert + 2, dtype=DTYPE, device=DEVICE)[:-1]
        if interface_id == 0:
            r = 0.2 - 0.1 * torch.cos(4 * t)
        elif interface_id == 1:
            r = 0.5 - 0.1 * torch.cos(5 * t)
        elif interface_id == 2:
            r = 0.8 - 0.1 * torch.cos(6 * t)
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        segs.append(torch.stack([x, y], dim=1))
    new2 = torch.cat(segs, dim=0)
    all2 = torch.cat([xf, new2], dim=0)

    # Dedup via coarse rounding (tolerance ≈ 1e-3)
    key = (all2 * 1e3).round().to(torch.int64)
    uniq, _ = torch.unique(key, dim=0, return_inverse=True)
    uniq2 = uniq.to(DTYPE) / 1e3
    return torch.cat([uniq2, torch.zeros(uniq2.size(0), 1, dtype=DTYPE, device=DEVICE)], dim=1)


def generate_grid_points(h: int) -> torch.Tensor:
    """Regular grid (h×h) over [-1,1]^2, returns (h*h, 3) with dummy last col."""
    xs = torch.linspace(-1, 1, steps=h, dtype=DTYPE, device=DEVICE)
    ys = torch.linspace(-1, 1, steps=h, dtype=DTYPE, device=DEVICE)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts2 = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    return torch.cat([pts2, torch.zeros(pts2.size(0), 1, dtype=DTYPE, device=DEVICE)], dim=1)


def add_grid(points_3d: torch.Tensor, refinement_factor: int, h: int) -> torch.Tensor:
    """
    Around each seed point, add a small local r×r grid with spacing scaled by 1/2^count.
    Input points_3d expected with last col "count" (refinement level), but we’ll accept any and treat last col as count.
    Returns (M,3) with updated counts, deduped & clipped to [-1,1]^2, and requires_grad for interior later.
    """
    pts = points_3d.to(DEVICE, DTYPE)
    counts = pts[:, -1:]  # (N,1), may be zeros
    # base spacing ≈ one cell width / 2^count
    base_spacing = (1.0 / (h - 1)) / torch.clamp(2.0 ** counts, min=1.0)

    span = torch.linspace(-1.0, 1.0, refinement_factor, device=DEVICE, dtype=DTYPE)
    dx, dy = torch.meshgrid(span, span, indexing="xy")
    offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)  # (r^2,2)
    # Expand offsets to (N, r^2, 2) and scale with per-point base_spacing -> (N, r^2, 2)
    N = pts.shape[0]
    offsets = offsets.unsqueeze(0).expand(N, -1, -1)
    offsets = offsets * base_spacing.view(N, 1, 1)

    new2 = pts[:, :2].unsqueeze(1) + offsets  # (N, r^2, 2)
    new2 = new2.reshape(-1, 2)

    # clip to domain box
    mask = (new2[:, 0] >= -1) & (new2[:, 0] <= 1) & (new2[:, 1] >= -1) & (new2[:, 1] <= 1)
    new2 = new2[mask]
    new_counts = (counts.repeat(1, refinement_factor * refinement_factor).reshape(-1, 1) + 1)[mask]

    out = torch.cat([new2, new_counts], dim=1)

    # dedup by rounding
    key = (out[:, :2] * 1e3).round().to(torch.int64)
    uq, _ = torch.unique(key, dim=0, return_inverse=True)
    uniq2 = uq.to(DTYPE) / 1e3
    uniqc = torch.zeros(uniq2.size(0), 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([uniq2, uniqc], dim=1).requires_grad_(True)


# ---------------- Visualization of refinement points ----------------

def plot_refinement_points(level: int,
                            xr: torch.Tensor,
                            xb: torch.Tensor,
                            xf_list: list,
                            candidates: torch.Tensor = None,
                            save: bool = True,
                            prefix: str = 'refine_level'):
    """
    Scatter-plot points used at a given adaptive level.
      - xr: interior points (N_i, 3)
      - xb: boundary points (N_b, 3)
      - xf_list: [xf0, xf1, xf2], each (N_fm, 3)
      - candidates: optional candidate refinement points (K, 3), shown as stars
    """
    def _to_np(t):
        if t is None or t.numel() == 0:
            return np.zeros((0, 2))
        t2 = t[:, :2].detach().to('cpu').numpy()
        return t2

    xr_np = _to_np(xr)
    xb_np = _to_np(xb)
    xf_np = [_to_np(xf) for xf in xf_list]
    cand_np = _to_np(candidates) if candidates is not None else np.zeros((0, 2))

    plt.figure(figsize=(6.5, 6))
    # draw domain box
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', lw=1, alpha=0.5)

    # interior
    if xr_np.size:
        plt.scatter(xr_np[:, 0], xr_np[:, 1], s=8, c='#777777', marker='o', alpha=0.6, label='interior')
    # boundary
    if xb_np.size:
        plt.scatter(xb_np[:, 0], xb_np[:, 1], s=10, c='#004c99', marker='s', alpha=0.7, label='boundary')
    # interfaces
    colors = ['#c70039', '#ff8c00', '#2ca02c']
    labels = ['Γ0', 'Γ1', 'Γ2']
    for k, xfk in enumerate(xf_np):
        if xfk.size:
            plt.scatter(xfk[:, 0], xfk[:, 1], s=10, c=colors[k%len(colors)], marker='.', alpha=0.9, label=labels[k])

    # candidates (top indicator)
    if cand_np.size:
        plt.scatter(cand_np[:, 0], cand_np[:, 1], s=60, c='red', marker='*', edgecolor='k', linewidths=0.5, label='candidates')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-1.05, 1.05]); plt.ylim([-1.05, 1.05])
    plt.title(f'Refinement points — level {level}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_{level:02d}.png", dpi=200)
    plt.show()


# ---------------- Classification ----------------
def initial_data_multisubdomain(h: int, Nf_list):
    """Start from a coarse grid + explicit interface samples."""
    grid = generate_grid_points(h)
    bmask = (torch.abs(grid[:, :2]).max(dim=1).values == 1.0)
    xb = grid[bmask]
    xr = grid[~bmask].clone().detach().requires_grad_(True)

    xf_list = []
    for i, Nf in enumerate(Nf_list):
        xf_i = generate_interface_points(i, Nf)
        xf_list.append(xf_i.clone().detach())
    return xr, xf_list, xb


def classify_x_all_multi(x_all_3d: torch.Tensor, pde: multi_interface_flower_gpt, tol: float = 2e-3):
    """
    Split mixed points (N,3) into xr (interior), xf_list (3), xb (boundary) using |phi|<tol.
    Only xr requires grad.
    """
    x2 = x_all_3d[:, :2]
    phi0, phi1, phi2 = pde.phi(x2)

    m_f0 = (phi0.abs() < tol)
    m_f1 = (phi1.abs() < tol)
    m_f2 = (phi2.abs() < tol)

    bmask = (torch.abs(x2).max(dim=1).values == 1.0)
    imask = ~(bmask | m_f0 | m_f1 | m_f2)

    xf0 = x_all_3d[m_f0].clone().detach()
    xf1 = x_all_3d[m_f1].clone().detach()
    xf2 = x_all_3d[m_f2].clone().detach()

    xb = x_all_3d[bmask].clone().detach()
    xr = x_all_3d[imask].clone().detach().requires_grad_(True)
    return xr, [xf0, xf1, xf2], xb


# ---------------- Training (single level) ----------------
def train_single_level(
    epochs: int,
    gamma_b,
    gamma_f_list,
    xr_2d: torch.Tensor,
    xf_list_2d: list,
    xb_2d: torch.Tensor,
    nets: list,
    opts: list,
    pde: multi_interface_flower_gpt,
    print_prefix: str = "",
):
    # Eval set
    x_t = pde.interior_points(10000).to(DEVICE, DTYPE)
    u_exact = pde.exact_judge(x_t).detach()

    for ep in range(epochs):
        total, loss_r, loss_f, loss_b = loss_nitsche_multi_net(
            nets, xr_2d, xb_2d, xf_list_2d, pde, gamma_b, gamma_f_list
        )
        for opt in opts:
            opt.zero_grad()
        total.backward()
        for opt in opts:
            opt.step()

        # if ep % 500 == 0:
        #     for net in nets:
        #         net.eval()
        #     with torch.no_grad():
        #         u_pred = pde.predict_solution(nets, x_t)
        #         rel = l2_error(u_exact, u_pred)
        #     for net in nets:
        #         net.train()
            # print(
            #     f"{print_prefix}[Iter {ep}] loss = {total.item():.3e}, "
            #     f"l2_error = {rel:.3e}, interior = {loss_r.item():.3e}, "
            #     f"bc = {loss_b.item():.3e}, interface = {loss_f.item():.3e}"
            # )

    with torch.no_grad():
        x_t = pde.interior_points(1200).to(DEVICE, DTYPE)
        u = pde.exact_judge(x_t)
        uh = pde.predict_solution(nets, x_t).reshape(-1, 1)
        err_l2 = l2_error(u, uh)
    return err_l2


# ---------------- Adaptive driver ----------------
def train_adaptive(
    pde: multi_interface_flower_gpt,
    epochs_per_level: int = 500,
    adapt_levels: int = 6,
    grid_h_start: int = 15,
    Nf_list_start=[64,64,64],
    local_grid_factor: int = 3,
    n_insert_per_edge: int = 1,
    gamma_b: float = 3000.0,
    gamma_f_list=(700.0, 250.0, 500.0),
    width: int = 32,
    depth: int = 5,
    d: int = 2,
    track_best: bool = True,
):
    setup_seed(888888)

    # Nets/opts
    nets = [FCNet(d, width, 1, depth=depth).to(DEVICE, DTYPE) for _ in range(4)]
    opts = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in nets]

    best_err = torch.tensor(float("inf"), dtype=torch.float32, device=DEVICE)
    best_level = 0
    best_states = None

    # Initial sampling
    xr, xf_list, xb = initial_data_multisubdomain(grid_h_start, Nf_list_start)
    # Ensure device/dtype
    xr = xr.to(DEVICE, DTYPE)
    xb = xb.to(DEVICE, DTYPE)
    xf_list = [xf.to(DEVICE, DTYPE) for xf in xf_list]
    err_l2 = None
    df_rows = []
    level = 0
    while level <= adapt_levels:
        # record sizes BEFORE refinement (so level 0 uses initial sizes)
        nr_before = int(xr.size(0))
        nb_before = int(xb.size(0))
        nf_list_before = [int(xf_k.size(0)) for xf_k in xf_list]

        # Train this level
        err_l2 = train_single_level(
            epochs_per_level,
            gamma_b,
            gamma_f_list,
            xr[:, :2],
            [xfi[:, :2] for xfi in xf_list],
            xb[:, :2],
            nets,
            opts,
            pde,
            print_prefix=f"[Level {level}] ",
        )
        print(f"At [Level {level}] Relative L2 error: {err_l2:.4e}")

        if track_best and (err_l2 < best_err):
            best_err = err_l2.detach()
            best_level = level
            # snapshot states
            best_states = [ {k: v.detach().clone() for k, v in net.state_dict().items()} for net in nets ]

        # --- Collect per-level stats for DataFrame (BEFORE refinement) ---
        df_rows.append({
            "Refine Level": int(level),
            "l2 error": float(err_l2.item()),
            "Num of (xr,xf,xb)": [nr_before, nf_list_before, nb_before],
        })

        # --- Refinement ---
        # 1) Add local grids around current interior+boundary points
        grid_add = add_grid(torch.cat([xr, xb], dim=0), local_grid_factor, grid_h_start)

        # 2) Classify new points via φ-tolerance
        xr_add, xf_add_list, xb_add = classify_x_all_multi(grid_add, pde)

        # 3) Densify each interface around existing xf
        xf_densified = []
        for iface, xfi in enumerate(xf_list):
            xf_dense = refine_interface_points(xfi[:, :2], iface, n_insert_per_edge)
            xf_densified.append(xf_dense)

        # 4) Build indicator over candidates (interior + densified interfaces + boundary)
        xf_densified_2d = [xfi[:, :2] for xfi in xf_densified]
        scores = indicator_grad_sqrt_beta(nets, xr_add[:, :2], xf_densified_2d, xb_add[:, :2], pde)

        # 5) Select top 10% by indicator
        pool_all = torch.cat([xr_add, xb_add] + xf_densified, dim=0)
        k = max(1, int(0.10 * len(scores)))
        vals, idx = torch.topk(scores, k)
        x_top = pool_all[idx]

        # visualize refinement candidates at this level
        plot_refinement_points(level, xr, xb, xf_list, candidates=x_top, save=True, prefix='refine_level')
        # Dedup by rounding to 1e-4 grid
        key = (x_top[:, :2] * 1e4).round().to(torch.int64)
        uq, _ = torch.unique(key, dim=0, return_inverse=True)
        uniq2 = uq.to(DTYPE) / 1e4
        x_top = torch.cat([uniq2, torch.zeros(uniq2.size(0), 1, dtype=DTYPE, device=DEVICE)], dim=1)

        # 6) Reclassify selected points and merge into datasets
        xr_top, xf_top_list, xb_top = classify_x_all_multi(x_top, pde)
        xr = torch.cat([xr, xr_top], dim=0)
        xb = torch.cat([xb, xb_top], dim=0)
        xf_list = [torch.cat([xf_list[j], xf_top_list[j]], dim=0) for j in range(len(xf_list))]

        print(
            f"[Level {level}] sizes → xr={tuple(xr.shape)}, xb={tuple(xb.shape)}, "
            + ", ".join([f"xf{j}={tuple(xf_list[j].shape)}" for j in range(3)])
        )
        level += 1

    df = pd.DataFrame(df_rows)

    if track_best and best_states is not None:
        # restore best states so that returned nets correspond to the best level
        for net, state in zip(nets, best_states):
            net.load_state_dict(state, strict=True)
        return nets, best_err, best_level, df
    else:
        return nets, err_l2, max(0, adapt_levels), df


# \begin{itemize}
#       \item Initial setting: $h = 6$, $N_f = [48, 60, 72]$
#         \item Refinement setting: $re\_factor = 3$, $num\_insert = 2$
#       \item Penalty: $\gamma_b = 3000$, $\gamma_f = (700.0, 500.0, 700.0)$
#       \item Epochs: $re\_level = 5$, $epochs = 1000$
#       \item Neural networks structure: $m = 32$, $depth = 3$
# \end{itemize}
# 3000	(500.0, 500.0, 300.0)	3	2	6	[48, 60, 72]	6	8	500	32	5
gamma_b, gamma_f = 3000,	(500.0, 500.0, 300.0)
m, depth = 32,5

re_factor, num_insert = 3, 2
# h, Nf = 6, [48, 60, 72]
h, Nf = 6, [64, 64, 128]

re_level, epoch_level = 7, 500
# re_level, epoch_level = 1,1

hyper_output = f"""
        \\item Initial setting: $h = {h}$, $N_f = {Nf}$
        \\item Refinement setting: $re\\_factor = {re_factor}$, $num\\_insert = {num_insert}$
      \\item Penalty: $\\gamma_b = {gamma_b}$, $\\gamma_f = {gamma_f}$
      \\item Epochs: $re\\_level = {re_level}$, $epochs = {epoch_level}$
      \\item Neural networks structure: $m = {m}$, $depth = {depth}$

"""

print(hyper_output)

# ---------------- Script entry ----------------
if __name__ == "__main__":
    # PDE with positive k0
    print('run adaptation')
    pde = multi_interface_flower_gpt(5.5, 1.0)

    nets, err_l2, used_level, df = train_adaptive(
        pde=pde,
        gamma_b=gamma_b,  # was 3000
        gamma_f_list=gamma_f,  # was (700, 250, 500)

        local_grid_factor=re_factor,
        n_insert_per_edge=num_insert,

        grid_h_start=h,
        Nf_list_start=Nf,  # bias Γ0 a bit

        epochs_per_level=epoch_level,         # was 30000
        adapt_levels=re_level,                # was 1
        width=m,                      # was 32
        depth=depth,                       # was 5
        d=2,
    )
    print(f"[Best] level = {int(used_level)}, err_l2 = {float(err_l2):.4e}")

    # Print LaTeX table with per-level stats
    num_levels = len(df)

    for i, row in df.iterrows():
        nr, nf_list_val, nb = row['Num of (xr,xf,xb)']
        err_percent = row['l2 error'] * 100.0
        level = row['Refine Level']
        nf_str = f"({nf_list_val[0]}, {nf_list_val[1]}, {nf_list_val[2]})"
        if i == 0:
            print(fr"    \multirow{{{num_levels}}}{{*}}{{Adaptive}} & Level {level} & \multirow{{{num_levels}}}{{*}}{{FcNet}} & {nr} & {nf_str} & {nb} & {err_percent:.2f} \\")
        else:
            print(fr"    & Level {level} &  & {nr} & {nf_str} & {nb} & {err_percent:.2f} \\")
    # print(r"    \bottomrule")
    # print(r"  \end{tabular}")
    # print(r"  \caption{Adaptive refinement statistics (multi-subdomain, flower interfaces).}")
    # print(r"  \label{tab:multi_flower_adapt}")
    # print(r"\end{table}")

    # Visualize

    nitsche_visualize_2d_whole_domain_2figures(250, pde, nets, kind="Adaptive")
    plt.show()

    # abs_error_2d_separate_bar(200, pde, nets, title="Adaptive Absolute Error (4 Subdomains)")
    import matplotlib

    matplotlib.use('Qt5Agg')
    from multi_noadapt_train import abs_error_nitsche
    abs_error_adapt_multi_now = abs_error_only(200, pde, nets)
    abs_error_compare(abs_error_nitsche, abs_error_adapt_multi_now)

    plt.show()
