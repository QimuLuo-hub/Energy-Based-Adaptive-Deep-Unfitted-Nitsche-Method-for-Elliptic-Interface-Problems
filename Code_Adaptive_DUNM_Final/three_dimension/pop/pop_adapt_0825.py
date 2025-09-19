# pop_adapt_beta_0818.py  (clean drop-in)
import time
from typing import Tuple, List

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from Fc_Net import FCNet
from indicator import *          # uses: indicator_grad_combined_beta_sqrt
from Loss_fun import *           # uses: loss_nitsche
from pde_pop_update import *        # provides popcorn_interface_gpt

# ---------------------------------------------------------
# Global setup / objects expected by external callers
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# interface radius baseline used by classify and sampling (kept for backward compatibility)
r0 = torch.tensor(0.6, device=device)

# expose pde & default perturb scale so external scripts (e.g., gpt_tune_pop_0825.py)
# can `from pop_adapt_beta_0818 import *` and use them directly
pde = popcorn_interface_gpt()
perturb_scale = 0.09


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def setup_seed(seed: int) -> None:
    """Deterministic CPU results for reproducibility."""
    torch.manual_seed(seed)


def l2_error(u: torch.Tensor, uh: torch.Tensor) -> torch.Tensor:
    """Relative L2 error over a discrete set of points."""
    num = torch.sqrt(torch.mean(torch.pow(u - uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)) + 1e-12)
    return num / den



# Uniform directions on the unit sphere (deterministic)
def fibonacci_sphere(n: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = (torch.pi * (1 + 5 ** 0.5)) * i
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)  # [n,3]

def interface_points(N_f):
    xf = pde.interface_points(N_f)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    # print('xf.shape', xf.shape)
    return xf

def interface_points_add(xf, num_new_points, perturb_scale):
    """Generate new points around the existing interface points using vectorized operations."""
    perturbations = torch.randn(len(xf) * num_new_points, 3, device=xf.device) * perturb_scale
    original_points = xf.repeat_interleave(num_new_points, dim=0)
    new_points = original_points[:, :3] + perturbations

    r_new = torch.norm(new_points, dim=1, keepdim=True)  # [N*num_new_points, 1]
    S_new = pde.S(new_points)

    # r0, A, sigma, factor, angles_first, angles_second = initialize_parameters()
    values_new = r_new - r0 - S_new
    mask = torch.abs(values_new).squeeze(1) < 0.1
    new_xf_candidates = new_points[mask]

    distances = torch.cdist(xf[:, :3], new_xf_candidates).float()
    overlap_mask = torch.any(distances < 1e-3, dim=0)
    final_new_xf = new_xf_candidates[~overlap_mask]
    final_new_xf = torch.cat((final_new_xf, torch.zeros(final_new_xf.size(0), 1, device=xf.device)), dim=-1)
    # print('final_new_xf.shape', final_new_xf.shape)
    return final_new_xf



def generate_grid_points(h: int) -> torch.Tensor:
    """Uniform 3D grid in [-1,1]^3 with an extra column (refinement count)."""
    x = torch.linspace(-1, 1, steps=h, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='xy')
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    return torch.cat((pts, torch.zeros(pts.size(0), 1, device=device)), dim=-1)


def add_grid(points: torch.Tensor, refinement_factor: int, h: int) -> torch.Tensor:
    """Refine given points by creating a small local offset grid around each point."""
    if points.numel() == 0:
        return points

    count = points[:, -1]
    # spacing halves with each refinement level
    initial_spacing = (1.0 / (h - 1)) / (2.0 ** count)

    offsets = torch.linspace(-1, 1, refinement_factor, device=points.device)
    dxs, dys, dzs = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    offset_grid = torch.stack([dxs, dys, dzs], dim=-1).reshape(-1, 3)

    expanded_offsets = (offset_grid[None, :, :] * initial_spacing[:, None, None]).reshape(-1, 3)
    expanded_points = points[:, :3].repeat_interleave(len(offset_grid), dim=0)
    refined_coords = expanded_points + expanded_offsets
    refined_count = (count + 1).repeat_interleave(len(offset_grid))

    refined_points = torch.cat((refined_coords, refined_count.unsqueeze(-1)), dim=-1)

    # Keep inside bounds
    refined_points = refined_points[(refined_points[:, :3].abs() <= 1).all(dim=1)]

    # Remove duplicates/overlaps with originals
    if points.size(0) > 0:
        distances = torch.cdist(refined_points[:, :3], points[:, :3])
        mask = distances.min(dim=1).values > 1e-6
        refined_points = refined_points[mask]

    refined_points = torch.unique(refined_points, dim=0).detach()
    refined_points.requires_grad_(True)
    return refined_points


def initial_data(h: int, Nf: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_points = generate_grid_points(h)
    boundary_mask = (torch.abs(grid_points[:, :3]).max(dim=1)[0] == 1.0)
    xb = grid_points[boundary_mask]
    xr = grid_points[~boundary_mask]
    xf = interface_points(Nf)

    for t in (xr, xf, xb):
        t.requires_grad_(True)
    return xr, xf, xb


def classify_x_all(x_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a mixed point cloud into (xr, xf, xb) by interface & boundary masks.
    Returns tensors with requires_grad=True and last column preserved.
    """
    if x_all.numel() == 0:
        empty = torch.empty((0, 4), device=device, dtype=torch.float32, requires_grad=True)
        return empty.clone(), empty.clone(), empty.clone()

    r = torch.norm(x_all[:, :3], dim=1, keepdim=True)
    S = pde.S(x_all[:, :3])
    interface_mask = (torch.abs(r - r0 - S) < 0.1).squeeze(1)

    boundary_mask = (torch.abs(x_all[:, :3]).max(dim=1)[0] == 1.0)
    combined_mask = interface_mask | boundary_mask
    interior_mask = ~combined_mask

    xb = x_all[boundary_mask]
    xf = x_all[interface_mask]
    xr = x_all[interior_mask]

    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_only1(
    epochs: int,
    gamma_b: float,
    gamma_f: float,
    xr: torch.Tensor,
    xf: torch.Tensor,
    xb: torch.Tensor,
    net_plus: torch.nn.Module,
    net_minus: torch.nn.Module,
    opt1: torch.optim.Optimizer,
    opt2: torch.optim.Optimizer,
    level_index: int,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]:
    """One stage of training over the current sampling sets."""
    net_plus.train(); net_minus.train()
    for _ in range(epochs):
        loss_energy, loss_interior, loss_interface, loss_bc = \
            loss_nitsche(net_plus, net_minus, xr[:, :3], xf[:, :3], xb[:, :3], pde, gamma_b, gamma_f)

        opt1.zero_grad(); opt2.zero_grad()
        loss_energy.backward()
        opt1.step(); opt2.step()

    # quick validation
    with torch.no_grad():
        num_test = 1200
        x_t = pde.interior_points(num_test).to(device)

        # x_r = pde.interior_points(num_test)
        # x_b = pde.boundary_points(num_test)
        # x_f = pde.interface_points(num_test)
        # x_t = torch.cat((x_r, x_f, x_b), dim=0)

        u = pde.exact_judge(x_t).to(device)
        uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
        err_l2 = l2_error(u, uh)
    print(f"at level {level_index}, Relative l2 error of u_pred (average over {num_test} points): {err_l2}")
    return net_plus, net_minus, err_l2


def train_adapt_grid(
    h: int,
    Nf: int,
    re_level: int,
    re_factor: int,
    num_insert: int,
    local_perturb_scale: float,
    epochs: int,
    gamma_b: float,
    gamma_f: float,
    m: int,
    depth: int,
    d: int = 3,
    lr: float = 1e-3,
):
    """Adaptive training with flux-based indicator and local grid refinement.

    Returns:
        net_plus, net_minus, df (per-level stats), final err_l2
    """
    setup_seed(888888)
    net_plus = FCNet(d, m, 1, depth=depth).to(device)
    net_minus = FCNet(d, m, 1, depth=depth).to(device)

    opt1 = torch.optim.Adam(net_plus.parameters(), lr=lr)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=lr)

    xr, xf, xb = initial_data(h, Nf)
    print('Initial Data (xr, xf, xb)', xr.shape, xf.shape, xb.shape)

    records: List[dict] = []

    # Level 0..re_level (inclusive): train then refine (while logic)
    level = 0
    best_err = float("inf")
    best_level = -1
    while level <= re_level:
        net_plus, net_minus, err_l2 = train_only1(
            epochs, gamma_b, gamma_f, xr, xf, xb, net_plus, net_minus, opt1, opt2, level
        )
        current_err = float(err_l2.detach().cpu())
        records.append({
            'Refine Level': level,
            'l2 error': current_err,
            'Num of (xr,xf,xb)': [xr.size(0), xf.size(0), xb.size(0)]
        })
        if current_err < best_err:
            best_err = current_err
            best_level = level

        # No refinement after the last level
        if level == re_level:
            break

        # --- refinement stage ---
        grid_add = add_grid(torch.cat((xr, xb), dim=0), re_factor, h)
        xr_add, _, xb_add = classify_x_all(grid_add)
        xf_add = interface_points_add(xf, num_insert, local_perturb_scale)

        # Rank by indicator (higher => more informative)
        scores = indicator_grad_combined_beta_sqrt(net_plus, net_minus, xr_add[:, :3], xf_add[:, :3], xb_add[:, :3], pde)
        scores_list = scores.tolist()
        k = max(1, int(0.10 * len(scores_list)))

        x_all_add = torch.cat((xr_add, xf_add, xb_add), dim=0)
        sorted_idx = sorted(range(len(scores_list)), key=lambda k_: scores_list[k_], reverse=True)
        x_all_sorted = x_all_add[sorted_idx]
        x_top_ten = x_all_sorted[:k]
        x_top_ten, _ = torch.unique(x_top_ten, dim=0, return_inverse=True)

        xr_top, xf_top, xb_top = classify_x_all(x_top_ten)
        xr = torch.cat((xr, xr_top), dim=0)
        xf = torch.cat((xf, xf_top), dim=0)
        xb = torch.cat((xb, xb_top), dim=0)

        print(f'After refine L{level+1}: xr={xr.shape}, xf={xf.shape}, xb={xb.shape}')
        level += 1

    df = pd.DataFrame(records)
    df.attrs['best_l2_error'] = best_err
    df.attrs['best_level'] = best_level
    print(f"Lowest l2 error {best_err} occurred at level {best_level}")

    '''3d solution'''
    # solution_3d('real',250,300,'real_pop.png', pde,net_plus, net_minus)

    return net_plus, net_minus, df, torch.tensor(records[-1]['l2 error']), best_err, best_level


def abs_error_v2(h, abs_error_adapt, abs_error_nitsche_3d):
    box = torch.tensor([-1, 1, -1, 1, -1, 1], dtype=torch.float64)  # ellipsoidal

    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    x3 = torch.linspace(box[4], box[5], h)
    X, Y, Z = torch.meshgrid(x1, x2, x3, indexing='ij')

    points = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]), dim=1)

    real = pde.exact_judge(points)
    real = real.detach().numpy()
    real = real.reshape(h, h, h)

    pred = torch.where(pde.phi(points), net_plus(points), net_minus(points))
    pred = pred.detach().numpy()
    pred = pred.reshape(h, h, h)

    # abs_error_adapt = np.abs(real - pred)
    # abs_error_adapt = abs_error_adapt.reshape(h, h, h)
    # from DUNM.Module.visualize_2d_surface import abs_error_nitsche_3d

    # slice_idx = int (h // 2.3)
    # slice_idx = int (h // 2.2)
    slice_idx = int (h // 2.2)

    error_adapt_slice = abs_error_adapt[:, :, slice_idx]
    error_random_slice = abs_error_nitsche_3d[:, :, slice_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    vmin = np.nanmin([error_random_slice.min(), error_adapt_slice.min()])
    vmax = np.nanmax([error_random_slice.max(), error_adapt_slice.max()])

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    heatmap_random = ax1.imshow(error_random_slice, cmap='jet', extent=[box[0], box[1], box[2], box[3]], origin='lower', norm=norm)
    heatmap_adapt = ax2.imshow(error_adapt_slice, cmap='jet', extent=[box[0], box[1], box[2], box[3]], origin='lower', norm=norm)


    cbar = fig.colorbar(heatmap_random, ax=[ax1, ax2])  # 现在范围一致，不会空一段
    # 如果要 ×10^-2 的刻度显示：
    ticks = np.linspace(vmin, vmax, 10)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t * 100:.3f}' for t in ticks])
    cbar.ax.text(0.1, 1.06, r'$\times 10^{-2}$', transform=cbar.ax.transAxes)

    # num_ticks = 10
    # tick_values = np.linspace(vmin, vmax, num_ticks)
    # cbar.set_ticks(tick_values)
    # cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 10])
    # cbar.ax.text(0.1, 1.06, r'$\times 10^{-1}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
    #              horizontalalignment='left')
    plt.show()

#
# 8000	300	2	2	6	64	6	1000	32	5
if __name__ == '__main__':
    gamma_b, gamma_f = 8000, 300

    # gamma_b, gamma_f = 8000, 300
    m, depth = 32, 5

    re_level, epoch_level = 6, 1000
    re_factor, num_insert = 2, 2
    h, Nf = 6, 64

    latex_output = f"""
            \\item Initial setting: $h = {h}$, $N_f = {Nf}$
            \\item Refinement setting: $re\\_factor = {re_factor}$, $num\\_insert = {num_insert}$
          \\item Penalty: $\\gamma_b = {gamma_b}$, $\\gamma_f = {gamma_f}$
          \\item Epochs: $re\\_level = {re_level}$, $epochs = {epoch_level}$
          \\item Neural networks structure: $m = {m}$, $depth = {depth}$
    """

    print(latex_output)

    net_plus, net_minus, df, final_err, best_err, best_level = train_adapt_grid(
        h=h,
        Nf=Nf,
        re_level=re_level,
        re_factor=re_factor,
        num_insert=num_insert,
        local_perturb_scale=0.09,
        epochs=epoch_level,
        gamma_b=gamma_b,
        gamma_f=gamma_f,
        m=m,
        depth=depth,
        d=3,
        lr=1e-3
    )

    print("Final relative L2 error:", final_err.item())
    print(df)

    num_levels = len(df)

    for i, row in df.iterrows():
        nr, nf, nb = row["Num of (xr,xf,xb)"]
        err_percent = row["l2 error"] * 100
        level = row["Refine Level"]

        if i == 0:
            print(
                fr"    \multirow{{{num_levels}}}{{*}}{{Adaptive}} & Level {level} & \multirow{{{num_levels}}}{{*}}{{FcNet}} & {nr} & {nf} & {nb} & {err_percent:.2f} \\")
        else:
            print(fr"    & Level {level} &  & {nr} & {nf} & {nb} & {err_percent:.2f} \\")

    from visualize_all import *

    # solution_3d('pred',250,300,'pred_pop.png', pde,net_plus, net_minus)

    import matplotlib
    matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
    h_error = 50
    abs_adaptive_pop = abs_error_only_3d(h_error, pde, net_plus, net_minus)
    from train_nitsche_2025_pop import abs_error_pop_nitsche

    abs_error_v2(h_error, abs_adaptive_pop, abs_error_pop_nitsche)



