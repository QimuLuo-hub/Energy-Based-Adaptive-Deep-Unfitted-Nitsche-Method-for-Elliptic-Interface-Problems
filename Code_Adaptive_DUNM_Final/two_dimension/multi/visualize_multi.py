import torch
import numpy as np
import matplotlib.pyplot as plt


def nitsche_visualize_2d_whole_domain_2figures(h, pde, nets, kind='Nitsche',
                                               cmap='jet', save=True):
    """
    Plot exact vs predicted solution as two 3D surfaces over the whole domain.
    Each subdomain is masked so surfaces don't overlap; both figures share the
    same color scale (vmin/vmax).

    Args:
        h:      grid resolution per axis (int)
        pde:    PDE object (multi_interface_flower_v2)
        nets:   list of 4 trained nets
        kind:   title prefix / filename stem
        cmap:   matplotlib colormap name
        save:   if True, saves PDFs to disk
    """
    # --- Build mesh on CPU (float32) for plotting ---
    box = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
    x1 = torch.linspace(box[0], box[1], h, dtype=torch.float32)
    x2 = torch.linspace(box[2], box[3], h, dtype=torch.float32)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (N,2) CPU

    # --- Move to same device/dtype as nets for eval ---
    sample_param = next(nets[0].parameters())
    XY_t = XY.to(device=sample_param.device, dtype=sample_param.dtype)

    # --- Evaluate exact/pred and masks on the SAME tensor ---
    for net in nets:
        net.eval()
    with torch.no_grad():
        u_pred = pde.predict_solution(nets, XY_t).detach().cpu().reshape(h, h)
        u_exact = pde.exact_judge(XY_t).detach().cpu().reshape(h, h)
        m0, m1, m2, m3 = pde.subdomain_mask(XY_t)

    masks = [m0.reshape(h, h).cpu(), m1.reshape(h, h).cpu(),
             m2.reshape(h, h).cpu(), m3.reshape(h, h).cpu()]

    # --- Shared color scale ---
    zmin = float(min(u_pred.min().item(), u_exact.min().item()))
    zmax = float(max(u_pred.max().item(), u_exact.max().item()))
    Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()

    # ================= Exact =================
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    fig1.patch.set_facecolor('white'); ax1.set_facecolor('white')

    surf1 = None
    for mask in masks:
        Zi = torch.where(mask, u_exact, torch.tensor(float('nan'))).numpy()
        surf1 = ax1.plot_surface(Xn, Yn, Zi, cmap=cmap, vmin=zmin, vmax=zmax,
                                 linewidth=0, antialiased=False)

    # prettify 3D box
    ax1.grid(False)
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    # ax1.set_title(f'Exact {kind} Solution')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlim(zmin, zmax)
    ax1.locator_params(axis='x', nbins=5)
    ax1.locator_params(axis='y', nbins=5)
    ax1.locator_params(axis='z', nbins=5)
    ax1.view_init(elev=35, azim=149)
    if surf1 is not None:
        fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    fig1.tight_layout()
    if save:
        fig1.savefig(f'{kind}_multi_exact_split.pdf', dpi=300, transparent=False)

    # ================= Predicted =================
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = None
    for mask in masks:
        Zi = torch.where(mask, u_pred, torch.tensor(float('nan'))).numpy()
        surf2 = ax2.plot_surface(Xn, Yn, Zi, cmap=cmap, vmin=zmin, vmax=zmax,
                                 linewidth=0, antialiased=False)

    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlim(zmin, zmax)
    ax2.locator_params(axis='x', nbins=5)
    ax2.locator_params(axis='y', nbins=5)
    ax2.locator_params(axis='z', nbins=5)
    ax2.view_init(elev=35, azim=149)
    ax2.grid(False)
    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    if surf2 is not None:
        fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    fig2.tight_layout()
    if save:
        fig2.savefig(f'{kind}_multi_pred_split.pdf', dpi=300)

    # plt.show()


def abs_error_2d_separate_bar(h, pde, nets,
                              title='Nitsche Absolute Error (4 Subdomains)',
                              cmap='jet', save=True, outfile='abs_error_2d_whole.png'):
    """
    Plot absolute error |u_exact - u_pred| over the whole domain as a heatmap.
    The error is masked per subdomain (no overlap) and uses a single colorbar.

    Args:
        h:        grid resolution per axis (int)
        pde:      PDE object (multi_interface_flower_v2)
        nets:     list of 4 trained nets
        title:    plot title
        cmap:     matplotlib colormap name
        save:     if True, writes PNG to disk
        outfile:  output filename
    """
    # --- Grid on CPU ---
    box = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
    x1 = torch.linspace(box[0], box[1], h, dtype=torch.float32)
    x2 = torch.linspace(box[2], box[3], h, dtype=torch.float32)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # --- Predict on the nets' device; bring back to CPU ---
    sample_param = next(nets[0].parameters())
    XY_t = XY.to(device=sample_param.device, dtype=sample_param.dtype)
    for net in nets:
        net.eval()
    with torch.no_grad():
        pred = pde.predict_solution(nets, XY_t).detach().cpu().reshape(h, h)
        real = pde.exact_judge(XY_t).detach().cpu().reshape(h, h)

    abs_error = torch.abs(real - pred)

    # --- Masks computed on same XY_t for consistency ---
    m0, m1, m2, m3 = pde.subdomain_mask(XY_t)
    masks = [m0.reshape(h, h).cpu(), m1.reshape(h, h).cpu(),
             m2.reshape(h, h).cpu(), m3.reshape(h, h).cpu()]

    # --- Compose masked error (NaN outside subdomains) ---
    A = torch.full_like(abs_error, torch.nan)
    for m in masks:
        A = torch.where(m, abs_error, A)

    A_np = A.detach().cpu().numpy()
    vmin = float(np.nanmin(A_np))
    vmax = float(np.nanmax(A_np))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(A_np, cmap=cmap, extent=[box[0], box[1], box[2], box[3]],
                   origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.3e}' for t in ticks])

    fig.tight_layout()
    if save:
        fig.savefig(outfile, dpi=300)
    plt.show()


def abs_error_only(h, pde, nets):
    # Set up the grid
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    # Predict
    pred = pde.predict_solution(nets, Z).reshape(h, h)
    real = pde.exact_judge(Z).reshape(h, h)
    abs_error = torch.abs(real - pred)
    return abs_error


def abs_error_compare(abs_nitsche, abs_adapt):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    heatmap_nitsche = ax1.imshow(abs_nitsche.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_title('Nitsche Method Error')

    heatmap_adapt = ax2.imshow(abs_adapt.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    # ax2.set_title('Adaptive Method Error')

    min_val = min(abs_nitsche.detach().numpy().min(), abs_adapt.detach().numpy().min())
    max_val = max(abs_nitsche.detach().numpy().max(), abs_adapt.detach().numpy().max())

    #wrong
    # cbar = fig.colorbar(heatmap_adapt, ax=[ax1, ax2])
    # heatmap_nitsche.set_cmap(heatmap_adapt.get_cmap())
    # heatmap_nitsche.set_clim(heatmap_adapt.get_clim())

    cbar = fig.colorbar(heatmap_nitsche, ax=[ax1, ax2])
    heatmap_adapt.set_cmap(heatmap_nitsche.get_cmap())
    heatmap_adapt.set_clim(heatmap_nitsche.get_clim())



    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    # plt.show()
