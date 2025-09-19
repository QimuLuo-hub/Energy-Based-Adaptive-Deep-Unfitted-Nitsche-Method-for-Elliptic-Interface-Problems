import torch
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib import rcParams
#
# matplotlib.use('Qt5Agg')

import numpy as np

rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels

rcParams['text.usetex'] = True


# matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端


def visualize_2d_sun(h, pde, net_plus, net_minus):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real_minus = torch.where(pde.phi(Z), torch.nan, pde.exact_minus(Z)).reshape(h, h)
    real_plus = torch.where(pde.phi(Z), pde.exact_plus(Z), torch.nan).reshape(h, h)
    pred_minus = torch.where(pde.phi(Z), torch.nan, net_minus(Z)).reshape(h, h)
    pred_plus = torch.where(pde.phi(Z), net_plus(Z), torch.nan).reshape(h, h)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)

    vmin = min(real.min().item(), pred.min().item())
    vmax = max(real.max().item(), pred.max().item())
    zmin = real.min().item()
    zmax = real.max().item()

    # Plot Real Surface
    fig_real = plt.figure(figsize=(8, 6))
    ax_real = fig_real.add_subplot(111, projection='3d')

    surf1 = ax_real.plot_surface(X, Y, real_plus, cmap='jet', vmin=vmin, vmax=vmax, alpha=0.78)  # sun:0.76
    ax_real.plot_surface(X, Y, real_minus, cmap='jet', vmin=vmin, vmax=vmax, alpha=1)

    ax_real.set_zlim(zmin, zmax)
    ax_real.view_init(elev=39, azim=-86)  # for sun

    ax_real.locator_params(axis='x', nbins=5)
    ax_real.locator_params(axis='y', nbins=5)
    ax_real.locator_params(axis='z', nbins=5)

    # ax_real.set_title('Real')
    ax_real.set_xlabel('X')
    ax_real.set_ylabel('Y')
    ax_real.set_zlabel('Z')

    ax_real.grid(False)
    for axis in (ax_real.xaxis, ax_real.yaxis, ax_real.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    fig_real.colorbar(surf1, ax=ax_real, shrink=0.5, aspect=5)
    fig_real.tight_layout()
    fig_real.savefig('exact_sun.png', dpi=600)

    # Plot Predicted Surface
    fig_pred = plt.figure(figsize=(8, 6))
    ax_pred = fig_pred.add_subplot(111, projection='3d')

    surf2 = ax_pred.plot_surface(X, Y, pred_plus.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax,
                                 alpha=0.78)  # sun:0.76
    ax_pred.plot_surface(X, Y, pred_minus.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax, alpha=1)

    ax_pred.set_zlim(zmin, zmax)  # 设置统一的 Z 轴范围
    ax_pred.view_init(elev=39, azim=-86)  # for sun

    ax_pred.locator_params(axis='x', nbins=5)
    ax_pred.locator_params(axis='y', nbins=5)
    ax_pred.locator_params(axis='z', nbins=5)

    # ax_pred.set_title(f'Predicted of level {i}')
    ax_pred.set_xlabel('X')
    ax_pred.set_ylabel('Y')
    ax_pred.set_zlabel('Z')

    ax_pred.grid(False)
    for axis in (ax_pred.xaxis, ax_pred.yaxis, ax_pred.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass
    fig_pred.colorbar(surf2, ax=ax_pred, shrink=0.5, aspect=5)
    fig_pred.tight_layout()
    fig_pred.savefig('adapt_sun.png', dpi=600)
    plt.show()


def visualize_2d_flower(h, pde, net_plus, net_minus):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real_minus = torch.where(pde.phi(Z), torch.nan, pde.exact_minus(Z)).reshape(h, h)
    real_plus = torch.where(pde.phi(Z), pde.exact_plus(Z), torch.nan).reshape(h, h)
    pred_minus = torch.where(pde.phi(Z), torch.nan, net_minus(Z)).reshape(h, h)
    pred_plus = torch.where(pde.phi(Z), net_plus(Z), torch.nan).reshape(h, h)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)

    vmin = min(real.min().item(), pred.min().item())
    vmax = max(real.max().item(), pred.max().item())
    zmin = real.min().item()
    zmax = real.max().item()

    # Plot Real Surface
    fig_real = plt.figure(figsize=(8, 6))
    ax_real = fig_real.add_subplot(111, projection='3d')
    surf1 = ax_real.plot_surface(X, Y, real_plus, cmap='jet', vmin=vmin, vmax=vmax, alpha=1)
    ax_real.plot_surface(X, Y, real_minus, cmap='jet', vmin=vmin, vmax=vmax, alpha=1)

    ax_real.set_zlim(zmin, zmax)

    ax_real.locator_params(axis='x', nbins=5)
    ax_real.locator_params(axis='y', nbins=5)
    ax_real.locator_params(axis='z', nbins=5)

    # ax_real.set_title('Real')
    ax_real.set_xlabel('X')
    ax_real.set_ylabel('Y')
    ax_real.set_zlabel('Z')

    ax_real.grid(False)
    for axis in (ax_real.xaxis, ax_real.yaxis, ax_real.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    fig_real.colorbar(surf1, ax=ax_real, shrink=0.5, aspect=5)
    fig_real.tight_layout()
    fig_real.savefig('exact_flower.png', dpi=300)

    # Plot Predicted Surface
    fig_pred = plt.figure(figsize=(8, 6))
    ax_pred = fig_pred.add_subplot(111, projection='3d')
    surf2 = ax_pred.plot_surface(X, Y, pred_plus.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax, alpha=1)
    ax_pred.plot_surface(X, Y, pred_minus.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax, alpha=1)

    ax_pred.set_zlim(zmin, zmax)  # 设置统一的 Z 轴范围

    ax_pred.locator_params(axis='x', nbins=5)
    ax_pred.locator_params(axis='y', nbins=5)
    ax_pred.locator_params(axis='z', nbins=5)

    # ax_pred.set_title(f'Predicted of level {i}')
    ax_pred.set_xlabel('X')
    ax_pred.set_ylabel('Y')
    ax_pred.set_zlabel('Z')

    ax_pred.grid(False)
    for axis in (ax_pred.xaxis, ax_pred.yaxis, ax_pred.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass
    fig_pred.colorbar(surf2, ax=ax_pred, shrink=0.5, aspect=5)
    fig_pred.tight_layout()
    fig_pred.savefig('adapt_flower.png', dpi=300)
    plt.show()

def visualize_2d_circle(h, pde, net_plus, net_minus):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real_minus = torch.where(pde.phi(Z), torch.nan, pde.exact_minus(Z)).reshape(h, h)
    real_plus = torch.where(pde.phi(Z), pde.exact_plus(Z), torch.nan).reshape(h, h)
    pred_minus = torch.where(pde.phi(Z), torch.nan, net_minus(Z)).reshape(h, h)
    pred_plus = torch.where(pde.phi(Z), net_plus(Z), torch.nan).reshape(h, h)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)

    vmin = min(real.min().item(), pred.min().item())
    vmax = max(real.max().item(), pred.max().item())
    zmin = real.min().item()
    zmax = real.max().item()

    # Plot Real Surface
    fig_real = plt.figure(figsize=(8, 6))
    ax_real = fig_real.add_subplot(111, projection='3d')
    surf1 = ax_real.plot_surface(X, Y, real_plus, cmap='jet', vmin=vmin, vmax=vmax, alpha=1)
    ax_real.plot_surface(X, Y, real_minus, cmap='jet', vmin=vmin, vmax=vmax, alpha=1)

    ax_real.set_zlim(zmin, zmax)

    ax_real.locator_params(axis='x', nbins=5)
    ax_real.locator_params(axis='y', nbins=5)
    ax_real.locator_params(axis='z', nbins=5)

    # ax_real.set_title('Real')
    ax_real.set_xlabel('X')
    ax_real.set_ylabel('Y')
    ax_real.set_zlabel('Z')
    fig_real.colorbar(surf1, ax=ax_real, shrink=0.5, aspect=5)
    fig_real.tight_layout()
    fig_real.savefig('exact_circle.png', dpi=1200)

    # Plot Predicted Surface
    fig_pred = plt.figure(figsize=(8, 6))
    ax_pred = fig_pred.add_subplot(111, projection='3d')
    surf2 = ax_pred.plot_surface(X, Y, pred_plus.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax, alpha=1)
    ax_pred.plot_surface(X, Y, pred_minus.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax, alpha=1)

    ax_pred.set_zlim(zmin, zmax)  # 设置统一的 Z 轴范围

    ax_pred.locator_params(axis='x', nbins=5)
    ax_pred.locator_params(axis='y', nbins=5)
    ax_pred.locator_params(axis='z', nbins=5)

    # ax_pred.set_title(f'Predicted of level {i}')
    ax_pred.set_xlabel('X')
    ax_pred.set_ylabel('Y')
    ax_pred.set_zlabel('Z')
    fig_pred.colorbar(surf2, ax=ax_pred, shrink=0.5, aspect=5)
    fig_pred.tight_layout()
    fig_pred.savefig('adapt_circle.png', dpi=1200)

    plt.show()


def visualize_2d_circle_full(h, pde, net_plus, net_minus):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)

    vmin = min(real.min().item(), pred.min().item())
    vmax = max(real.max().item(), pred.max().item())
    zmin = real.min().item()
    zmax = real.max().item()

    # Plot Combined Exact Surface
    fig_real_full = plt.figure(figsize=(8, 6))
    ax_real_full = fig_real_full.add_subplot(111, projection='3d')
    surf_real = ax_real_full.plot_surface(X, Y, real.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax)
    ax_real_full.set_zlim(zmin, zmax)
    ax_real_full.set_xlabel('X')
    ax_real_full.set_ylabel('Y')
    ax_real_full.set_zlabel('Z')

    ax_real_full.locator_params(axis='x', nbins=5)
    ax_real_full.locator_params(axis='y', nbins=5)
    ax_real_full.locator_params(axis='z', nbins=5)

    ax_real_full.grid(False)
    for axis in (ax_real_full.xaxis, ax_real_full.yaxis, ax_real_full.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    fig_real_full.colorbar(surf_real, ax=ax_real_full, shrink=0.5, aspect=5)
    ax_real_full.view_init(elev=26, azim=-160)

    fig_real_full.tight_layout()
    fig_real_full.savefig('exact_full.png', dpi=300)

    # Plot Combined Predicted Surface
    fig_pred_full = plt.figure(figsize=(8, 6))
    ax_pred_full = fig_pred_full.add_subplot(111, projection='3d')
    surf_pred = ax_pred_full.plot_surface(X, Y, pred.detach().numpy(), cmap='jet', vmin=vmin, vmax=vmax)
    ax_pred_full.set_zlim(zmin, zmax)
    ax_pred_full.set_xlabel('X')
    ax_pred_full.set_ylabel('Y')
    ax_pred_full.set_zlabel('Z')

    ax_pred_full.locator_params(axis='x', nbins=5)
    ax_pred_full.locator_params(axis='y', nbins=5)
    ax_pred_full.locator_params(axis='z', nbins=5)

    ax_pred_full.grid(False)
    for axis in (ax_pred_full.xaxis, ax_pred_full.yaxis, ax_pred_full.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            axis._axinfo["pane"]["fill"] = False
            axis._axinfo["pane"]["edgecolor"] = (1, 1, 1, 0)
            axis._axinfo["grid"]["linewidth"] = 0
        except Exception:
            pass

    fig_pred_full.colorbar(surf_pred, ax=ax_pred_full, shrink=0.5, aspect=5)
    ax_pred_full.view_init(elev=26, azim=-160)

    fig_pred_full.tight_layout()
    fig_pred_full.savefig('adapt_full_circle.png', dpi=300)
    plt.show()


'''
# common color bar, 1 figure'''
def abs_error_2d(h, pde, net_plus, net_minus, abs_error_nitsche):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)
    abs_error_adapt = torch.abs(real - pred)

    #random
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    heatmap1 = ax1.imshow(abs_error_nitsche.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_title('Nitsche Method Error')

    heatmap2 = ax2.imshow(abs_error_adapt.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    # ax2.set_title('Adaptive Method Error')

    min_val = min(abs_error_nitsche.detach().numpy().min(), abs_error_adapt.detach().numpy().min())
    max_val = max(abs_error_nitsche.detach().numpy().max(), abs_error_adapt.detach().numpy().max())

    cbar = fig.colorbar(heatmap1, ax=[ax1, ax2])
    heatmap2.set_cmap(heatmap1.get_cmap())
    heatmap2.set_clim(heatmap1.get_clim())

    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    plt.show()
    # return abs_error

def abs_error_only(h, pde, net_plus, net_minus):
    # Set up the grid
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    # Predict
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)
    real = pde.exact_judge(Z).reshape(h, h)
    abs_error = torch.abs(real - pred)
    return abs_error


'''# separate colour bar, 2 figure'''
def abs_error_2d_v2(h, pde, net_plus, net_minus, abs_error_nitsche, file_random, file_adapt):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)
    abs_error_adapt = torch.abs(real - pred)

    # 计算颜色条的范围
    min_val = min(abs_error_nitsche.detach().numpy().min(), abs_error_adapt.detach().numpy().min())
    max_val = max(abs_error_nitsche.detach().numpy().max(), abs_error_adapt.detach().numpy().max())

    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)

    # 保存 Nitsche 方法误差图
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    heatmap1 = ax1.imshow(abs_error_nitsche.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_title('Nitsche Method Error')
    heatmap1.set_clim(min_val, max_val)

    # 创建与图片一样高的颜色条
    cbar_ax1 = fig1.add_axes([0.95, 0.11, 0.03, 0.77])  # [左, 向上➕, 宽, 拉长+]
    cbar1 = fig1.colorbar(heatmap1, cax=cbar_ax1)
    cbar1.set_ticks(tick_values)
    cbar1.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar1.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar1.ax.transAxes, fontsize=10, verticalalignment='top',
                  horizontalalignment='left')
    plt.savefig(f'{file_random}', bbox_inches='tight', dpi = 1200)
    plt.close(fig1)

    # 保存自适应方法误差图
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    heatmap2 = ax2.imshow(abs_error_adapt.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    # ax2.set_title('Adaptive Method Error')
    heatmap2.set_clim(min_val, max_val)

    # 创建与图片一样高的颜色条
    cbar_ax2 = fig2.add_axes([0.95, 0.11, 0.03, 0.77])  # [左, 下, 宽, 高]
    cbar2 = fig2.colorbar(heatmap2, cax=cbar_ax2)
    cbar2.set_ticks(tick_values)
    cbar2.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar2.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar2.ax.transAxes, fontsize=10, verticalalignment='top',
                  horizontalalignment='left')
    plt.savefig(f'{file_adapt}', bbox_inches='tight', dpi=1200)
    plt.close(fig2)

'''#common bar but 2 figure
'''
def abs_error_2d_v3(h, pde, net_plus, net_minus, abs_error_nitsche):

    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)
    abs_error_adapt = torch.abs(real - pred)

    min_val = min(abs_error_nitsche.detach().numpy().min(), abs_error_adapt.detach().numpy().min())
    max_val = max(abs_error_nitsche.detach().numpy().max(), abs_error_adapt.detach().numpy().max())

    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)

    # save Nitsche
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap = ax.imshow(abs_error_nitsche.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                        origin='lower', vmin=min_val, vmax=max_val)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title('Nitsche Method Error')

    cbar_ax = fig.add_axes([0.95, 0.11, 0.03, 0.77])  # [左, 下, 宽, 高]
    cbar2 = fig.colorbar(heatmap, cax=cbar_ax)
    cbar2.set_ticks(tick_values)
    cbar2.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar2.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar2.ax.transAxes, fontsize=10, verticalalignment='top',
                  horizontalalignment='left')

    plt.savefig("nitsche_error.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

    #save Adaptive 方法的图（不包含颜色条）
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(abs_error_adapt.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
              origin='lower', vmin=min_val, vmax=max_val)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title('Adaptive Method Error')
    plt.savefig("adaptive_error.png", bbox_inches='tight',dpi=300)
    plt.close(fig)

    print("Saved Nitsche error plot as 'nitsche_error.png'")
    print("Saved Adaptive error plot as 'adaptive_error.png'")


def abs_error_compare(abs_nitsche, abs_adapt):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    heatmap1 = ax1.imshow(abs_nitsche.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_title('Nitsche Method Error')

    heatmap2 = ax2.imshow(abs_adapt.detach().numpy(), cmap='jet', extent=[box[0], box[1], box[2], box[3]],
                          origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    # ax2.set_title('Adaptive Method Error')

    min_val = min(abs_nitsche.detach().numpy().min(), abs_adapt.detach().numpy().min())
    max_val = max(abs_nitsche.detach().numpy().max(), abs_adapt.detach().numpy().max())

    #flower
    cbar = fig.colorbar(heatmap2, ax=[ax1, ax2])
    heatmap1.set_cmap(heatmap2.get_cmap())
    heatmap1.set_clim(heatmap2.get_clim())

    #sun
    # cbar = fig.colorbar(heatmap1, ax=[ax1, ax2])
    # heatmap2.set_cmap(heatmap1.get_cmap())
    # heatmap2.set_clim(heatmap1.get_clim())
    # cbar = fig.colorbar(heatmap2, ax=[ax1, ax2])
    # heatmap1.set_cmap(heatmap2.get_cmap())
    # heatmap1.set_clim(heatmap2.get_clim())

    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar.ax.text(0.1, 1.04, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    plt.show()



def abs_error_compare_2025(abs_nitsche, abs_adapt):
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
    plt.show()

import pyvista as pv

def solution_3d(object, h_draw, num_surface, file_name, pde, net_plus, net_minus):
    box = torch.tensor([-1, 1, -1, 1, -1, 1], dtype=torch.float64)
    x1 = torch.linspace(box[0], box[1], h_draw)
    x2 = torch.linspace(box[2], box[3], h_draw)
    x3 = torch.linspace(box[4], box[5], h_draw)
    X, Y, Z = torch.meshgrid(x1, x2, x3, indexing='ij')
    points = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]), dim=1)
    x_indices = points[:, 0].numpy()
    mask = x_indices < 0

    # real
    plus_r, minus_r = pde.exact_plus(points), pde.exact_minus(points)
    plus_r[mask] = 0
    real_vals = torch.where(pde.phi(points), plus_r, minus_r).detach().numpy()
    # pred
    plus_p, minus_p = net_plus(points), net_minus(points)
    plus_p[mask] = 0
    pred_vals = torch.where(pde.phi(points), plus_p, minus_p).detach().numpy()

    vmin = float(min(real_vals.min(), pred_vals.min()))
    vmax = float(max(real_vals.max(), pred_vals.max()))

    sol_plus, sol_minus = None, None
    if object == 'real':
        sol_plus, sol_minus = pde.exact_plus(points), pde.exact_minus(points)
    elif object == 'pred':
        sol_plus, sol_minus = net_plus(points), net_minus(points)

    sol_plus[mask] = 0
    sol_clipped = torch.where(pde.phi(points), sol_plus, sol_minus)
    sol_clipped = sol_clipped.detach().numpy()

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    Z_np = Z.cpu().numpy()

    # 创建网格
    grid = pv.StructuredGrid(X_np, Y_np, Z_np)
    grid["values"] = sol_clipped.ravel(order="K")

    # resolution = (3840, 2160)
    resolution = (2500, 2000)
    plotter = pv.Plotter(off_screen=True, border=False)  # `border=False` 去掉边框

    contours = grid.contour(isosurfaces=num_surface)

    plotter.add_mesh(
        contours, opacity=0.95, cmap="jet", specular=0.5, specular_power=50, show_scalar_bar=False,
        clim=(vmin, vmax)
    )
    # plotter_real.add_scalar_bar(title="Real Solution", label_font_size=30, n_labels=5)

    # # Add scalar bar with LaTeX CMR font
    scalar_bar = plotter.add_scalar_bar(vertical=False, label_font_size=45, n_labels=5)
    scalar_bar.GetLabelTextProperty().SetFontFamilyToCourier()  # Use LaTeX CMR-like font
    scalar_bar.GetLabelTextProperty().SetFontFile("cmr10.ttf")
    scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
    scalar_bar.SetPosition(0.2, 0.1)

    # Add axes with LaTeX CMR font
    axes_actor = plotter.add_axes()
    axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToCourier()
    axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontFile("cmr10.ttf")
    axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)

    axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToCourier()
    axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontFile("cmr10.ttf")
    axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)

    axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToCourier()
    axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontFile("cmr10.ttf")
    axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)

    # # Add light
    light = pv.Light()
    light.position = (-1, 2, -3)
    light.focal_point = (0, 0, 0.08)
    light.intensity = 0.1
    plotter.add_light(light)

    # # 创建一个小球表示光源的位置
    # sphere = pv.Sphere(radius=0.1, center=light.position)
    # plotter_real.add_mesh(
    #     sphere, color="yellow", specular=1.0, smooth_shading=True, label="Light Source"
    # )

    plotter.render()
    plotter.set_background("white")  # 确保背景干净

    plotter.view_isometric()
    plotter.camera.zoom(1.2)
    plotter.camera_position = [(-1.0255720816450757, -2.5088134859528712, -7.320875692418873),
 (0.0, 0.0, 0.08169332146644592),
 (-0.9914538097211282, 0.05102518092499317, 0.12006570743152539)]
    plotter.show(screenshot = file_name, window_size=resolution, return_img=False)

    print('plotter.camera_position', plotter.camera_position)
    print(f"Saved {object} solution plot as {file_name} with resolution {resolution}")


def abs_error_only_3d(h, pde, net_plus, net_minus):
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

    abs_error = np.abs(real - pred)
    return abs_error




