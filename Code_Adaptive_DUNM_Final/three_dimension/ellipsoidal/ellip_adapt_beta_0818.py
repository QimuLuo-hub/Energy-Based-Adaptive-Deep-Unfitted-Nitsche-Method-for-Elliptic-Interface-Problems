import pandas as pd

import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels

rcParams['text.usetex'] = True
# matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端

from Fc_Net import FCNet
from Loss_fun import *

from indicator import *
from visualize_all import *



def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
# setup_seed(888888)

def weights_init_normal(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight, 0, 0.5)
        torch.nn.init.constant_(m.bias, 0.0)

def l2_error(u, uh):
    num = torch.sqrt(torch.mean(torch.pow(u-uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)))
    error = num / den
    return error

#v1
def interface_points(N_f=32):
    a = torch.sqrt(torch.tensor(1.69 / 2))
    b = torch.sqrt(torch.tensor(1.69 / 3))
    c = torch.sqrt(torch.tensor(1.69 / 6))

    # indices = torch.arange(0, N_f+1, dtype=torch.float32)#points from [0,N_f]
    indices = torch.arange(0, N_f, dtype=torch.float32) + 0.5 #do not include the extreme points
    theta = torch.pi * (1 + 5 ** 0.5) * indices #otherwise: equatorial dense, polar sparse; lambda
    phi = torch.acos(2 * indices / N_f-1) #phi-z axis，map the indicies into 1 - 2 * indices / N_f[-1,1]; use z=rcos(phi)，unit r=1 #mu

    x = a * torch.cos(theta) * torch.sin(phi)
    y = b * torch.sin(theta) * torch.sin(phi)
    z = c * torch.cos(phi)
    xf = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    return xf

def interface_points_add(xf, n_insert):
    N_f = xf.shape[0]
    new_xf = []

    a = torch.sqrt(torch.tensor(1.69 / 2))
    b = torch.sqrt(torch.tensor(1.69 / 3))
    c = torch.sqrt(torch.tensor(1.69 / 6))

    for i in range(N_f):
        # i_coords (\theta, \phi)
        theta_start = torch.atan2(xf[i, 1], xf[i, 0]).item()  # the same as two_d
        r_i = torch.sqrt((xf[i, 0] / a) ** 2 + (xf[i, 1] / b) ** 2 + (xf[i, 2] / c) ** 2)
        phi_start = torch.acos(xf[i, 2] / (r_i * c)).item()  #z =r*cos(\phi)

        # (i+1)_coords (\theta_{i+1}, \phi_{i+1})
        theta_end = torch.atan2(xf[(i + 1) % N_f, 1], xf[(i + 1) % N_f, 0]).item()
        r_next = torch.sqrt((xf[(i + 1) % N_f, 0] / a) ** 2 + (xf[(i + 1) % N_f, 1] / b) ** 2 + (xf[(i + 1) % N_f, 2] / c) ** 2)
        phi_end = torch.acos(xf[(i + 1) % N_f, 2] / (r_next * c)).item()

        if theta_end < theta_start:
            theta_end += 2 * torch.pi
        # insert points between (\theta, \phi)
        theta_insert = torch.linspace(theta_start, theta_end, n_insert + 2, dtype=torch.float32)[:-1]#2 for start and end; [:-1] for drop theta_end
        phi_insert = torch.linspace(phi_start, phi_end, n_insert + 2, dtype=torch.float32)[:-1]

        #make sure the inserted points lie on the interface
        x_insert = a * torch.cos(theta_insert) * torch.sin(phi_insert)
        y_insert = b * torch.sin(theta_insert) * torch.sin(phi_insert)
        z_insert = c * torch.cos(phi_insert)

        # form new interface points
        new_xf.append(torch.stack([x_insert, y_insert, z_insert], dim=1))

    new_xf = torch.cat(new_xf, dim=0)
    distances = torch.cdist(xf[:, :3], new_xf).float()
    overlap_indices = torch.any(distances < 1e-3, dim=0)
    new_xf = new_xf[~overlap_indices]
    new_xf = torch.cat((new_xf, torch.zeros(new_xf.size(0), 1)), dim=-1)
    # print('new_xf.shape', new_xf.shape)
    return new_xf

def generate_grid_points(h):#default = 5
    x_coords = torch.linspace(-1, 1, steps=h)
    y_coords = torch.linspace(-1, 1, steps=h)
    z_coords = torch.linspace(-1, 1, steps=h)
    X, Y, Z = torch.meshgrid(x_coords, y_coords,z_coords,  indexing = 'xy')
    grid_points_3d = torch.stack([X, Y, Z], dim=-1)
    grid_points_3d = grid_points_3d.view(-1, 3)
    grid_points_4d = torch.cat((grid_points_3d, torch.zeros(grid_points_3d.size(0), 1)), dim=-1)
    return grid_points_4d

# def add_grid(points, refinement_factor, h):#pass in points-3d
#     refined_points = []
#     for i, point in enumerate(points):
#         count = point[-1].item()
#         initial_spacing = (1 / (h - 1)) / (2 ** count)
#         for dx in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
#             for dy in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
#                 for dz in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
#                     new_point = point[:4] + torch.tensor([dx, dy, dz,0], dtype=points.dtype)
#                     new_point[-1] = count + 1
#                     if not any(torch.allclose(new_point[:3], p[:3]) for p in points):
#                         refined_points.append(new_point)
#     refined_points = [p for p in refined_points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1 and -1 <= p[2] <= 1]
#     refined_points = torch.stack(refined_points)
#     refined_points, _ = torch.unique(refined_points, dim=0, return_inverse=True)
#     refined_points = refined_points.clone().detach().requires_grad_(True)
#     return refined_points

# def add_grid(points, refinement_factor, h):
#     refined_points = []
#     for point in points:
#         count = point[-1].item()
#         initial_spacing = (1 / (h - 1)) / (2 ** count)
#         offsets = torch.linspace(-initial_spacing, initial_spacing, refinement_factor)
#         offsets = offsets.clone().detach()
#         dxs, dys, dzs = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
#         offset_grid = torch.stack([dxs.ravel(), dys.ravel(), dzs.ravel()], dim=-1)
#         for offset in offset_grid:
#             new_point = point[:4] + torch.cat((offset, torch.tensor([0], dtype=points.dtype)))
#             new_point[-1] = count + 1
#             if not any(torch.allclose(new_point[:3], p[:3]) for p in points):
#                 refined_points.append(new_point)
#     # refined_points = [p for p in refined_points if all(abs(x) <= 1 for x in p[:3])]
#     refined_points = [p for p in refined_points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1 and -1 <= p[2] <= 1]
#     refined_points = torch.stack(refined_points)
#     refined_points = torch.unique(refined_points, dim=0)
#     refined_points = refined_points.clone().detach().requires_grad_(True)
#     return refined_points

def add_grid(points, refinement_factor, h):
    count = points[:, -1]
    initial_spacing = (1 / (h - 1)) / (2 ** count)
    # Create offset grid
    offsets = torch.linspace(-1, 1, refinement_factor, device=points.device)
    dxs, dys, dzs = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    offset_grid = torch.stack([dxs, dys, dzs], dim=-1).reshape(-1, 3)

    # Expand initial_spacing to match each offset, then calculate offsets for each point
    expanded_offsets = (offset_grid[None, :, :] * initial_spacing[:, None, None]).reshape(-1, 3)

    # Expand points and add calculated offset
    expanded_points = points[:, :3].repeat_interleave(len(offset_grid), dim=0)
    refined_coords = expanded_points + expanded_offsets

    # Increment count for each new refined point
    refined_count = (count + 1).repeat_interleave(len(offset_grid))

    # Combine refined coordinates and counts
    refined_points = torch.cat((refined_coords, refined_count.unsqueeze(-1)), dim=-1)

    # Filter points within bounds
    refined_points = refined_points[(refined_points[:, :3].abs() <= 1).all(dim=1)]

    # Remove points overlapping with original points
    distances = torch.cdist(refined_points[:, :3], points[:, :3])
    mask = (distances.min(dim=1).values > 1e-6)  # Remove new points that are very close to existing points
    refined_points = refined_points[mask]
    refined_points = torch.unique(refined_points, dim=0).detach().requires_grad_(True)
    return refined_points


def initial_data(h, Nf):
    grid_points = generate_grid_points(h)
    boundary_mask = (torch.abs(grid_points[:,:3]).max(dim=1)[0] == 1.0).bool() #only max tensor not indices
    xb = grid_points[boundary_mask]
    xr = grid_points[~boundary_mask]
    xf = interface_points(Nf)
    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb


def classify_x_all(x_all):
    x = x_all[:, 0]
    y = x_all[:, 1]
    z = x_all[:, 2]
    interface_value = 2 * x ** 2 + 3 * y ** 2 + 6 * z ** 2-1.69
    interface_value = interface_value.clone().detach()
    interface_mask = torch.isclose(interface_value, torch.tensor(0.0), atol=0.01)
    boundary_threshold = 1.0
    boundary_mask = (torch.abs(x_all[:,:3]).max(dim=1)[0] == boundary_threshold).bool() #only max tensor not indices
    combined_mask = interface_mask | boundary_mask
    interior_mask = ~combined_mask
    xb = x_all[boundary_mask]
    xf = x_all[interface_mask]
    xr = x_all[interior_mask]
    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb


def train_only(epochs, gamma_b, gamma_f, xr, xf, xb, net_plus, net_minus, opt1, opt2, i):
    loss_list = []
    l2_list = []
    x_t = pde.interior_points(10000)
    u_exact = pde.exact_judge(x_t)
    for ep in range(epochs):
        loss_energy, loss_interior, loss_interface, loss_bc =\
        loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)
        '''train pde network, record loss'''
        opt1.zero_grad()
        opt2.zero_grad()
        loss_energy.backward()
        opt1.step()
        opt2.step()
        # if i % 500 == 0:
        #     loss_list.append(loss_energy.item())
        #     u_pred = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
        #     relative_l2 = l2_error(u_exact, u_pred)
        #     l2_list.append(relative_l2.item())
        #     print(f'At iteration = {i}, pde_loss = {loss_energy}, l2_error={relative_l2}, interior_loss = {loss_interior},'
        #           f'bc loss = {loss_bc}, interface loss = {loss_interface}')
    num_test = 1200
    with torch.no_grad():
       x_t = pde.interior_points(num_test)
       u = pde.exact_judge(x_t)
       uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
       uh = uh.reshape(-1, 1)
       err_l2 = l2_error(u, uh)
    print(f'At level {i-1}, Relative l2 error of the u_pred (average over {num_test} points) = {err_l2}.\n')
    return net_plus, net_minus, err_l2


def train_adapt_grid(h, Nf, re_level, re_factor, num_insert, epochs,
                      gamma_b, gamma_f, m, depth, d=3, lr=1e-3):
    # net_plus = ResNet(d, m, 1, depth=depth)
    # net_minus = ResNet(d, m, 1, depth=depth)
    setup_seed(888888)
    net_plus = FCNet(d, m, 1, depth=depth)
    net_minus = FCNet(d, m, 1, depth=depth)

    opt1 = torch.optim.Adam(net_plus.parameters(), lr=lr)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=lr)

    xr, xf, xb = initial_data(h, Nf)
    df = []
    i = 0
    err_l2 = None
    err_l2_lower = 1e-2
    err_l2_upper = 1e-1
    print(f'at level 0, points: Nr={xr.size(0)}, Nf={xf.size(0)}, Nb={xb.size(0)}')
    while i <= re_level:
        i +=1
        old_point = torch.cat((xr, xf, xb), dim=0)
        prev_err_l2 = err_l2
        net_plus, net_minus, err_l2 = train_only(epochs, gamma_b, gamma_f, xr[:,:3], xf[:,:3], xb[:,:3], net_plus, net_minus, opt1, opt2, i)
        result = {'Refine Level': i - 1, 'l2 error': err_l2, 'Num of (xr,xf,xb)': [xr.size(0), xf.size(0), xb.size(0)]}
        df.append(result)

        # if err_l2.item() > err_l2_upper and i > 4:
        #     print(f"Bad Stop at level {i - 1}: err_l2={err_l2.item():.4e} > {err_l2_upper}")
        #     return net_plus, net_minus, pd.DataFrame(df), prev_err_l2
        # elif err_l2.item() < err_l2_lower:
        #     print(f"Good Stop at level {i}: err_l2={err_l2.item():.4e} < {err_l2_lower}")
        #     return net_plus, net_minus, pd.DataFrame(df), err_l2

        '''add and store more sampling points'''
        grid_add = add_grid(torch.cat((xr, xb), dim=0), re_factor, h)
        xr_add, _, xb_add = classify_x_all(grid_add)
        xf_add = interface_points_add(xf[:,:3], num_insert)#Nf*3

        # v3-rank grad
        start_time_grad = time.time()
        # res = indicator_grad(net_plus, net_minus, xr_add[:,:3], xf_add[:,:3], xb_add[:,:3], pde)#v1
        res = indicator_grad_combined_beta_sqrt(net_plus, net_minus, xr_add[:,:3], xf_add[:,:3], xb_add[:,:3], pde)#v1
        res_list = res.tolist()
        # num_top10 = int(0.10 * len(res_list))
        num_top10 = int(0.10 * len(res_list))

        sorted_indices = sorted(range(len(res_list)), key=lambda k: res_list[k], reverse=True)
        x_all_add = torch.cat((xr_add, xf_add, xb_add), dim=0)#N*3
        x_all_sorted_points = x_all_add[sorted_indices]
        x_top_ten = x_all_sorted_points[:num_top10]

        x_top_ten, _ = torch.unique(x_top_ten, dim=0, return_inverse=True)
        xr_top, xf_top, xb_top = classify_x_all(x_top_ten)#N*3 with the 3rd:+1
        xr = torch.cat((xr, xr_top), dim=0)
        xf = torch.cat((xf, xf_top), dim=0)
        xb = torch.cat((xb, xb_top), dim=0)
        add_point = x_top_ten
        #
        # print(f'xr_top.shape={xr_top.shape}, xf_top.shape={xf_top.shape}, xb_top.shape={xb_top.shape}')
        print(f'At level {i}, xr.shape={xr.shape}, xf.shape={xf.shape}, xb.shape={xb.shape}')
        # result = {'Refine Level': i, 'l2 error': err_l2.item(), 'Num of (xr,xf,xb)': [xr.size(0),xf.size(0), xb.size(0)]}
        # df.append(result)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.scatter(old_point[:, 0].detach().numpy(), old_point[:, 1].detach().numpy(), old_point[:, 2].detach().numpy(), c='r', label='Old')
        # ax.scatter(add_point[:, 0].detach().numpy(), add_point[:, 1].detach().numpy(), add_point[:, 2].detach().numpy(), c='b', label='Add')
        #
        # import matplotlib.ticker as ticker
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        # ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
        #
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # # ax.set_title(f"Top 10% Points (ranked by gradients) for level {i}")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'ellip_{i}.png', dpi=1200)
        # plt.show()
        # visualize_3d_volume(50, pde, net_plus, net_minus, i)

    '''3d solution'''
    # solution_3d('real',250,300,'real_ellip_beta.png', pde,net_plus, net_minus)
    # solution_3d('pred', 250, 300, 'pred_ellip_beta.png', pde, net_plus, net_minus)
    return net_plus, net_minus, pd.DataFrame(df), err_l2





'''ellipsoidal'''

#v1
# pde = ellipsoidal_interface()
from pde_ellip_update import *
pde = ellipsoidal_interface_gpt()


#fixed
# gamma_b, gamma_f = 8000, 250
# m, depth = 32, 5
# re_factor, num_insert = 2, 2
#
# # h, Nf = 6, 256
# h, Nf = 5, 64
# re_level, epoch_level = 8, 1500

gamma_b, gamma_f = 8000, 300

# gamma_b, gamma_f = 9000, 300
m, depth = 32, 3
re_factor, num_insert = 2, 2

# h, Nf = 6, 256
h, Nf = 5, 100
re_level, epoch_level = 3, 1500
# re_level, epoch_level = 1, 1
latex_output = f"""
        \\item Initial setting: $h = {h}$, $N_f = {Nf}$
        \\item Refinement setting: $re\\_factor = {re_factor}$, $num\\_insert = {num_insert}$
      \\item Penalty: $\\gamma_b = {gamma_b}$, $\\gamma_f = {gamma_f}$
      \\item Epochs: $re\\_level = {re_level}$, $epochs = {epoch_level}$
      \\item Neural networks structure: $m = {m}$, $depth = {depth}$

"""
print(latex_output)

net_plus, net_minus, df, err_l2 = train_adapt_grid(h, Nf, re_level, re_factor,num_insert, epoch_level,
                      gamma_b, gamma_f,m, depth)

num_levels = len(df)

for i, row in df.iterrows():
    nr, nf, nb = row["Num of (xr,xf,xb)"]
    err_percent = row["l2 error"] * 100
    level = row["Refine Level"]
    if i == 0:
        print(fr"    \multirow{{{num_levels}}}{{*}}{{Adaptive}} & Level {level} & \multirow{{{num_levels}}}{{*}}{{FcNet}} & {nr} & {nf} & {nb} & {err_percent:.2f} \\")
    else:
        print(fr"    & Level {level} &  & {nr} & {nf} & {nb} & {err_percent:.2f} \\")


def abs_error(h, abs_error_adapt, abs_error_nitsche_3d):
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

    slice_idx = h // 2
    error_adapt_slice = abs_error_adapt[:, :, slice_idx]
    error_random_slice = abs_error_nitsche_3d[:, :, slice_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    heatmap1 = ax1.imshow(error_random_slice, cmap='jet',extent=[box[0], box[1], box[2], box[3]], origin='lower')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_title('Random Error')

    # 绘制第一个热力图
    heatmap2 = ax2.imshow(error_adapt_slice, cmap='jet',extent=[box[0], box[1], box[2], box[3]], origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    # ax2.set_title('Adaptive Error')

    min_val = min(error_random_slice.min(), error_adapt_slice.min())
    max_val = max(error_random_slice.max(), error_adapt_slice.max())

    cbar = fig.colorbar(heatmap1, ax=[ax1, ax2])
    heatmap2.set_cmap(heatmap1.get_cmap())
    heatmap2.set_clim(heatmap1.get_clim())

    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar.ax.text(0.1, 1.06, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    plt.show()
    # plt.savefig('pop_error2.png', dpi=600)
#
# import matplotlib
matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
abs_adaptive_ellip = abs_error_only_3d(250, pde, net_plus, net_minus)

from train_nitsche_2025_ellip import abs_error_ellip_nitsche
abs_error(250, abs_adaptive_ellip, abs_error_ellip_nitsche)



