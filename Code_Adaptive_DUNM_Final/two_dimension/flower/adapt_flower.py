import pandas as pd


from Fc_Net import FCNet
from indicator import *
from visualize_all import *

'''#jump2-1'''
from Loss_fun import *
#two_d


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
    theta_initial = torch.linspace(0, 2 * torch.pi, N_f, dtype=torch.float32)
    theta_initial = theta_initial.reshape(-1, 1)
    r = 1/2 + torch.sin(5*theta_initial)/7
    x_initial = r * torch.cos(theta_initial)
    y_initial = r * torch.sin(theta_initial)
    xf = torch.cat((x_initial, y_initial), dim=1)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    return xf

def interface_points_add(xf, n_insert=1):
    N_f = xf.shape[0]
    new_xf = []
    for i in range(N_f):
        theta_start = torch.atan2(xf[i, 1], xf[i, 0]).item()
        theta_end = torch.atan2(xf[(i + 1) % N_f, 1], xf[(i + 1) % N_f, 0]).item()
        if theta_end < theta_start:
            theta_end += 2 * torch.pi
        theta_insert = torch.linspace(theta_start, theta_end, n_insert + 2, dtype=torch.float32)[:-1]
        #flower shape
        r_insert = 1 / 2 + torch.sin(5 * theta_insert) / 7
        x_insert = r_insert * torch.cos(theta_insert)
        y_insert = r_insert * torch.sin(theta_insert)
        new_xf.append(torch.stack([x_insert, y_insert], dim=1))
    new_xf = torch.cat(new_xf, dim=0)
    distances = torch.cdist(xf, new_xf).float()
    overlap_indices = torch.any(distances < 1e-2, dim=0)
    new_xf = new_xf[~overlap_indices]
    new_xf = torch.cat((new_xf, torch.zeros(new_xf.size(0), 1)), dim=-1)
    return new_xf

def generate_grid_points(h):
    x_coords = torch.linspace(-1, 1, steps=h)
    y_coords = torch.linspace(-1, 1, steps=h)
    X, Y = torch.meshgrid(x_coords, y_coords,  indexing = 'xy')
    grid_points_2d = torch.stack([X, Y], dim=-1)
    grid_points_2d = grid_points_2d.view(-1, 2)
    grid_points_3d = torch.cat((grid_points_2d, torch.zeros(grid_points_2d.size(0), 1)), dim=-1)
    return grid_points_3d

def add_grid(points, refinement_factor, h):#pass in points-3d
    refined_points = []
    for i, point in enumerate(points):
        count = point[-1].item()
        initial_spacing = (1 / (h - 1)) / (2 ** count)
        for dx in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
            for dy in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
                new_point = point[:3] + torch.tensor([dx, dy,0], dtype=points.dtype)
                new_point[-1] = count + 1
                if not any(torch.allclose(new_point[:2], p[:2]) for p in points):                    refined_points.append(new_point)
    refined_points = [p for p in refined_points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1]
    refined_points = torch.stack(refined_points)
    refined_points, _ = torch.unique(refined_points, dim=0, return_inverse=True)
    refined_points = refined_points.clone().detach().requires_grad_(True)
    return refined_points

def initial_data(h=10, Nf=32):
    grid_points = generate_grid_points(h)
    boundary_mask = (torch.abs(grid_points[:,:2]).max(dim=1)[0] == 1.0).bool() #only max tensor not indices
    xb = grid_points[boundary_mask]
    xr = grid_points[~boundary_mask]
    xf = interface_points(Nf)
    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb

def classify_x_all(x_all):
    r = torch.norm(x_all[:, :2], dim=1)
    theta = torch.atan2(x_all[:, 1], x_all[:, 0])
    interface_r = 0.5 + torch.sin(5*theta)/7
    interface_mask = torch.isclose(r, interface_r, atol=0).bool()
    boundary_threshold = 1.0
    boundary_mask = (torch.abs(x_all[:,:2]).max(dim=1)[0] == boundary_threshold).bool() #only max tensor not indices
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
    # loss_list = []
    # l2_list = []
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
        #     # loss_list.append(loss_energy.item())
        #     u_pred = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
        #     relative_l2 = l2_error(u_exact, u_pred)
            # l2_list.append(relative_l2.item())
            # print(f'At iteration = {i}, pde_loss = {loss_energy}, l2_error={relative_l2}, interior_loss = {loss_interior},'
            #       f'bc loss = {loss_bc}, interface loss = {loss_interface}')

    num_test = 1200
    with torch.no_grad():
        x_t = pde.interior_points(num_test)
        u = pde.exact_judge(x_t)
        uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
        uh = uh.reshape(-1, 1)
        err_l2 = l2_error(u, uh)
    print(f'At level {i-1}, Relative l2 error of u_pred (average on {num_test} points) = {err_l2.item()}.\n')
    return net_plus, net_minus, err_l2


def train_adapt_grid( h, Nf, re_level, re_factor,num_insert, epochs,
                      gamma_b, gamma_f,
                      m, depth, d=2):
    # net_plus = ResNet(d, m, 1, depth=depth)
    # net_minus = ResNet(d, m, 1, depth=depth)
    setup_seed(888888)
    net_plus = FCNet(d, m, 1, depth=depth)
    net_minus = FCNet(d, m, 1, depth=depth)

    opt1 = torch.optim.Adam(net_plus.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=1e-3)

    xr, xf, xb = initial_data(h, Nf)
    # print('xr.shape, xf.shape', xr.shape, xf.shape)
    df = []
    i = 0
    err_l2 = None
    err_l2_lower = 1e-2
    err_l2_upper = 1e-1


    n_r, n_f, n_b = xr.size(0), xf.size(0), xb.size(0)
    print('Level 0', n_r, n_f, n_b)

    level_0_point = torch.cat((xr, xf, xb), dim=0)

    plt.figure()
    plt.scatter(level_0_point[:, 0].detach().numpy(), level_0_point[:, 1].detach().numpy(), color='red', label='Old')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'flower_0.png', dpi=300)
    plt.show()

    while i <= re_level:
        i +=1
        old_point = torch.cat((xr, xf, xb), dim=0)

        prev_err_l2 = err_l2 #for bad stop, we use the l2 of the previous level

        net_plus, net_minus, err_l2 = train_only(epochs, gamma_b, gamma_f, xr[:, :2], xf[:, :2], xb[:, :2], net_plus, net_minus, opt1, opt2, i)
        result = {'Refine Level': i-1, 'l2 error': err_l2, 'Num of (xr,xf,xb)': [xr.size(0), xf.size(0), xb.size(0)]}
        df.append(result)
        # if err_l2 is not None:
        # if err_l2.item() > err_l2_upper and i > 4:
        #     print(f"Bad Stop at level {i - 1}: err_l2={err_l2.item():.4e} > {err_l2_upper}")
        #     return net_plus, net_minus, pd.DataFrame(df), prev_err_l2
        #
        # elif err_l2.item() < err_l2_lower:
        #     print(f"Good Stop at level {i}: err_l2={err_l2.item():.4e} < {err_l2_lower}")
        #     # result = {'Refine Level': i, 'l2 error': err_l2.item(),
        #     #           'Num of (xr,xf,xb)': [n_r, n_f, n_b]}
        #     # df.append(result)
        #     import matplotlib
        #     matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
        #     abs_adapt = abs_error_only(300, pde, net_plus, net_minus)
        #     from train_nitsche_2025_2d import abs_nitsche_flower
        #     abs_error_compare_2025(abs_nitsche_flower, abs_adapt)
        #     return net_plus, net_minus, pd.DataFrame(df), err_l2


        '''add and store more sampling points'''
        grid_add = add_grid(torch.cat((xr, xb), dim=0), re_factor, h)
        xr_add, _, xb_add = classify_x_all(grid_add)
        xf_add = interface_points_add(xf[:, :2], num_insert)  # Nf*3

        # v3-rank grad
        res = indicator_grad_combined_beta_sqrt(net_plus, net_minus, xr_add[:, :2], xf_add[:, :2], xb_add[:, :2], pde)  # v1

        res_list = res.tolist()
        num_top10 = int(0.10 * len(res_list))
        sorted_indices = sorted(range(len(res_list)), key=lambda k: res_list[k], reverse=True)
        x_all_add = torch.cat((xr_add, xf_add, xb_add), dim=0)  # N*3
        x_all_sorted_points = x_all_add[sorted_indices]
        x_top_ten = x_all_sorted_points[:num_top10]

        x_top_ten, _ = torch.unique(x_top_ten, dim=0, return_inverse=True)
        distances = torch.cdist(x_top_ten[:, :2], x_top_ten[:, :2]).float()
        unique_points = []
        seen = set()
        threshold = 1e-3
        for a in range(x_top_ten.size(0)):
            if a not in seen:
                distances_to_a = distances[a, :]
                distances_to_a[a] = float('inf')  # 排除自身
                if torch.all(distances_to_a > threshold):
                    unique_points.append(x_top_ten[a])
                else:
                    unique_points.append(x_top_ten[a])
                    for j, distance in enumerate(distances_to_a):
                        if distance <= threshold:
                            seen.add(j)
        unique_points = torch.stack(unique_points)
        x_top_ten = unique_points
        # print('AFTER x_top_ten.shape', x_top_ten.shape)
        xr_top, xf_top, xb_top = classify_x_all(x_top_ten)#N*3 with the 3rd:+1
        # print(f'xr_top.shape={xr_top.shape}, xf_top.shape={xf_top.shape}, xb_top.shape={xb_top.shape}')
        xr = torch.cat((xr, xr_top), dim=0)
        xf = torch.cat((xf, xf_top), dim=0)
        xb = torch.cat((xb, xb_top), dim=0)
        print(f'At level {i}, xr.shape={xr.shape}, xf.shape={xf.shape}, xb.shape={xb.shape}')
        add_point = x_top_ten


        # result = {'Refine Level': i, 'l2 error': prev_err_l2, 'Num of (xr,xf,xb)': [xr.size(0),xf.size(0), xb.size(0)]}
        # df.append(result)

        #v1
        plt.figure()
        plt.scatter(old_point[:, 0].detach().numpy(), old_point[:, 1].detach().numpy(), color='red', label='Old')
        plt.scatter(add_point[:, 0].detach().numpy(), add_point[:, 1].detach().numpy(), color='blue', label='Add', s=20)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'flower_{i}.png', dpi=300)
        plt.title(f"Level {i}")
        plt.show()


    df = pd.DataFrame(df)
    # visualize_2d_flower(300, pde, net_plus, net_minus, i)

    '''abs error'''
    # import matplotlib
    # matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
    # abs_adapt = abs_error_only(300, pde, net_plus, net_minus)
    # from high2D_my_trainnoadapt_flower import abs_nitsche_flower
    # abs_error_compare_2025(abs_nitsche_flower, abs_adapt)

    return net_plus, net_minus, df, err_l2

'''flower'''

bp = 10
bm= 1

# from gpt_pde_flower_nobeta import *
# pde = flowerInterface(bp, bm)
from pde_flower_update import *
pde = flowerInterface(bp, bm)

re_level, epoch_level = 5, 1500
re_factor, num_insert = 3,2
h, Nf = 10, 32
gamma_b, gamma_f = 8000, 250
m, depth = 32, 5


# latex_output = f"""
#         \\item Initial setting: $h = {h}$, $N_f = {Nf}$
#         \\item Refinement setting: $re\\_factor = {re_factor}$, $num\\_insert = {num_insert}$
#       \\item Penalty: $\\gamma_b = {gamma_b}$, $\\gamma_f = {gamma_f}$
#       \\item Epochs: $re\\_level = {re_level}$, $epochs = {epoch_level}$
#       \\item Neural networks structure: $m = {m}$, $depth = {depth}$
# """
# print(latex_output)
#
#
net_plus, net_minus, df, err_l2 = train_adapt_grid(h, Nf, re_level, re_factor, num_insert, epoch_level, gamma_b, gamma_f, m, depth)
print(df)

# # visualize_2d_flower(300, pde, net_plus, net_minus)
num_levels = len(df)

for i, row in df.iterrows():
    nr, nf, nb = row["Num of (xr,xf,xb)"]
    err_percent = row["l2 error"] * 100
    level = row["Refine Level"]

    if i == 0:
        print(fr"    \multirow{{{num_levels}}}{{*}}{{Adaptive}} & Level {level} & \multirow{{{num_levels}}}{{*}}{{FcNet}} & {nr} & {nf} & {nb} & {err_percent:.2f} \\")
    else:
        print(fr"    & Level {level} &  & {nr} & {nf} & {nb} & {err_percent:.2f} \\")


# visualize_2d_flower(300, pde, net_plus, net_minus)
