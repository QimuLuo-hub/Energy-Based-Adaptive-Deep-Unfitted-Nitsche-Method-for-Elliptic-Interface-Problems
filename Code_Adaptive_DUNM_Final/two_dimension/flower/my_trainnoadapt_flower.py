import time
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Qt5Agg')  # 使用 Qt5Agg 后端
import matplotlib.pyplot as plt
from Fc_Net import FCNet

'''#jump1-2'''
# from my_loss import *
# from Interface_2d import *

'''#jump2-1'''
from Loss_fun import *

#2d




def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
setup_seed(888888)

def weights_init_normal(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight, 0, 0.5)
        torch.nn.init.constant_(m.bias, 0.0)

def l2_error(u, uh):
    num = torch.sqrt(torch.mean(torch.pow(u-uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)))
    error = num / den
    return error

batchstep = 10

def train(epochs, gamma_b, gamma_f, Nr, Nb, Nf, m, depth, d):
    loss_list = []
    l2_list = []
    # net_plus = ResNet(d, m, 1, depth=depth)
    # net_minus = ResNet(d, m, 1, depth=depth)

    net_plus = FCNet(d, m, 1, depth=depth)
    net_minus = FCNet(d, m, 1, depth=depth)

    # net_plus.apply(weights_init_normal)
    # net_minus.apply(weights_init_normal)
    x_t = pde.interior_points(10000)
    u_exact = pde.exact_judge(x_t)

    #generate train points
    xr = pde.interior_points(Nr)
    xf = pde.interface_points(Nf)
    xb = pde.boundary_points(Nb)

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()


    opt1 = torch.optim.Adam(net_plus.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=1e-3)
    for i in range(epochs):
    #     # if epochs % batchstep == 0:
    #     #     xr = pde.interior_points(Nr)
    #     #     xr.requires_grad_()
    #     #     xf = pde.interface_points(Nf)
    #     #     xf.requires_grad_()
    #     #     xb = pde.boundary_points(Nb)
    #     #     xb.requires_grad_()
    #
        loss_energy, loss_interior, loss_interface, loss_bc =\
        loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)
    #
    #     # loss_energy, loss_interior, loss_interface, loss_bc =\
    #     # loss_least_square(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)
    #
        '''train pde network, record loss'''
        opt1.zero_grad()
        opt2.zero_grad()
        loss_energy.backward()
        opt1.step()
        opt2.step()
    # opt1 = torch.optim.LBFGS(net_plus.parameters(), lr=1.0, max_iter=20, history_size=100,
    #                          line_search_fn="strong_wolfe")
    # opt2 = torch.optim.LBFGS(net_minus.parameters(), lr=1.0, max_iter=20, history_size=100,
    #                          line_search_fn="strong_wolfe")

    # def closure():
    #     opt1.zero_grad(set_to_none=True)
    #     opt2.zero_grad(set_to_none=True)
    #     loss_energy, loss_interior, loss_interface, loss_bc = \
    #         loss_nitsche(net_plus, net_minus, xr[:, :2], xf[:, :2], xb[:, :2], pde, gamma_b, gamma_f)
    #     loss_energy.backward()
    #     return loss_energy
    #
    # for ep in range(epochs):
    #     opt1.step(closure)
    #     opt2.step(closure)
    #
    #     if ep % 100 == 0:
    #         # loss_list.append(loss_energy.item())
    #         u_pred = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
    #         relative_l2 = l2_error(u_exact, u_pred)
    #         l2_list.append(relative_l2.item())
            # print(f'At iteration = {i}, pde_loss = {loss_energy}, l2_error={relative_l2}, interior_loss = {loss_interior},'
            #       f'bc loss = {loss_bc}, interface loss = {loss_interface}')
    # min_l2error = min(l2_list)
    # print('min_l2error', min_l2error)

    #PDE Loss
    # plt.figure(figsize=(8, 6))
    # plt.plot(loss_list, label='PDE Loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss (every 100 epochs)')
    # plt.title(f'{epochs} epochs, (bp, bm) = ({bp}, {bm}), {d}d: \n (gamma_b, f_parameter)=({gamma_b}, {gamma_f}), '
    #           f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}), (m, depth)=({m}, {depth})')
    # plt.legend()

    # if epochs > 5:
        # pde_path = (f'{save_path_ellipsoidal}pdeloss{d}d{epochs}iterations, (bp, bm) = ({bp}, {bm}), (gamma_b, gamma_f)=({gamma_b}, {gamma_f}),'
        #           f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}), (m, depth)=({m}, {depth}).pdf')
        # plt.savefig(pde_path)

    # L2
    # plt.figure(figsize=(8, 6))
    # plt.plot(l2_list, label='relative l2 error')
    # plt.xlabel('Iterations')
    # plt.ylabel('Relative l2 error (every 100 epochs)')
    # plt.yscale("log")
    # plt.title(f'{epochs}epochs, (bp, bm) = ({bp}, {bm}), {d}d: \n (gamma_b, f_parameter)=({gamma_b}, {gamma_f}),'
    #           f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}), (m, depth)=({m}, {depth})')
    # plt.legend()
    # plt.show()

    # if epochs > 5:
    #     l2_path = (f'{save_path_ellipsoidal}l2error{d}d{epochs}iterations, (bp, bm) = ({bp}, {bm}), (gamma_b, gamma_f)=({gamma_b}, {gamma_f}), '
    #               f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}),(m, depth)=({m}, {depth}).pdf')
    #     plt.savefig(l2_path)
    # plt.show()

    #Final l2 error
    num_test = 1200
    with torch.no_grad():
       x_t = pde.interior_points(num_test)
       u = pde.exact_judge(x_t)
       uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
       uh = uh.reshape(-1, 1)
       err_l2 = l2_error(u, uh)
    print(f'Relative l2 error of u_pred (average on {num_test} points)', err_l2)

    return net_plus, net_minus, err_l2




#here-circle- 4 angle
# r0 = 0.5
# # # # save_path_circle = f'Save_results/circle/bp={bp}, bm={bm}/'
# bm = 100000 #beta1
# bp = 1 #beta2
# pde = circle_interface(bp, bm, r0)
# # from circle_adapt_sqrtbeta import gamma_b, gamma_f, re_level, epoch_level
# # epochs = epoch_level * re_level
# # print('gamma_b, gamma_f, epochs', gamma_b, gamma_f, epochs)
# net_plus, net_minus, err_l2 = train(7500,
#                                          3000, 250,
#                                          1024, 64, 128,
#                                          32, 5, 2)
# from visualize_all import *
# abs_nitsche_circle_4angle_100000 = abs_error_only(300, pde, net_plus, net_minus)






'''flower'''
# save_path_flower = 'Save_results/flower/'
import matplotlib
matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
bp = 10
bm= 1
# from gpt_pde_flower_nobeta import *
from pde_flower_update import *
pde = flowerInterface(bp, bm)
net_plus, net_minus, err_l2 = train(9000,
                                         8000, 250,
                                         1024, 64, 128,
                                         32, 5, 2)
# from visualize_all import *
# visualize_2d_flower(300, pde, net_plus, net_minus)

# from visualize_all import *
# abs_nitsche_flower = abs_error_only(300, pde, net_plus, net_minus)

# net_plus, net_minus, min_l2error, err_l2 = train(4500,
#                                          3000, 100,
#                                          100, 100, 100,



















