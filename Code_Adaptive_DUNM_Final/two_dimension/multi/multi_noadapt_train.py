# multi_noadapt_train_flower_customized.py

import torch
from Fc_Net import FCNet
from multi_loss import *
from multi_3flowers_pde import *
from visualize_multi import *
import matplotlib
matplotlib.use('Qt5Agg')

# ------------------------------------------------------------------------
# Global device / dtype
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
# ------------------------------------------------------------------------


def setup_seed(seed: int = 888888):
    """Deterministic CPU ops; CUDA determinism left to user as needed."""
    torch.manual_seed(seed)


def weights_init_normal(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight, 0, 0.5)
        torch.nn.init.constant_(m.bias, 0.0)


def l2_error(u: torch.Tensor, uh: torch.Tensor) -> torch.Tensor:
    """Relative L2 error ||u-uh|| / ||u|| over a set of points."""
    num = torch.sqrt(torch.mean(torch.pow(u - uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)))
    error = num / den
    return error


def train_multi(
    epochs: int,
    gamma_b,
    gamma_f_list,
    Nr: int,
    Nb: int,
    Nf_list,
    m: int,
    depth: int,
    d: int,
    pde,
):
    """
    Train 4 subdomain networks with symmetric Nitsche coupling.
    """
    setup_seed(888888)
    loss_list = []
    l2_list = []

    # Initialize 4 networks and optimizers (one per subdomain)
    nets = [FCNet(d, m, 1, depth=depth).to(DEVICE).to(DTYPE) for _ in range(4)]
    opts = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in nets]

    # ---------------- Test set (no grad) ----------------
    x_t = pde.interior_points(10000).to(DEVICE, DTYPE)
    u_exact = pde.exact_judge(x_t).detach()

    # ---------------- Training points -------------------
    # xr needs gradients; xb/xf don't (the loss function will wrap xf with requires_grad where needed)
    xr = pde.interior_points(Nr).to(DEVICE, DTYPE)
    xb = pde.boundary_points(Nb).to(DEVICE, DTYPE)
    xf_list = [pde.interface_points(i, Nf_list[i]).to(DEVICE, DTYPE) for i in range(len(Nf_list))]

    xr.requires_grad_()

    # ---------------- Training loop ---------------------
    for i in range(epochs):
        loss_energy, loss_interior, loss_interface, loss_bc = loss_nitsche_multi_net(
            nets, xr, xb, xf_list, pde, gamma_b, gamma_f_list
        )

        for opt in opts:
            opt.zero_grad()
        loss_energy.backward()
        for opt in opts:
            opt.step()

        if i % 500 == 0:
            loss_list.append(loss_energy.item())

            # eval prediction (no grad, eval mode)
            for net in nets:
                net.eval()
            with torch.no_grad():
                u_pred = pde.predict_solution(nets, x_t)
                relative_l2 = l2_error(u_exact, u_pred)
            for net in nets:
                net.train()

            l2_list.append(relative_l2.item())
            print(
                f"[Iter {i}] loss = {loss_energy.item():.3e}, "
                f"l2_error = {relative_l2:.3e}, "
                f"interior = {loss_interior.item():.3e}, "
                f"bc = {loss_bc.item():.3e}, "
                f"interface = {loss_interface.item():.3e}"
            )
            # Optional early stop
            if i > 499 and relative_l2 > 1e+00:
                print(f"  Early stopping: err_l2 = {relative_l2:.3e} > 1e+00 at epoch {i}")
                return None, relative_l2

    # ---------------- Final test ------------------------
    num_test = 1200
    with torch.no_grad():
        x_t = pde.interior_points(num_test).to(DEVICE, DTYPE)
        u = pde.exact_judge(x_t)
        uh = pde.predict_solution(nets, x_t).reshape(-1, 1)
        err_l2 = l2_error(u, uh)

    print(f"[Final] Relative L2 error over {num_test} points= {err_l2:.3e}")
    abserror = abs_error_only(200, pde, nets)
    return nets, err_l2,  abserror


Nr = 1024

Nb = 64
Nf_list = [128,256,512]  # Interface points per interface (3 interfaces)
# Nf_list = [256, 256, 256]

# PDE model (must match multi_interface_flower_v2 in multi_pde_0823.py)
# k0 positive (manufactured RHS in Ω0 assumes -k0 r^2 + 3): use k0=3.0; k3=1.0
# pde = multi_interface_flower_gpt(-3.0, 1.0)

pde = multi_interface_flower_gpt(5.5, 1.0)

gamma_b, gamma_f = 3000,	(500.0, 500.0, 300.0)
m, depth = 32,5

epochs = 4000

print('No adaptation, Nitsche Method Result:')
print(f'Nr = {Nr}, Nb = {Nb}, Nf_list = {Nf_list}')
print(f'neurons = {m}, depth = {depth}')
print('gamma_b, gamma_f_list', gamma_b, gamma_f)

nets, err_l2, abs_error_nitsche = train_multi(
    epochs, gamma_b, gamma_f, Nr, Nb, Nf_list, m, depth, 2, pde
    )

# if __name__ == "__main__":
    # Sampling counts
    # Nr = 2048
    # Nr = 1024
    #
    # Nb = 64
    # Nf_list = [256,256,256]  # Interface points per interface (3 interfaces)
    # # Nf_list = [512, 128, 128]
    #
    # # PDE model (must match multi_interface_flower_v2 in multi_pde_0823.py)
    # # k0 positive (manufactured RHS in Ω0 assumes -k0 r^2 + 3): use k0=3.0; k3=1.0
    # # pde = multi_interface_flower_gpt(-3.0, 1.0)
    # pde = multi_interface_flower_gpt(3, 1.0)
    #
    # # Nitsche penalties
    # gamma_b = 3000
    # gamma_f_list = (300, 300, 300)  # per interface Γ0..Γ2
    #
    # # Network size
    # m, depth = 32, 5
    #
    # # Epochs (adjust as you wish)
    # epochs = 4000
    #
    # print('No adaptation, Nitsche Method Result:')
    # print(f'Nr = {Nr}, Nb = {Nb}, Nf_list = {Nf_list}')
    # print(f'neurons = {m}, depth = {depth}')
    # print('gamma_b, gamma_f_list', gamma_b, gamma_f_list)
    #
    # nets, err_l2, abs_error_nitsche = train_multi(
    #     epochs, gamma_b, gamma_f_list, Nr, Nb, Nf_list, m, depth, 2, pde
    #     )
    #
    # # Visualizations (guarded)
    # if nets is not None:
    #     nitsche_visualize_2d_whole_domain_2figures(250, pde, nets, kind='Nitsche')
    #     abs_error_2d_separate_bar(250, pde, nets, title='Nitsche Absolute Error (4 Subdomains)')