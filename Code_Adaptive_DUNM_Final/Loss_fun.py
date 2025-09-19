import torch


def gradients(input, output):
    return torch.autograd.grad(outputs=output, inputs=input, grad_outputs = torch.ones_like(output),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

def weight_avg(w1, w2, K1, K2):
    avg = w1 * K1 + w2 * K2
    return avg

import time
def loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b = 3000, gamma_f = 500):
    xr, xf, xb = map(lambda x: x.clone().detach().requires_grad_(True), (xr, xf, xb))

    # part1_time = time.time()
    # interior value and gradient
    dim = pde.dimension()

    xr_sign = pde.phi(xr)
    mask_plus = xr_sign
    mask_minus = ~xr_sign
    xr_plus = torch.masked_select(xr, mask_plus).reshape(-1, dim)
    xr_minus = torch.masked_select(xr, mask_minus).reshape(-1, dim)

    output_r_plus = net_plus(xr_plus)
    output_r_minus = net_minus(xr_minus)
    grad_r_plus = gradients(xr_plus, output_r_plus)
    grad_r_minus = gradients(xr_minus, output_r_minus)
    rhs_r_plus = pde.rhs_plus(xr_plus)
    rhs_r_minus = pde.rhs_minus(xr_minus)

    diff_coeff_r_plus = pde.diffusion_plus(xr_plus)
    diff_coeff_r_minus = pde.diffusion_minus(xr_minus)

    domain_area_plus, domain_area_minus = pde.subdomain_areas()

    # part1_time = time.time() - part1_time
    # print(f'part1_time: {part1_time}')
    # boundary value and gradient
    #not straight
    # output_b = net_minus(xb)
    # output_b = net_plus(xb) # pay attention which one to use use net_2_plus

    #straight
    output_b = torch.where(pde.phi(xb), net_plus(xb), net_minus(xb))
    bnd_val = pde.BC_Dirichlet(xb)

    # interface value and gradient
    output_f_plus = net_plus(xf)
    output_f_minus = net_minus(xf)
    grad_f_plus = gradients(xf, output_f_plus)
    grad_f_minus = gradients(xf, output_f_minus)
    interface_normal = pde.interface_normal(xf)

    beta_plus = pde.diffusion_plus(xf)
    beta_minus = pde.diffusion_minus(xf)

    den = beta_plus + beta_minus
    kappa_plus = beta_minus / den
    kappa_minus = beta_plus / den
    lam = 2*beta_plus*beta_minus / den

    output_f_val_jump = output_f_plus - output_f_minus
    flux_plus = torch.sum(grad_f_plus*interface_normal, dim=1)
    flux_plus = flux_plus.reshape(-1, 1)
    flux_plus *= beta_plus

    flux_minus = torch.sum(grad_f_minus*interface_normal, dim=1)
    flux_minus = flux_minus.reshape(-1, 1)
    flux_minus *= beta_minus

    output_f_flux_average = kappa_plus * flux_plus + kappa_minus * flux_minus
    #output_f_flux_average = (flux_plus + flux_minus)/2
    output_f_val_dual_average = kappa_minus * output_f_plus + kappa_plus * output_f_minus
    #output_f_val_dual_average = (output_f_plus + output_f_minus)/2
    fun_jump = pde.function_jump_p(xf)
    flux_jump = pde.flux_jump_q(xf)
    interface_length = pde.interface_length()

    # compute interior loss_r
    gradpart_plus = torch.sum(torch.pow(grad_r_plus, 2),dim=1)
    gradpart_plus = gradpart_plus.reshape(-1, 1) * diff_coeff_r_plus
    loss_r_plus = 0.5*gradpart_plus - output_r_plus*rhs_r_plus
    loss_r_plus = torch.mean(loss_r_plus)
    loss_r_plus = loss_r_plus * domain_area_plus
    gradpart_minus = torch.sum(torch.pow(grad_r_minus,2),dim=1)
    gradpart_minus = gradpart_minus.reshape(-1,1) * diff_coeff_r_minus
    loss_r_minus = 0.5*gradpart_minus - output_r_minus*rhs_r_minus
    loss_r_minus = torch.mean(loss_r_minus)
    loss_r_minus = loss_r_minus * domain_area_minus
    loss_r = loss_r_plus + loss_r_minus


    # compute boundary loss_b
    loss_b = torch.pow(output_b-bnd_val, 2)
    loss_b = torch.mean(loss_b)

    # compute interface loss_f
    #u_outside - u_inside, u2-u1
    loss_f = 0.5*gamma_f*lam*(torch.square(output_f_val_jump - fun_jump) - torch.square(fun_jump)) + \
            (output_f_val_jump - fun_jump)*output_f_flux_average + \
            flux_jump*output_f_val_dual_average

    #same
    loss_f = torch.mean(loss_f)
    loss_f = loss_f * interface_length

    # total loss
    bd_len = pde.boundary_length()
    loss_b = bd_len * gamma_b * loss_b
    loss = loss_r + loss_f + loss_b
    return loss, loss_r, loss_f, loss_b





def residuals_nitsche_weighted(net_plus, net_minus, xr, xf, xb, pde, gamma_b=3000, gamma_f=500):
    xr, xf, xb = map(lambda x: x.clone().detach().requires_grad_(True), (xr, xf, xb))
    dim = pde.dimension()

    # Geometry weights
    area_plus, area_minus = pde.subdomain_areas()
    bd_len = pde.boundary_length()
    interface_len = pde.interface_length()

    sqrt_area_plus = torch.sqrt(torch.tensor(area_plus))
    sqrt_area_minus = torch.sqrt(torch.tensor(area_minus))
    sqrt_bd_len = torch.sqrt(torch.tensor(bd_len))
    sqrt_interface_len = torch.sqrt(torch.tensor(interface_len))

    sqrt_gamma_b = torch.sqrt(torch.tensor(gamma_b))
    sqrt_gamma_f = torch.sqrt(torch.tensor(gamma_f))

    # Interior split
    xr_sign = pde.phi(xr)
    mask_plus = xr_sign
    mask_minus = ~xr_sign
    xr_plus = torch.masked_select(xr, mask_plus).reshape(-1, dim)
    xr_minus = torch.masked_select(xr, mask_minus).reshape(-1, dim)

    output_r_plus = net_plus(xr_plus)
    output_r_minus = net_minus(xr_minus)
    grad_r_plus = gradients(xr_plus, output_r_plus)
    grad_r_minus = gradients(xr_minus, output_r_minus)
    rhs_r_plus = pde.rhs_plus(xr_plus)
    rhs_r_minus = pde.rhs_minus(xr_minus)

    diff_coeff_r_plus = pde.diffusion_plus(xr_plus)
    diff_coeff_r_minus = pde.diffusion_minus(xr_minus)

    # Interior residuals with area scaling
    gradpart_plus = torch.sum(grad_r_plus**2, dim=1, keepdim=True) * diff_coeff_r_plus
    residual_r_plus = sqrt_area_plus * (0.5 * gradpart_plus - output_r_plus * rhs_r_plus)

    gradpart_minus = torch.sum(grad_r_minus**2, dim=1, keepdim=True) * diff_coeff_r_minus
    residual_r_minus = sqrt_area_minus * (0.5 * gradpart_minus - output_r_minus * rhs_r_minus)

    # Boundary residuals
    output_b = torch.where(pde.phi(xb), net_plus(xb), net_minus(xb))
    bnd_val = pde.BC_Dirichlet(xb)
    residual_b = sqrt_gamma_b * sqrt_bd_len * (output_b - bnd_val)

    # Interface residuals
    output_f_plus = net_plus(xf)
    output_f_minus = net_minus(xf)
    grad_f_plus = gradients(xf, output_f_plus)
    grad_f_minus = gradients(xf, output_f_minus)
    interface_normal = pde.interface_normal(xf)

    beta_plus = pde.diffusion_plus(xf)
    beta_minus = pde.diffusion_minus(xf)

    den = beta_plus + beta_minus
    kappa_plus = beta_minus / den
    kappa_minus = beta_plus / den
    lam = 2 * beta_plus * beta_minus / den

    output_f_val_jump = output_f_plus - output_f_minus

    flux_plus = torch.sum(grad_f_plus * interface_normal, dim=1, keepdim=True) * beta_plus
    flux_minus = torch.sum(grad_f_minus * interface_normal, dim=1, keepdim=True) * beta_minus
    output_f_flux_average = kappa_plus * flux_plus + kappa_minus * flux_minus

    output_f_val_dual_average = kappa_minus * output_f_plus + kappa_plus * output_f_minus
    fun_jump = pde.function_jump_p(xf)
    flux_jump = pde.flux_jump_q(xf)

    interface_res_raw = (
        0.5 * lam * ((output_f_val_jump - fun_jump)**2 - fun_jump**2) +
        (output_f_val_jump - fun_jump) * output_f_flux_average +
        flux_jump * output_f_val_dual_average
    )

    residual_f = sqrt_gamma_f * sqrt_interface_len * interface_res_raw

    # Concatenate all residuals
    residuals = torch.cat([
        residual_r_plus,
        residual_r_minus,
        residual_b,
        residual_f
    ], dim=0)

    return residuals.squeeze()
































# def loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b = 3000, gamma_f = 500):
#     '''Interior'''
#     # interior value and gradient
#     dim = pde.dimension()
#     xr_sign = pde.phi(xr)
#     mask_plus = xr_sign
#     mask_minus = ~xr_sign
#     xr_plus = torch.masked_select(xr, mask_plus)
#     xr_plus = xr_plus.reshape(-1, dim)
#     xr_minus = torch.masked_select(xr, mask_minus)
#     xr_minus = xr_minus.reshape(-1, dim)
#
#     output_r_plus = net_plus(xr_plus)
#     output_r_minus = net_minus(xr_minus)
#     grad_r_plus = gradients(xr_plus, output_r_plus)
#     grad_r_minus = gradients(xr_minus, output_r_minus)
#     rhs_r_plus = pde.rhs_plus(xr_plus)
#     rhs_r_minus = pde.rhs_minus(xr_minus)
#
#     diff_coeff_r_plus = pde.diffusion_plus(xr_plus)
#     diff_coeff_r_minus = pde.diffusion_minus(xr_minus)
#     domain_area_plus, domain_area_minus = pde.subdomain_areas()
#
#     '''Boundary'''
#     # boundary value and gradient
#     bd_len = pde.boundary_length()
#     output_b = net_minus(xb)
#     bnd_val = pde.BC_Dirichlet(xb)
#
#     # compute boundary loss_b
#     loss_b = torch.pow(output_b - bnd_val, 2)
#     loss_b = torch.mean(loss_b)
#
#     '''Interface'''
#     # interface value and gradient
#     output_f_plus = net_plus(xf)
#     output_f_minus = net_minus(xf)
#     grad_f_plus = gradients(xf, output_f_plus)
#     grad_f_minus = gradients(xf, output_f_minus)
#
#     #avg
#     interface_normal = pde.interface_normal(xf)
#     beta_plus = pde.diffusion_plus(xf)
#     beta_minus = pde.diffusion_minus(xf)
#     den = beta_plus + beta_minus
#     kappa_plus = beta_minus / den
#     kappa_minus = beta_plus / den
#     lam = 2 * beta_plus * beta_minus / den
#
#     output_f_val_jump = output_f_plus - output_f_minus
#     flux_plus = torch.sum(grad_f_plus*interface_normal, dim=1)
#     flux_plus = flux_plus.reshape(-1, 1)
#     flux_plus *= beta_plus
#     flux_minus = torch.sum(grad_f_minus*interface_normal, dim=1)
#     flux_minus = flux_minus.reshape(-1, 1)
#     flux_minus *= beta_minus
#
#     output_f_flux_average = kappa_plus*flux_plus + kappa_minus*flux_minus
#     # output_f_flux_average = weight_avg(flux_plus, flux_minus, kappa_plus, kappa_minus)
#     #output_f_flux_average = (flux_plus + flux_minus)/2
#
#     output_f_val_dual_average = kappa_minus*output_f_plus + kappa_plus*output_f_minus
#     # output_f_val_dual_average = weight_avg(output_f_plus, output_f_minus, kappa_minus, kappa_plus)
#     #output_f_val_dual_average = (output_f_plus + output_f_minus)/2
#
#     fun_jump = pde.function_jump_p(xf)
#     flux_jump = pde.flux_jump_q(xf)
#     interface_length = pde.interface_length()
#
#     '''Interior Loss'''
#     gradpart_plus = torch.sum(torch.pow(grad_r_plus, 2), dim=1)
#     gradpart_plus = gradpart_plus.reshape(-1, 1) * diff_coeff_r_plus
#     loss_r_plus = 0.5 * gradpart_plus - output_r_plus * rhs_r_plus
#     loss_r_plus = torch.mean(loss_r_plus)
#     loss_r_plus = loss_r_plus * domain_area_plus
#     gradpart_minus = torch.sum(torch.pow(grad_r_minus, 2), dim=1)
#     gradpart_minus = gradpart_minus.reshape(-1, 1) * diff_coeff_r_minus
#     loss_r_minus = 0.5 * gradpart_minus - output_r_minus * rhs_r_minus
#     loss_r_minus = torch.mean(loss_r_minus)
#     loss_r_minus = loss_r_minus * domain_area_minus
#
#     # loss_r = loss_r_plus + loss_r_minus
#     loss_r = loss_r_plus + loss_r_minus
#
#     '''Interface Loss'''
#     loss_f = 0.5*gamma_f * lam * (torch.square(output_f_val_jump - fun_jump) - torch.square(fun_jump)) + \
#             (output_f_val_jump-fun_jump) * output_f_flux_average +\
#             flux_jump * output_f_val_dual_average
#
#     loss_f = torch.mean(loss_f)
#     loss_f = loss_f * interface_length
#
#     '''Boundary loss'''
#     loss = loss_r + loss_f + bd_len * gamma_b * loss_b
#     # loss = loss_r + loss_f + gamma_b * loss_b
#     return loss, loss_r, loss_f, loss_b





