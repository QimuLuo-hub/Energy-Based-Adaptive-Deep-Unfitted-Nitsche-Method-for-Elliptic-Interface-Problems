import torch


def gradients(input, output):
    return torch.autograd.grad(outputs=output, inputs=input, grad_outputs = torch.ones_like(output),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]


def indicator_grad(net_plus, net_minus, xr, xf, xb, pde):

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()

    output_r = torch.where(pde.phi(xr), net_plus(xr), net_minus(xr))
    grad_r = gradients(xr, output_r)
    grad_r_magnitude = torch.norm(grad_r, dim=1)

    # #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)

    #xf-grad
    grad_f_plus = gradients(xf, net_plus(xf))
    # print('grad_f_plus.shape', grad_f_plus.shape)
    plus_magnitude = torch.norm(grad_f_plus, dim=1)
    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)
    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude
    # grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude), dim=0)
    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)

    return grad_all_magnitude

'''no sqrt'''
def indicator_grad_combined_beta_nosqrt(net_plus, net_minus, xr, xf, xb, pde):

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()

    #xr-grad
    output_r = torch.where(pde.phi(xr), net_plus(xr), net_minus(xr))
    grad_r = gradients(xr, output_r)
    grad_r_magnitude = torch.norm(grad_r, dim=1)

    #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)

    #xf-grad
    grad_f_plus = gradients(xf, net_plus(xf))
    plus_magnitude = torch.norm(grad_f_plus, dim=1)
    plus_magnitude = plus_magnitude

    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)
    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude

    # coefficient * indicator:
    diff_coeff_r = pde.diffusion_coefficient_judge(xr)[:, 0] #both
    diff_coeff_b = pde.diffusion_coefficient_judge(xb)[:, 0] #all bp = 1
    diff_coeff_f = pde.diffusion_coefficient_judge(xf)[:, 0]# both

    grad_r_magnitude = diff_coeff_r * grad_r_magnitude
    grad_b_magnitude = diff_coeff_b * grad_b_magnitude
    grad_f_magnitude = diff_coeff_f * grad_f_magnitude
    # print('grad_r_magnitude.shape', grad_r_magnitude.shape)
    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)

    return grad_all_magnitude



def indicator_grad_beta_separate_nosqrt(net_plus, net_minus, xr, xf, xb, pde):
    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()
    dim = pde.dimension()

    #xr-grad
    xr_sign = pde.phi(xr)
    mask_plus = xr_sign
    mask_minus = ~xr_sign
    xr_plus = torch.masked_select(xr, mask_plus).reshape(-1, dim)
    xr_minus = torch.masked_select(xr, mask_minus).reshape(-1, dim)

    grad_r_plus = gradients(xr_plus, net_plus(xr_plus))
    grad_r_minus = gradients(xr_minus, net_minus(xr_minus))

    diff_coeff_r_plus = pde.diffusion_plus(xr_plus)[:,0]
    diff_coeff_r_minus = pde.diffusion_minus(xr_minus)[:,0]

    grad_r_plus_magnitude = torch.norm(grad_r_plus, dim=1) * diff_coeff_r_plus
    grad_r_minus_magnitude = torch.norm(grad_r_minus, dim=1) * diff_coeff_r_minus
    grad_r_magnitude = torch.cat((grad_r_plus_magnitude, grad_r_minus_magnitude), dim=0)

    #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)
    diff_coeff_b = pde.diffusion_coefficient_judge(xb)[:, 0] #all bp = 1
    grad_b_magnitude = diff_coeff_b * grad_b_magnitude

    #xf-grad
    beta_plus = pde.diffusion_plus(xf)[:,0]
    beta_minus = pde.diffusion_minus(xf)[:, 0]

    grad_f_plus = gradients(xf, net_plus(xf))
    plus_magnitude = torch.norm(grad_f_plus, dim=1)*beta_plus
    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)*beta_minus

    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude

    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)
    return grad_all_magnitude

'''sqrt: '''
def indicator_grad_beta_separate_sqrt(net_plus, net_minus, xr, xf, xb, pde):
    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()
    dim = pde.dimension()

    #xr-grad
    xr_sign = pde.phi(xr)
    mask_plus = xr_sign
    mask_minus = ~xr_sign
    xr_plus = torch.masked_select(xr, mask_plus).reshape(-1, dim)
    xr_minus = torch.masked_select(xr, mask_minus).reshape(-1, dim)

    grad_r_plus = gradients(xr_plus, net_plus(xr_plus))
    grad_r_minus = gradients(xr_minus, net_minus(xr_minus))

    diff_coeff_r_plus = torch.sqrt(pde.diffusion_plus(xr_plus)[:,0])
    diff_coeff_r_minus =torch.sqrt(pde.diffusion_minus(xr_minus)[:,0])

    grad_r_plus_magnitude = torch.norm(grad_r_plus, dim=1) * diff_coeff_r_plus
    grad_r_minus_magnitude = torch.norm(grad_r_minus, dim=1) * diff_coeff_r_minus
    grad_r_magnitude = torch.cat((grad_r_plus_magnitude, grad_r_minus_magnitude), dim=0)

    #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)
    diff_coeff_b = torch.sqrt(pde.diffusion_coefficient_judge(xb)[:, 0]) #all bp = 1
    grad_b_magnitude = diff_coeff_b * grad_b_magnitude

    #xf-grad
    beta_plus = torch.sqrt(pde.diffusion_plus(xf)[:,0])
    beta_minus = torch.sqrt(pde.diffusion_minus(xf)[:, 0])

    grad_f_plus = gradients(xf, net_plus(xf))
    plus_magnitude = torch.norm(grad_f_plus, dim=1)*beta_plus
    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)*beta_minus

    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude

    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)

    return grad_all_magnitude


def indicator_grad_combined_beta_sqrt(net_plus, net_minus, xr, xf, xb, pde):

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()

    #xr-grad
    output_r = torch.where(pde.phi(xr), net_plus(xr), net_minus(xr))
    grad_r = gradients(xr, output_r)
    grad_r_magnitude = torch.norm(grad_r, dim=1)

    #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)

    #xf-grad
    grad_f_plus = gradients(xf, net_plus(xf))
    plus_magnitude = torch.norm(grad_f_plus, dim=1)
    plus_magnitude = plus_magnitude

    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)
    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude

    # coefficient * indicator:
    diff_coeff_r = torch.sqrt(pde.diffusion_coefficient_judge(xr)[:, 0]) #both
    diff_coeff_b = torch.sqrt(pde.diffusion_coefficient_judge(xb)[:, 0]) #all bp = 1
    diff_coeff_f = torch.sqrt(pde.diffusion_coefficient_judge(xf)[:, 0]) # both

    grad_r_magnitude = diff_coeff_r * grad_r_magnitude
    grad_b_magnitude = diff_coeff_b * grad_b_magnitude
    grad_f_magnitude = diff_coeff_f * grad_f_magnitude
    # print('grad_r_magnitude.shape', grad_r_magnitude.shape)
    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)

    return grad_all_magnitude


# ----------------- Mixed indicator: residual + grad*sqrt(beta) -----------------

def _gradients(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((0, x.shape[1] if x.ndim == 2 else 1), requires_grad=True)
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def _divergence(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # v shape: (N, dim); returns (N, 1)
    parts = []
    for j in range(v.shape[1]):
        dvj_dx = torch.autograd.grad(
            v[:, j:j+1], x, torch.ones_like(v[:, j:j+1]),
            create_graph=True, retain_graph=True
        )[0][:, j:j+1]
        parts.append(dvj_dx)
    return torch.sum(torch.cat(parts, dim=1), dim=1, keepdim=True)

def indicator_residual_full(net_plus, net_minus, xr, xf, xb, pde):
    """
    Residual-like indicator aligned with [xr ; xf ; xb].
      - interior: | -div(beta ∇u) - f |
      - interface: |jump value| + |jump flux|
      - boundary: |u - g_D|
    Returns a 1D tensor of length xr.size(0)+xf.size(0)+xb.size(0).
    """
    # ---- interior ----
    xr = xr.clone().detach().requires_grad_(True)
    mask_plus = pde.phi(xr)                 # (N_r,1) bool
    mp = mask_plus.reshape(-1, 1).to(xr.dtype)
    mm = 1.0 - mp

    up = net_plus(xr)
    um = net_minus(xr)
    u_r = up * mp + um * mm

    gu = _gradients(xr, u_r)                           # (N_r,2)
    beta = pde.diffusion_coefficient_judge(xr)         # (N_r,1)
    beta_gu = beta * gu                                # broadcast
    div_beta_gu = _divergence(xr, beta_gu)             # (N_r,1)
    f = pde.rhs_judge(xr)                              # (N_r,1)
    res_r = torch.abs(-div_beta_gu - f)                # (N_r,1)

    # ---- interface (jump magnitudes) ----
    if xf.numel() == 0:
        res_f = xf.new_zeros((0, 1), dtype=xf.dtype)
    else:
        xf = xf.clone().detach().requires_grad_(True)
        upf = net_plus(xf)
        umf = net_minus(xf)
        jp = torch.abs(upf - umf)                      # |jump value|
        gp = _gradients(xf, upf)
        gm = _gradients(xf, umf)
        n  = pde.interface_normal(xf)
        bp = pde.diffusion_plus(xf)
        bm = pde.diffusion_minus(xf)
        jf = torch.abs((gp * n).sum(1, keepdim=True) * bp - (gm * n).sum(1, keepdim=True) * bm)
        res_f = jp + jf

    # ---- boundary (Dirichlet) ----
    if xb.numel() == 0:
        res_b = xb.new_zeros((0, 1), dtype=xb.dtype)
    else:
        xb = xb.clone().detach()
        ub = torch.where(pde.phi(xb), net_plus(xb), net_minus(xb))
        gb = pde.BC_Dirichlet(xb)
        res_b = torch.abs(ub - gb)

    return torch.cat((res_r, res_f, res_b), dim=0).reshape(-1)

def indicator_mixed_select_indices(
    net_plus, net_minus,
    xr, xf, xb,
    pde,
    mix_fraction: float = 0.3,     # fraction of overall-top allocated to residual
    total_top_fraction: float = 0.10,  # overall top percentage
):
    """
    Select indices (into [xr; xf; xb]) mixing residual and grad*sqrt(beta) indicators.
    Example: mix_fraction=0.3 and total_top_fraction=0.10 → 3% residual + 7% gradient.
    """
    # previous gradient*sqrt(beta) indicator
    ind_grad = indicator_grad_combined_beta_sqrt(net_plus, net_minus, xr, xf, xb, pde)
    # residual indicator
    ind_res  = indicator_residual_full(net_plus, net_minus, xr, xf, xb, pde)

    N = ind_grad.numel()
    n_top = max(1, int(total_top_fraction * N))
    n_res = int(mix_fraction * n_top)
    n_grd = n_top - n_res

    idx_grad = torch.argsort(ind_grad, descending=True)
    idx_res  = torch.argsort(ind_res,  descending=True)

    pick_res  = idx_res[:n_res]
    pick_grad = idx_grad[:n_grd]

    picks = torch.unique(torch.cat((pick_res, pick_grad), dim=0))
    if picks.numel() < n_top:
        need = n_top - picks.numel()
        extra = idx_grad[~torch.isin(idx_grad, picks)][:need]
        picks = torch.cat((picks, extra), dim=0)

    return picks.long()