import torch

# ----------------------- autodiff helper -----------------------
def gradients(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇_x outputs(x) for scalar-valued outputs with shape (N,1).
    inputs:  (N,2) requires_grad=True
    outputs: (N,1)
    returns: (N,2)
    """
    if outputs.ndim != 2 or outputs.shape[1] != 1:
        raise ValueError("outputs must have shape (N,1)")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("inputs/outputs batch sizes must match")
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad


# -------------------- Nitsche loss (4 subdomains) --------------------
def loss_nitsche_multi_net(
    nets,
    xr: torch.Tensor,
    xb: torch.Tensor,
    xf_list: list,
    pde,
    gamma_b: float,
    gamma_f_list: list,
):
    """
    Symmetric Nitsche loss for 4 subdomains Ω0..Ω3 with 3 interfaces Γ0..Γ2.

    nets         : [net0, net1, net2, net3], each maps (Ni,2)->(Ni,1)
    xr           : interior samples in the box, shape (Nr,2)
    xb           : boundary samples on ∂Ω, shape (Nb,2)
    xf_list      : list of length 3, each (Nf_m, 2) samples on interface Γm
    pde          : instance of multi_interface_flower_v2
    gamma_b      : Dirichlet penalty weight
    gamma_f_list : list/tuple length 3 with interface penalty weights
    Returns      : total_loss, loss_r, loss_f, loss_b
    """
    # ---- prepare tensors ----
    xr = xr.detach().clone().requires_grad_(True)
    xb = xb.detach().clone()  # boundary term doesn't need grads

    # ---- Interior energy: sum_i |Ωi| * mean( 1/2 β_i |∇u|^2 - u f ) ----
    loss_r = xr.new_tensor(0.0)
    areas = pde.subdomain_areas()  # tuple of 4 tensors (areas)

    masks_r = pde.subdomain_mask(xr)  # boolean masks for Ω0..Ω3
    for i in range(4):
        xri = xr[masks_r[i]]
        if xri.numel() == 0:
            continue
        ui = nets[i](xri)                   # (Ni,1)
        gi = gradients(xri, ui)             # (Ni,2)
        fi = pde.rhs(xri)                   # (Ni,1)
        beta_i = torch.as_tensor(pde.beta_list[i], dtype=xri.dtype, device=xri.device)

        gradpart = gi.pow(2).sum(dim=1, keepdim=True) * beta_i
        loss_i = 0.5 * gradpart - ui * fi
        loss_r = loss_r + loss_i.mean() * areas[i].to(xri)

    # ---- Boundary penalty: γ_b * |∂Ω| * mean( (u - g)^2 ) ----
    masks_b = pde.subdomain_mask(xb)
    output_b = torch.zeros_like(xb[:, 0:1])  # allocate on xb's device/dtype
    for i in range(4):
        xb_i = xb[masks_b[i]]
        if xb_i.numel() == 0:
            continue
        output_b[masks_b[i]] = nets[i](xb_i)
    bc_val = pde.BC_Dirichlet(xb)
    loss_b = (output_b - bc_val).pow(2).mean()
    loss_b = loss_b * float(gamma_b) * pde.boundary_length().to(xb)

    # ---- Interfaces: sum_m |Γm| * mean( symmetric Nitsche integrand ) ----
    loss_f = xr.new_tensor(0.0)
    inter_lengths = pde.interface_length()  # tuple of 3 tensors (lengths)

    for m in range(3):
        if m >= len(xf_list):
            continue
        xf = xf_list[m]
        if xf is None or xf.numel() == 0:
            continue
        xf = xf.detach().clone().requires_grad_(True)

        ui = nets[m](xf)           # (Nf,1) value from minus side (Ω_m)
        uj = nets[m + 1](xf)       # (Nf,1) value from plus side  (Ω_{m+1})
        gi = gradients(xf, ui)     # (Nf,2)
        gj = gradients(xf, uj)     # (Nf,2)

        beta_i = torch.as_tensor(pde.beta_list[m],     dtype=xf.dtype, device=xf.device)
        beta_j = torch.as_tensor(pde.beta_list[m + 1], dtype=xf.dtype, device=xf.device)
        den = beta_i + beta_j
        kappa_i = beta_j / den
        kappa_j = beta_i / den
        lam = 2.0 * beta_i * beta_j / den

        # unit normal from Ω_m → Ω_{m+1}
        n = pde.interface_normal(m, xf)
        flux_i = beta_i * (gi * n).sum(dim=1, keepdim=True)
        flux_j = beta_j * (gj * n).sum(dim=1, keepdim=True)

        jump_u = uj - ui
        flux_avg = kappa_i * flux_j + kappa_j * flux_i
        val_avg = kappa_j * uj + kappa_i * ui

        fun_jump = pde.function_jump_multi(xf, m, m + 1)  # [u]_exact = u+ - u-
        flux_jump = pde.flux_jump_multi(m, xf)            # [β ∂_n u]_exact (same n)

        gamma_f = float(gamma_f_list[m])
        loss_interface = (
            0.5 * gamma_f * lam * ((jump_u - fun_jump).pow(2) - fun_jump.pow(2))
            + (jump_u - fun_jump) * flux_avg
            + flux_jump * val_avg
        )
        loss_f = loss_f + loss_interface.mean() * inter_lengths[m].to(xf)

    total_loss = loss_r + loss_b + loss_f
    return total_loss, loss_r, loss_f, loss_b