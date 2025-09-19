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


def indicator_grad_sqrt_beta(nets, xr, xf, xb, pde):
    """
    Indicator based on gradient magnitude weighted by sqrt(beta).
    - Interior/Boundary:  sqrt(β_i) * ||∇u_i||  within each subdomain i
    - Interfaces Γ_i:     max( sqrt(β_i) * ||∇u_i||,  sqrt(β_{i+1}) * ||∇u_{i+1}|| )

    This avoids ambiguity of calling pde.diffusion_coefficient on interface points
    (which may fall on either side); instead, we use the known β for each adjacent
    subdomain.
    """
    # Ensure required grads only where needed
    xr = xr.requires_grad_(True)
    xb = xb.requires_grad_(True)

    # ---------------- Interior ----------------
    masks_r = pde.subdomain_mask(xr)
    grad_r_magnitude = []
    for i in range(len(nets)):
        xr_i = xr[masks_r[i]]
        if xr_i.shape[0] == 0:
            continue
        out_r = nets[i](xr_i)
        grad_r = gradients(xr_i, out_r)  # (Ni,2)
        mag = torch.linalg.norm(grad_r, dim=1)  # (Ni,)
        beta_i = torch.as_tensor(pde.beta_list[i], dtype=xr_i.dtype, device=xr_i.device)
        grad_r_magnitude.append(torch.sqrt(beta_i) * mag)
    grad_r_magnitude = torch.cat(grad_r_magnitude, dim=0) if len(grad_r_magnitude) else torch.empty(0, device=xr.device, dtype=xr.dtype)

    # ---------------- Boundary ----------------
    masks_b = pde.subdomain_mask(xb)
    grad_b_magnitude = []
    for i in range(len(nets)):
        xb_i = xb[masks_b[i]]
        if xb_i.shape[0] == 0:
            continue
        out_b = nets[i](xb_i)
        grad_b = gradients(xb_i, out_b)
        mag = torch.linalg.norm(grad_b, dim=1)
        beta_i = torch.as_tensor(pde.beta_list[i], dtype=xb_i.dtype, device=xb_i.device)
        grad_b_magnitude.append(torch.sqrt(beta_i) * mag)
    grad_b_magnitude = torch.cat(grad_b_magnitude, dim=0) if len(grad_b_magnitude) else torch.empty(0, device=xb.device, dtype=xb.dtype)

    # ---------------- Interfaces ----------------
    grad_f_magnitude = []
    num_interfaces = 3
    for i in range(num_interfaces):
        xf_i = xf[i]
        if xf_i is None or xf_i.numel() == 0:
            continue
        xf_i = xf_i.requires_grad_(True)

        # grads from each side
        out_plus = nets[i + 1](xf_i)  # Ω_{i+1}
        out_minus = nets[i](xf_i)     # Ω_i
        grad_plus = gradients(xf_i, out_plus)
        grad_minus = gradients(xf_i, out_minus)

        mag_plus = torch.linalg.norm(grad_plus, dim=1)
        mag_minus = torch.linalg.norm(grad_minus, dim=1)

        beta_minus = torch.as_tensor(pde.beta_list[i],     dtype=xf_i.dtype, device=xf_i.device)
        beta_plus  = torch.as_tensor(pde.beta_list[i + 1], dtype=xf_i.dtype, device=xf_i.device)

        scaled_plus  = torch.sqrt(beta_plus)  * mag_plus
        scaled_minus = torch.sqrt(beta_minus) * mag_minus

        max_mag = torch.maximum(scaled_plus, scaled_minus)
        grad_f_magnitude.append(max_mag)

    grad_f_magnitude = torch.cat(grad_f_magnitude, dim=0) if len(grad_f_magnitude) else torch.empty(0, device=xr.device, dtype=xr.dtype)

    # ---------------- Combine all ----------------
    return torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)