import torch
import numpy as np

class multi_interface_flower_gpt:
    """
    4-subdomain manufactured-solution Poisson/Nitsche problem in 2D.
    Geometry: three flower-shaped interfaces given by r(θ) = r_i - 0.1 cos(n_i θ),
              with (n0, n1, n2) = (4, 5, 6).
    Subdomains: Ω0 (innermost), Ω1, Ω2, Ω3 (outermost).
    """

    def __init__(self, k0=3.0, k3=1.0, r0=0.2, r1=0.5, r2=0.8):
        # PDE parameters
        self.k0 = float(k0)
        self.k3 = float(k3)

        # Interface radii
        self.r0 = float(r0)
        self.r1 = float(r1)
        self.r2 = float(r2)

        # Domain box [-1,1]^2 → area = 4
        self.domain_area = torch.tensor(4.0, dtype=torch.float32)

        # Diffusion per subdomain [Ω0, Ω1, Ω2, Ω3]
        self.beta_list = [
            torch.tensor(1.0),
            torch.tensor(2.0),
            torch.tensor(3.0),
            torch.tensor(4.0),
        ]

        # Precompute approximate subdomain areas by polar integration
        a0, a1tot, a2tot = self._compute_omega_areas_np(self.r0, self.r1, self.r2)
        self._omega0_area = torch.tensor(a0, dtype=torch.float32)
        self._omega1_total = torch.tensor(a1tot, dtype=torch.float32)
        self._omega2_total = torch.tensor(a2tot, dtype=torch.float32)

        # (Optional) precomputed interface lengths (can keep your numbers)
        self._len0 = torch.tensor(2.46580894287, dtype=torch.float32)
        self._len1 = torch.tensor(3.82477218065571, dtype=torch.float32)
        self._len2 = torch.tensor(5.48780963497857, dtype=torch.float32)

    # ---------- Basic meta ----------
    def dimension(self):
        return 2

    def subdomain_areas(self):
        omega_0 = self._omega0_area
        omega_1 = self._omega1_total - omega_0
        omega_2 = self._omega2_total - self._omega1_total
        omega_3 = self.domain_area - self._omega2_total
        return omega_0, omega_1, omega_2, omega_3

    def interface_length(self):
        return self._len0, self._len1, self._len2

    def boundary_length(self):
        return torch.tensor(8.0, dtype=torch.float32)

    # ---------- Geometry / masks ----------
    def _compute_omega_areas_np(self, r0, r1, r2):
        theta = np.linspace(0, 2*np.pi, 10000, endpoint=False)
        r_0 = r0 - 0.1 * np.cos(4 * theta)
        r_1 = r1 - 0.1 * np.cos(5 * theta)
        r_2 = r2 - 0.1 * np.cos(6 * theta)
        # area of region inside r(θ): (1/2) ∫ r(θ)^2 dθ
        integrand0 = 0.5 * r_0**2
        integrand1 = 0.5 * r_1**2
        integrand2 = 0.5 * r_2**2
        dθ = (2*np.pi) / len(theta)
        area_ring_0 = integrand0.sum() * dθ    # |Ω0|
        area_ring_1 = integrand1.sum() * dθ    # |Ω0|+|Ω1|
        area_ring_2 = integrand2.sum() * dθ    # |Ω0|+|Ω1|+|Ω2|
        return area_ring_0, area_ring_1, area_ring_2

    def phi(self, p):
        """Level-set functions for the three interfaces."""
        x = p[..., 0]
        y = p[..., 1]
        r = torch.sqrt((p**2).sum(dim=1))
        θ = torch.atan2(y, x)
        θ = torch.where(θ < 0, θ + 2*torch.pi, θ)

        phi0 = r - self.r0 + 0.1 * torch.cos(4 * θ)
        phi1 = r - self.r1 + 0.1 * torch.cos(5 * θ)
        phi2 = r - self.r2 + 0.1 * torch.cos(6 * θ)
        return phi0, phi1, phi2

    def subdomain_mask(self, p):
        phi0, phi1, phi2 = self.phi(p)
        mask_0 = (phi0 < 0)
        mask_1 = (phi0 >= 0) & (phi1 < 0)
        mask_2 = (phi1 >= 0) & (phi2 < 0)
        mask_3 = (phi2 >= 0)
        return mask_0, mask_1, mask_2, mask_3

    # ---------- Exact solution / RHS ----------
    def exact_judge(self, p):
        """Piecewise exact u (N,2)->(N,1)."""
        x = p[:, 0:1]
        y = p[:, 1:2]
        r = torch.sqrt((p**2).sum(dim=1, keepdim=True))

        m0, m1, m2, m3 = self.subdomain_mask(p)
        u = torch.zeros_like(x)

        # Ω0: u = -k0 r^2 + 3
        u[m0] = -self.k0 * (r[m0]**2) + torch.pi
        # Ω1: u = exp(x) + 1.3
        u[m1] = torch.exp(x[m1]) + 1.3
        # Ω2: u = sin(x) + 0.5
        u[m2] = torch.sin(x[m2]) + 0.5
        # Ω3: u = -x + log(k3 r + 2)
        u[m3] = -x[m3] + torch.log(self.k3 * r[m3] + 2.0)

        return u

    def exact_subdomain(self, p, idx):
        """Exact u_i evaluated at (N,2)->(N,1) for index i=0..3."""
        x = p[:, 0:1]
        y = p[:, 1:2]
        r = torch.sqrt((p**2).sum(dim=1, keepdim=True))
        if idx == 0:
            return -self.k0 * r**2 + torch.pi
        elif idx == 1:
            return torch.exp(x) + 1.3
        elif idx == 2:
            return torch.sin(x) + 0.5
        elif idx == 3:
            return -x + torch.log(self.k3 * r + 2.0)
        else:
            raise ValueError("idx must be 0..3")

    def rhs(self, p):
        """
        Return f such that  -β Δu = f  in each subdomain.
        (N,2)->(N,1)
        """
        x = p[:, 0:1]
        y = p[:, 1:2]
        r = torch.sqrt((p**2).sum(dim=1, keepdim=True))

        m0, m1, m2, m3 = self.subdomain_mask(p)
        rhs = torch.zeros_like(x)

        β = self.beta_list

        # Ω0: u = -k0 r^2 + 3 → Δu = -4k0 → -Δu = 4k0
        rhs[m0] = β[0] * (4.0 * self.k0)

        # Ω1: u = e^x + 1.3 → Δu = ∂xx u = e^x → -Δu = -e^x
        rhs[m1] = -β[1] * torch.exp(x[m1])

        # Ω2: u = sin x + 0.5 → Δu = ∂xx u = -sin x → -Δu = +sin x
        rhs[m2] = β[2] * torch.sin(x[m2])

        # Ω3: u = -x + log(k3 r + 2) → -Δu = k3^2/(k3 r + 2)^2 - k3/(r (k3 r + 2))
        rm3 = r[m3]
        rhs[m3] = β[3] * (
            (self.k3**2) / (self.k3 * rm3 + 2.0)**2
            - self.k3 / (rm3 * (self.k3 * rm3 + 2.0))
        )
        return rhs

    def BC_Dirichlet(self, p):
        return self.exact_judge(p)

    # ---------- Jumps for Nitsche ----------
    def function_jump_multi(self, p, i, j):
        """[u] = u_j - u_i on interface between Ω_i and Ω_j."""
        return self.exact_subdomain(p, j) - self.exact_subdomain(p, i)

    def interface_normal(self, interface_id, p):
        """
        Unit normal n pointing from Ω_i to Ω_{i+1} (outward, increasing r).
        Works for interface_id=0,1,2.
        """
        x = p[..., 0]
        y = p[..., 1]
        dtype, device = p.dtype, p.device
        eps = torch.as_tensor(1e-12, dtype=dtype, device=device)

        θ = torch.atan2(y, x)
        θ = torch.where(θ < 0, θ + 2*torch.pi, θ).view(-1, 1)

        if interface_id == 0:
            r = self.r0 - 0.1 * torch.cos(4 * θ)
            dr = 0.4 * torch.sin(4 * θ)
        elif interface_id == 1:
            r = self.r1 - 0.1 * torch.cos(5 * θ)
            dr = 0.5 * torch.sin(5 * θ)
        elif interface_id == 2:
            r = self.r2 - 0.1 * torch.cos(6 * θ)
            dr = 0.6 * torch.sin(6 * θ)
        else:
            raise ValueError("interface_id must be 0,1,2")

        # tangent t = d/dθ [ r cosθ, r sinθ ]
        t_x = dr * torch.cos(θ) - r * torch.sin(θ)
        t_y = dr * torch.sin(θ) + r * torch.cos(θ)

        # rotate by +90° to get a normal
        n_tilde = torch.cat([t_y, -t_x], dim=1)
        n_norm = torch.linalg.norm(n_tilde, dim=1, keepdim=True).clamp_min(eps)
        n = n_tilde / n_norm

        # enforce outward (increasing r)
        radial = torch.nn.functional.normalize(torch.cat([x.view(-1,1), y.view(-1,1)], dim=1), dim=1)
        sign = torch.sign((n * radial).sum(dim=1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return n * sign

    def grad_u_i(self, i, p):
        """
        ∇u_i(x,y) for exact solution in Ω_i (N,2)->(N,2)
        """
        x = p[:, 0:1]
        y = p[:, 1:2]
        r = torch.sqrt((p**2).sum(dim=1, keepdim=True))
        if i == 0:
            # u = -k0 r^2 + 3  ⇒ ∇u = -2 k0 (x, y)
            gx = -2.0 * self.k0 * x
            gy = -2.0 * self.k0 * y
            return torch.cat([gx, gy], dim=1)
        elif i == 1:
            gx = torch.exp(x)
            gy = torch.zeros_like(y)
            return torch.cat([gx, gy], dim=1)
        elif i == 2:
            gx = torch.cos(x)
            gy = torch.zeros_like(y)
            return torch.cat([gx, gy], dim=1)
        elif i == 3:
            denom = r * (self.k3 * r + 2.0)
            gx = -1.0 + self.k3 * x / denom
            gy =        self.k3 * y / denom
            return torch.cat([gx, gy], dim=1)
        else:
            raise ValueError("i must be 0..3")

    def flux_jump_multi(self, interface_id, p):
        """
        [β ∂_n u] = β_plus (∇u_+ · n) - β_minus (∇u_- · n),
        where n points from Ω_i → Ω_{i+1}.
        """
        n = self.interface_normal(interface_id, p)
        β_minus = torch.as_tensor(self.beta_list[interface_id],   dtype=p.dtype, device=p.device)
        β_plus  = torch.as_tensor(self.beta_list[interface_id+1], dtype=p.dtype, device=p.device)

        g_minus = self.grad_u_i(interface_id, p)
        g_plus  = self.grad_u_i(interface_id+1, p)

        dot_plus  = (g_plus  * n).sum(dim=1, keepdim=True)
        dot_minus = (g_minus * n).sum(dim=1, keepdim=True)
        return β_plus * dot_plus - β_minus * dot_minus

    # ---------- Sampling utilities ----------
    def interior_points(self, N_omega=1024):
        # uniform in box [-1,1]^2
        return (torch.rand(N_omega, 2) * 2.0 - 1.0)

    def boundary_points(self, N_b=32):
        # axis-aligned square boundary sampling (2N_b per axis)
        xb = (torch.rand(4 * N_b, 2) * 2.0 - 1.0)
        xb[:N_b, 0] = -1.0; xb[N_b:2*N_b, 0] =  1.0  # left/right
        xb[2*N_b:3*N_b, 1] = -1.0; xb[3*N_b:4*N_b, 1] = 1.0  # bottom/top
        return xb

    def num_interfaces(self):
        return 3

    def interface_r(self, idx, theta):
        if idx == 0:
            return self.r0 - 0.1 * torch.cos(4 * theta)
        elif idx == 1:
            return self.r1 - 0.1 * torch.cos(5 * theta)
        elif idx == 2:
            return self.r2 - 0.1 * torch.cos(6 * theta)
        else:
            raise ValueError("idx must be 0,1,2")

    def interface_points(self, interface_id, N_f=256):
        θ = 2 * torch.pi * torch.rand(N_f, 1)
        if interface_id == 0:
            r = self.r0 - 0.1 * torch.cos(4 * θ)
        elif interface_id == 1:
            r = self.r1 - 0.1 * torch.cos(5 * θ)
        elif interface_id == 2:
            r = self.r2 - 0.1 * torch.cos(6 * θ)
        else:
            raise ValueError("Invalid interface_id: 0, 1, or 2.")
        x = r * torch.cos(θ)
        y = r * torch.sin(θ)
        return torch.cat([x, y], dim=1)

    # ---------- Prediction over masks ----------
    def predict_solution(self, nets, x):
        """
        Evaluate u_pred(x) by routing each point to its subdomain net.
        nets: [net0, net1, net2, net3] ; each net: (N_i,2)->(N_i,1)
        x: (N,2) or (N,3)  (if 3, last column ignored)
        """
        x_input = x[:, :2] if x.shape[1] == 3 else x

        m0, m1, m2, m3 = self.subdomain_mask(x_input)

        # allocate like input to avoid device/dtype mismatches
        u_pred = torch.zeros(x_input.shape[0], 1, dtype=x_input.dtype, device=x_input.device)

        # eval mode + no_grad for inference
        prev_modes = []
        for net in nets:
            prev_modes.append(net.training)
            net.eval()
        with torch.no_grad():
            for i, mask in enumerate([m0, m1, m2, m3]):
                if mask.any():
                    u_pred[mask] = nets[i](x_input[mask])

        # restore modes (optional)
        for net, was_train in zip(nets, prev_modes):
            net.train(was_train)
        return u_pred

    # ---------- Diffusion coefficient field (optional utility) ----------
    def diffusion_coefficient(self, p):
        m0, m1, m2, m3 = self.subdomain_mask(p)
        β = torch.zeros(p.shape[0], 1, dtype=p.dtype, device=p.device)
        for i, m in enumerate([m0, m1, m2, m3]):
            β[m] = self.beta_list[i].to(dtype=p.dtype, device=p.device)
        return β