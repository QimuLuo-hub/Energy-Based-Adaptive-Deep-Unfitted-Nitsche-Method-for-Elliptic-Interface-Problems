import torch

class ellipsoidal_interface:
    """
    Diffusion-form PDE in 3D:   -div( beta * grad u ) = f

    Geometry:
      Interface: 2 x^2 + 3 y^2 + 6 z^2 = 1.69  (ellipsoid)
      phi(p) > 0 denotes the OUTSIDE (plus) region.

    Exact solutions:
      Outside (plus): u_+(x,y,z) = x + y + z          => Δu_+ = 0         => f_+ = 0
      Inside (minus): u_-(x,y,z) = x^2 + y^2 + z^2    => Δu_- = 2*d = 6   => f_- = -β_- * 6

    Notes:
      • All outputs match the input dtype/device.
      • β_+, β_- stored as tensors with torch's default float dtype.
    """

    def __init__(self, bp: float = 60.0, bm: float = 2.0, dim: int = 3):
        dtype = torch.get_default_dtype()
        self.dim = dim  # keep as plain int
        self.bp = torch.tensor(bp, dtype=dtype)
        self.bm = torch.tensor(bm, dtype=dtype)
        # Volume of cube [-1,1]^dim
        self.domain_area = (torch.tensor(2.0, dtype=dtype) ** self.dim)

    # ----------------------------- Basics -----------------------------
    def dimension(self):
        return 3

    def subdomain_areas(self):
        # Precomputed volume for inside ellipsoid (from your original code)
        dtype = torch.get_default_dtype()
        domain_area_minus = torch.tensor(1.53379534665262, dtype=dtype)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        # Kept as provided (used by your loss)
        return torch.tensor(6.7083943643066, dtype=torch.get_default_dtype())

    def boundary_length(self):
        # Surface area of cube [-1,1]^3
        return torch.tensor(24.0, dtype=torch.get_default_dtype())

    # --------------------------- Indicator phi -------------------------
    def phi(self, p: torch.Tensor) -> torch.Tensor:
        """Return True for points outside the ellipsoid (plus region)."""
        x = p[..., 0]; y = p[..., 1]; z = p[..., 2]
        s = 2.0 * x**2 + 3.0 * y**2 + 6.0 * z**2 - 1.69
        return s.reshape(-1, 1) > 0

    # ----------------------- Diffusion coefficients --------------------
    def diffusion_coefficient_judge(self, p: torch.Tensor) -> torch.Tensor:
        bp = self.bp.to(dtype=p.dtype, device=p.device)
        bm = self.bm.to(dtype=p.dtype, device=p.device)
        return torch.where(self.phi(p), bp, bm)

    def diffusion_plus(self, p: torch.Tensor) -> torch.Tensor:
        m, _ = p.size()
        return self.bp.to(dtype=p.dtype, device=p.device).expand(m, 1).clone()

    def diffusion_minus(self, p: torch.Tensor) -> torch.Tensor:
        m, _ = p.size()
        return self.bm.to(dtype=p.dtype, device=p.device).expand(m, 1).clone()

    # ----------------------- Exact solutions & RHS ---------------------
    def exact_minus(self, p: torch.Tensor) -> torch.Tensor:
        r2 = torch.sum(p * p, dim=1, keepdim=True)
        return r2  # x^2 + y^2 + z^2

    def exact_plus(self, p: torch.Tensor) -> torch.Tensor:
        sol = torch.sum(p, dim=1, keepdim=True)  # x + y + z
        return sol

    def exact_judge(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))

    def rhs_plus(self, p: torch.Tensor) -> torch.Tensor:
        m, _ = p.size()
        return torch.zeros(m, 1, dtype=p.dtype, device=p.device)  # Δ(x+y+z)=0

    def rhs_minus(self, p: torch.Tensor) -> torch.Tensor:
        m, _ = p.size()
        bm = self.bm.to(dtype=p.dtype, device=p.device)
        return -6.0 * bm * torch.ones(m, 1, dtype=p.dtype, device=p.device)  # -β_- * 6

    def rhs_judge(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))

    # -------------------------- Boundary data -------------------------
    def BC_Dirichlet(self, p: torch.Tensor) -> torch.Tensor:
        return self.exact_judge(p)

    # ------------------------- Interface data -------------------------
    def interface_normal(self, p: torch.Tensor) -> torch.Tensor:
        """Unit normal of s(x,y,z)=2x^2+3y^2+6z^2-1.69 (outward of minus)."""
        x = p[..., 0:1]; y = p[..., 1:2]; z = p[..., 2:3]
        n_x, n_y, n_z = 4.0 * x, 6.0 * y, 12.0 * z
        length = torch.sqrt(n_x * n_x + n_y * n_y + n_z * n_z)
        return torch.cat((n_x / length, n_y / length, n_z / length), dim=1)

    def function_jump_p(self, p: torch.Tensor) -> torch.Tensor:
        return self.exact_plus(p) - self.exact_minus(p)

    def flux_jump_q(self, p: torch.Tensor) -> torch.Tensor:
        """Jump of normal flux: [β ∂u/∂n] = β_+ ∇u_+·n - β_- ∇u_-·n."""
        unv = self.interface_normal(p)               # (N,3)
        m, _ = p.size()
        grad_minus = 2.0 * p                         # ∇u_- = 2 p
        grad_plus  = torch.ones(m, 3, dtype=p.dtype, device=p.device)  # ∇u_+ = [1,1,1]
        bp = self.bp.to(dtype=p.dtype, device=p.device)
        bm = self.bm.to(dtype=p.dtype, device=p.device)
        jump = (torch.sum(grad_plus * unv, dim=1) * bp
               - torch.sum(grad_minus * unv, dim=1) * bm)
        return jump.reshape(-1, 1)

    # ------------------------- Point generators -----------------------
    def interior_points(self, N_omega: int = 1024) -> torch.Tensor:
        # Uniform in [-1,1]^3
        return 2 * torch.rand(N_omega, 3, dtype=torch.get_default_dtype()) - 1

    def boundary_points(self, N_b: int = 32) -> torch.Tensor:
        # Six faces of the cube [-1,1]^3
        dim = 3
        xb = 2 * torch.rand(2 * dim * N_b, dim, dtype=torch.get_default_dtype()) - 1
        for i in range(dim):
            xb[2 * i * N_b:(2 * i + 1) * N_b, i] = -1
            xb[(2 * i + 1) * N_b:(2 * i + 2) * N_b, i] = 1
        return xb

    def interface_points(self, N_f: int = 512) -> torch.Tensor:
        # Simple ellipsoid sampling via spherical angles (not area-uniform)
        dtype = torch.get_default_dtype()
        a = torch.sqrt(torch.tensor(1.69 / 2.0, dtype=dtype))
        b = torch.sqrt(torch.tensor(1.69 / 3.0, dtype=dtype))
        c = torch.sqrt(torch.tensor(1.69 / 6.0, dtype=dtype))
        theta = 2 * torch.pi * torch.rand(N_f, 1, dtype=dtype)
        phi   = torch.pi * torch.rand(N_f, 1, dtype=dtype)
        x = a * torch.cos(theta) * torch.sin(phi)
        y = b * torch.sin(theta) * torch.sin(phi)
        z = c * torch.cos(phi)
        return torch.cat((x, y, z), dim=1)