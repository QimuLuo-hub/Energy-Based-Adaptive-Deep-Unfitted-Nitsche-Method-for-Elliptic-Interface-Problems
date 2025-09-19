import torch

class popcorn_interface:
    """
    3D diffusion-form PDE:  -div( beta * grad u ) = f

    Geometry ("popcorn" surface):
      Level set g(p) = ||p|| - (r0 + S(p)) = 0,
      where S(p) = sum_k A * exp(-||p - p_k||^2 / sigma^2), k=1,...,12 precomputed centers.
      phi(p) > 0 denotes OUTSIDE (plus) region.

    Exact solutions:
      Outside (plus): u_+(x,y,z) = x + y + z        => Δu_+ = 0      => f_+ = 0
      Inside  (minus): u_-(x,y,z) = x^2+y^2+z^2     => Δu_- = 6      => f_- = -6 * β_-

    Notes:
      • All outputs preserve input dtype/device.
      • β_+, β_- stored as tensors with torch's default float dtype.
    """

    def __init__(self, bp: float = 60.0, bm: float = 2.0, dim: int = 3):
        dtype = torch.get_default_dtype()
        self.dim = dim
        self.bp = torch.tensor(bp, dtype=dtype)
        self.bm = torch.tensor(bm, dtype=dtype)
        self.domain_area = (torch.tensor(2.0, dtype=dtype) ** dim)

        # Interface parameters
        self.r0 = torch.tensor(0.6, dtype=dtype)
        self.A = torch.tensor(2.0, dtype=dtype)
        self.sigma = torch.tensor(0.2, dtype=dtype)
        self.factor = self.r0 / torch.sqrt(torch.tensor(5.0, dtype=dtype))

        # Precompute centers
        self.angles_first = [torch.tensor(2 * k * torch.pi / 5, dtype=dtype) for k in range(5)]
        self.angles_second = [torch.tensor((2 * k - 11) * torch.pi / 5, dtype=dtype) for k in range(5, 10)]
        self.precomputed_points = self._precompute_points()

    # --------------------------- Basics ---------------------------
    def dimension(self): return 3

    def subdomain_areas(self):
        dtype = torch.get_default_dtype()
        domain_area_minus = torch.tensor(1.93776, dtype=dtype)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        return torch.tensor(6.73656, dtype=torch.get_default_dtype())

    def boundary_length(self):
        return torch.tensor(24.0, dtype=torch.get_default_dtype())

    # ---------------------- Geometry helpers ----------------------
    def _point_k(self, k: int):
        if 0 <= k <= 4:
            angle = self.angles_first[k]
            return torch.stack([self.factor * 2 * torch.cos(angle),
                                self.factor * 2 * torch.sin(angle),
                                self.factor * 1])
        elif 5 <= k <= 9:
            angle = self.angles_second[k - 5]
            return torch.stack([self.factor * 2 * torch.cos(angle),
                                self.factor * 2 * torch.sin(angle),
                                -self.factor * 1])
        elif k == 10:
            return torch.tensor([0.0, 0.0, self.r0], dtype=self.r0.dtype)
        elif k == 11:
            return torch.tensor([0.0, 0.0, -self.r0], dtype=self.r0.dtype)
        else:
            raise ValueError("k must be between 0 and 11")

    def _precompute_points(self):
        return torch.stack([self._point_k(k) for k in range(12)], dim=0)

    def S(self, p: torch.Tensor) -> torch.Tensor:
        centers = self.precomputed_points.to(dtype=p.dtype, device=p.device)
        diffs = p.unsqueeze(1) - centers.unsqueeze(0)
        sq = torch.sum(diffs * diffs, dim=2)
        A, sig = self.A.to(p), self.sigma.to(p)
        return (A * torch.exp(-sq / (sig ** 2))).sum(dim=1, keepdim=True)

    def phi(self, p: torch.Tensor) -> torch.Tensor:
        r = torch.norm(p, dim=1, keepdim=True)
        s = r - self.r0.to(p) - self.S(p)
        return s > 0

    # ------------------ Diffusion coefficients --------------------
    def diffusion_coefficient_judge(self, p: torch.Tensor) -> torch.Tensor:
        bp = self.bp.to(dtype=p.dtype, device=p.device)
        bm = self.bm.to(dtype=p.dtype, device=p.device)
        return torch.where(self.phi(p), bp, bm)

    def diffusion_plus(self, p: torch.Tensor) -> torch.Tensor:
        return self.bp.to(dtype=p.dtype, device=p.device).expand(p.size(0), 1).clone()

    def diffusion_minus(self, p: torch.Tensor) -> torch.Tensor:
        return self.bm.to(dtype=p.dtype, device=p.device).expand(p.size(0), 1).clone()

    # ---------------------- Exact & RHS (3D) ----------------------
    def exact_minus(self, p: torch.Tensor) -> torch.Tensor:
        return torch.sum(p * p, dim=1, keepdim=True)

    def exact_plus(self, p: torch.Tensor) -> torch.Tensor:
        return torch.sum(p, dim=1, keepdim=True)

    def exact_judge(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))

    def rhs_plus(self, p: torch.Tensor) -> torch.Tensor:
        return torch.zeros(p.size(0), 1, dtype=p.dtype, device=p.device)

    def rhs_minus(self, p: torch.Tensor) -> torch.Tensor:
        bm = self.bm.to(dtype=p.dtype, device=p.device)
        return -6.0 * bm * torch.ones(p.size(0), 1, dtype=p.dtype, device=p.device)

    def rhs_judge(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))

    # ---------------------- Boundary condition --------------------
    def BC_Dirichlet(self, p: torch.Tensor) -> torch.Tensor:
        return self.exact_judge(p)

    # ---------------------- Interface quantities ------------------
    def interface_normal(self, p: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(p.dtype).eps
        r = torch.clamp(torch.norm(p, dim=1, keepdim=True), min=eps)
        centers = self.precomputed_points.to(dtype=p.dtype, device=p.device)
        diffs = p.unsqueeze(1) - centers.unsqueeze(0)
        sq = torch.sum(diffs * diffs, dim=2)
        A, sig2 = self.A.to(p), (self.sigma.to(p) ** 2)
        exp_terms = torch.exp(-sq / sig2)
        grad_S = (2.0 * A * exp_terms).unsqueeze(-1) * diffs / sig2
        grad_S = grad_S.sum(dim=1)
        n = p / r - grad_S
        return n / torch.clamp(torch.norm(n, dim=1, keepdim=True), min=eps)

    def function_jump_p(self, p: torch.Tensor) -> torch.Tensor:
        return self.exact_plus(p) - self.exact_minus(p)

    def flux_jump_q(self, p: torch.Tensor) -> torch.Tensor:
        unv = self.interface_normal(p)
        grad_minus = 2.0 * p
        grad_plus = torch.ones(p.size(0), 3, dtype=p.dtype, device=p.device)
        bp = self.bp.to(dtype=p.dtype, device=p.device)
        bm = self.bm.to(dtype=p.dtype, device=p.device)
        jump = (torch.sum(grad_plus * unv, dim=1) * bp - torch.sum(grad_minus * unv, dim=1) * bm)
        return jump.reshape(-1, 1)

    # ---------------------- Point generation ----------------------
    def interior_points(self, N_omega: int) -> torch.Tensor:
        return 2 * torch.rand(N_omega, 3, dtype=torch.get_default_dtype()) - 1

    def boundary_points(self, N_b: int) -> torch.Tensor:
        dim = 3
        xb = 2 * torch.rand(2 * dim * N_b, dim, dtype=torch.get_default_dtype()) - 1
        for i in range(dim):
            xb[2*i*N_b:(2*i+1)*N_b, i] = -1
            xb[(2*i+1)*N_b:(2*i+2)*N_b, i] = 1
        return xb

    def interface_points(self, N_f: int) -> torch.Tensor:
        dtype = torch.get_default_dtype()
        device = self.bp.device
        target = int(N_f)
        band = 0.1
        batch = max(4 * target, 2048)
        pts, total, max_batches = [], 0, 30
        for _ in range(max_batches):
            p = (2 * torch.rand(batch, 3, dtype=dtype, device=device) - 1)
            r = torch.norm(p, dim=1, keepdim=True)
            Sval = self.S(p)
            mask = (torch.abs(r - self.r0.to(p) - Sval) < band).squeeze(1)
            sel = p[mask]
            if sel.numel() > 0:
                pts.append(sel); total += sel.size(0)
                if total >= target: break
        if len(pts) == 0:
            return 2 * torch.rand(target, 3, dtype=dtype, device=device) - 1
        out = torch.cat(pts, dim=0)
        if out.size(0) > target:
            idx = torch.randperm(out.size(0), device=out.device)[:target]
            out = out[idx]
        return out