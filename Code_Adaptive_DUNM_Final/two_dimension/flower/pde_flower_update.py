import torch


interface_constant = 0.5

class flowerInterface:
    """
    Diffusion-form PDE:  -div( beta * grad u ) = f

    Regions:
      Outside (phi(p) == True):   u_plus(r)  = (0.1 r^4 - 0.01 log(2r)) / beta_plus
      Inside  (phi(p) == False):  u_minus(r) = exp(r^2)

    In 2D:
      Δu_plus  = (1.6 / beta_plus) r^2  =>  f_plus  = - beta_plus * Δu_plus  = -1.6 r^2   (β cancels)
      Δu_minus = 4(1+r^2)e^{r^2}        =>  f_minus = - beta_minus * Δu_minus = -4 * beta_minus * (1+r^2) e^{r^2}
    """

    # ------------------------ Init & constants ------------------------
    def __init__(self, bp, bm):
        # Keep dtype consistent with your model's default (float32 unless changed)
        dtype = torch.get_default_dtype()
        self.beta_plus  = torch.tensor(bp, dtype=dtype)
        self.beta_minus = torch.tensor(bm, dtype=dtype)
        self.domain_area = torch.tensor(4.0, dtype=dtype)  # square [-1,1]^2

    def dimension(self):
        return 2

    def subdomain_areas(self):
        # area inside flower + tiny petal correction (original choice)
        r0 = torch.tensor(0.5, dtype=torch.get_default_dtype())
        domain_area_minus = torch.pi * r0 * r0 + torch.pi / 98
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        # Precomputed length (matches your previous value)
        return torch.tensor(4.40279704689901, dtype=torch.get_default_dtype())

    def boundary_length(self):
        return torch.tensor(8.0, dtype=torch.get_default_dtype())  # perimeter of square [-1,1]^2

    # ------------------------ Geometry / Phase ------------------------
    def phi(self, p):
        """
        Return True (outside) if r^2 > r0(θ)^2 where r0(θ) = 0.5 + sin(5θ)/7.
        Uses r^2 for branchless thresholding.
        """
        x = p[..., 0]
        y = p[..., 1]
        r2 = (p * p).sum(dim=1)                    # r^2
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        r0 = interface_constant + torch.sin(5 * theta) / 7
        s = r2 - (r0 * r0)
        return s.reshape(-1, 1) > 0  # outside interface

    # ------------------------ Diffusion coefficients ------------------------
    def diffusion_coefficient_judge(self, p):
        # Outside -> beta_plus, inside -> beta_minus
        return torch.where(self.phi(p), self.beta_plus, self.beta_minus)

    def diffusion_plus(self, p):
        m, _ = p.size()
        return self.beta_plus.to(dtype=p.dtype, device=p.device).expand(m, 1).clone()

    def diffusion_minus(self, p):
        m, _ = p.size()
        return self.beta_minus.to(dtype=p.dtype, device=p.device).expand(m, 1).clone()

    # ------------------------ Exact solutions ------------------------
    def exact_minus(self, p):
        r2 = (p * p).sum(dim=1, keepdim=True)
        sol = torch.exp(r2)
        return sol

    def exact_plus(self, p):
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        r4 = torch.pow(r, 4)
        plus = (r4 - 0.1 * torch.log(2 * r)) / self.beta_plus
        return  plus

    def exact_judge(self, p):
        return torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))

    # ------------------------ Right-hand sides (diffusion form) ------------------------
    def rhs_minus(self, p):
        """
        f_minus = - beta_minus * Δu_minus, with Δu_minus = 4 (1 + r^2) e^{r^2}
        """
        r2 = (p * p).sum(dim=1, keepdim=True)
        beta_m = self.beta_minus.to(dtype=p.dtype, device=p.device)
        return -4.0 * beta_m * (1.0 + r2) * torch.exp(r2)

    def rhs_plus(self, p):
        r2 = torch.sum(torch.square(p), dim=1)
        r2 = r2.reshape(-1, 1)
        plus = -16 * r2
        return plus

    def rhs_judge(self, p):
        return torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))

    # ------------------------ Boundary condition ------------------------
    def BC_Dirichlet(self, p):
        # Dirichlet: u = exact solution on boundary
        return self.exact_judge(p)

    # ------------------------ Interface quantities ------------------------
    def interface_normal(self, p):
        """
        Unit normal for r(θ) = 1/2 + sin(5θ)/7 (points outward of the inner region).
        """
        x = p[..., 0]
        y = p[..., 1]
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        theta = theta.reshape(-1, 1)

        r = interface_constant + torch.sin(5 * theta) / 7
        dr_dtheta = (5.0 / 7.0) * torch.cos(5 * theta)
        n_x = dr_dtheta * torch.cos(theta) - r * torch.sin(theta)
        n_y = dr_dtheta * torch.sin(theta) + r * torch.cos(theta)
        length = torch.sqrt(n_x * n_x + n_y * n_y)
        return torch.cat((n_y / length, -n_x / length), dim=1)  # rotate tangent + normalize

    def flux_jump_q(self, p):
        """
        Jump of normal flux: [β ∂u/∂n] = β_plus * grad(u_plus)·n - β_minus * grad(u_minus)·n
        """
        unv = self.interface_normal(p)                   # (N,2)
        r2 = (p * p).sum(dim=1, keepdim=True)
        r = torch.sqrt(r2)

        # grad u_plus = [(0.4 r^2 - 0.01 r^{-2}) * p] / beta_plus  (since u_plus includes 1/beta_plus)
        beta_p = self.beta_plus.to(dtype=p.dtype, device=p.device)
        grad_plus = (4 * p * r2 - 0.1 * p * torch.pow(r, -2)) / beta_p

        beta_m = self.beta_minus.to(dtype=p.dtype, device=p.device)

        # grad u_minus = 2 p e^{r^2}
        grad_minus = 2.0 * p * torch.exp(r2)

        jump_q = (torch.sum(grad_plus * unv, dim=1) * beta_p -
                  torch.sum(grad_minus * unv, dim=1) * beta_m)
        return jump_q.reshape(-1, 1)

    def function_jump_p(self, p):
        return self.exact_plus(p) - self.exact_minus(p)

    # ------------------------ Point sampling ------------------------
    def interior_points(self, N_omega=1024):
        # Uniform in [-1,1]^2
        return 2 * torch.rand(N_omega, 2) - 1

    def boundary_points(self, N_b=32):
        # Four edges of the square [-1,1]^2
        xb = 2 * torch.rand(4 * N_b, 2) - 1
        # left/right edges for x, bottom/top for y
        xb[0 * N_b:1 * N_b, 0] = -1  # left
        xb[1 * N_b:2 * N_b, 0] =  1  # right
        xb[2 * N_b:3 * N_b, 1] = -1  # bottom
        xb[3 * N_b:4 * N_b, 1] =  1  # top
        return xb

    def interface_points(self, N_f=256):
        theta = 2 * torch.pi * torch.rand(N_f, 1)
        r = interface_constant + torch.sin(5 * theta) / 7
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.cat((x, y), dim=1)