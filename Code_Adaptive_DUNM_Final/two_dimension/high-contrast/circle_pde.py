import torch
class circle_interface():
    def __init__(self, bp, bm, r0=0.5):
        self.beta_plus = torch.tensor(bp, dtype=torch.float64)
        self.beta_minus = torch.tensor(bm, dtype=torch.float64)
        self.domain_area = torch.tensor(4.0, dtype=torch.float64)
        self.r0 = torch.tensor(r0, dtype=torch.float64)

    def dimension(self):
        d = 2
        return d

    def subdomain_areas(self):
        domain_area_minus = torch.pi*torch.square(self.r0)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        interface_length = 2 * torch.pi * self.r0
        return interface_length

    def boundary_length(self):
        boundary_length = torch.tensor(8.0, dtype=torch.float64)
        return boundary_length
    '''Differentiate value inside and outside the interface'''

    def phi(self, p):
        r = torch.sum(torch.square(p), dim=1)
        s = r - torch.square(self.r0)
        s = s.reshape(-1, 1)
        return s > 0 #outside interface

    #meet the condition (in interface), return beta_plus, otherwise return beta_minus
    def diffusion_coefficient_judge(self, p): # #pass domain points
        judge_coefficient = torch.where(self.phi(p), self.beta_plus, self.beta_minus)
        return judge_coefficient

    def diffusion_plus(self, p): #p.shape(m, 2)
        m, n = p.size()
        beta_plus = self.beta_plus * torch.ones(m, 1)
        return beta_plus

    def diffusion_minus(self, p):
        m, n = p.size()
        beta_minus = self.beta_minus * torch.ones(m, 1)
        return beta_minus

    '''exact and rhs'''
    def exact_minus(self, p): #blue1
        #example 1
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        solution_minus = torch.pow(r, 3)/self.beta_minus
        solution_minus = solution_minus.reshape(-1, 1)
        #example 2
        # m, n = p.size()
        # solution_minus = torch.ones(m, 1)
        return solution_minus

    def exact_plus(self, p): #red2
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        solution_plus = (torch.pow(r, 3)/self.beta_plus +
                          (1/self.beta_minus - 1/self.beta_plus) *
                          torch.pow(self.r0, 3))

        # solution_plus = 1 + torch.log(2*r)
        solution_plus = solution_plus.reshape(-1, 1)

        # m, n = p.size()
        # solution_plus = torch.zeros(m, 1)
        return solution_plus

    def exact_judge(self, p):         #pass domain or boundary points
        judge_solution = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge_solution

    def rhs_plus(self, p): #subdomain 1 point

        #example 1
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        rhs_plus = -9 * r

        #example 2
        # m,n = p.size()
        # rhs_plus = torch.zeros(m, 1)
        return rhs_plus

    def rhs_minus(self, p):
        #example 1
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        rhs_minus = -9 * r
        #example 2
        # m, n = p.size()
        # rhs_minus = torch.zeros(m, 1)
        return rhs_minus

    def rhs_judge(self, p): #pass domain points
        rhs_judge = torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))
        return rhs_judge

    #BC conditions u = g
    def BC_Dirichlet(self, p):
        BC_real = self.exact_judge(p) #Pass in BC_points--judge--return exact_minus2
        # print(BC_real.shape)
        return BC_real

    '''Jump'''
    def interface_normal(self, p): #pass in: point on the interface
        len = torch.sqrt(torch.sum(torch.square(p), dim=1))
        len = len.reshape(-1, 1)
        # interface_normal = p/len
        #examle 2
        interface_normal = p/len
        return interface_normal

    def function_jump_p(self, p): #pass interface points
        m, n = p.size()
        jump_p = torch.zeros(m, 1)
        return jump_p

    def flux_jump_q(self, p): #pass interface points
        #example1
        m, n = p.size()
        jump_q = torch.zeros(m, 1)

        #example 2
        # unv = self.interface_normal(p)
        # r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        # r = r.reshape(-1, 1)
        # grad_plus = p * torch.pow(r, -2)
        # jump_q = torch.sum(grad_plus * unv, dim=1) * self.beta_plus
        # jump_q = jump_q.reshape(-1, 1)
        # print('jump_q.shape', jump_q.shape)
        # print(jump_q)
        # jump_q = 2 * self.diffusion_plus(p)
        # jump_q = jump_q.reshape(-1, 1)
        return jump_q

    '''Data Generation'''
    def interior_points(self, N_omega=1024):
        dimension = 2
        xr = torch.rand(N_omega, dimension) * 2 - 1 #(N_omega, 2)
        return xr

    def boundary_points(self, N_b=64):
        dimension = 2
        xb = torch.rand(2*dimension*N_b, dimension) * 2 - 1 #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b:(2*i+1)*N_b, i] = -1 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b:(2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    def interface_points(self, N_f=128):
        theta = 2*torch.pi*torch.rand(N_f,1)
        x = self.r0 * torch.cos(theta)
        y = self.r0 * torch.sin(theta)
        xf = torch.cat((x, y), dim=1)
        return xf