import torch


# class Block(torch.nn.Module):
#     def __init__(self, in_N, width, out_N, phi=torch.nn.Tanh()):
#         super(Block, self).__init__()
#         self.L1 = torch.nn.Linear(in_N, width)
#         self.L2 = torch.nn.Linear(width, out_N)
#         self.phi = torch.nn.Tanh()
#         self.phi = phi
#
#     def forward(self, x):
#         return self.phi(self.L2(self.phi(self.L1(x)))) + x
#
#
# class ResNet(torch.nn.Module):
#     def __init__(self, in_N, m, out_N, depth=3, phi=torch.nn.Tanh()):
#         super(ResNet, self).__init__()
#         self.in_N = in_N
#         self.m = m
#         self.out_N = out_N
#         self.depth = depth
#         self.phi = phi
#         self.stack = torch.nn.ModuleList()
#         # first layer
#         self.stack.append(torch.nn.Linear(in_N, m))
#         # self.stack.append(self.phi)
#         # middle layers
#         for i in range(depth):
#             self.stack.append(Block(m, m, m))
#         # last layer
#         self.stack.append(torch.nn.Linear(m, out_N))
#
#     def forward(self, x):
#         for i in range(len(self.stack)):
#             x = self.stack[i](x)
#         return x

# in_N = 2
# m = 10
# out_N = 1
# depth = 3
# ResNet_omega1 = ResNet(in_N, m, out_N, depth=depth)
# ResNet_omega2 = ResNet(in_N, m, out_N, depth=depth)

# ResNet_omega1 = ResNet_omega1.double()
# ResNet_omega2 = ResNet_omega2.double()
class FCNet(torch.nn.Module):
    def __init__(self, in_N, m, out_N, depth, phi=torch.nn.Tanh()):
        super(FCNet, self).__init__()
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.stack = torch.nn.ModuleList()
        self.phi = phi
        # first layer
        self.stack.append(torch.nn.Linear(in_N, m))
        self.stack.append(self.phi)
        # middle layers
        for i in range(depth - 1):
            self.stack.append(torch.nn.Linear(m, m, m))
            self.stack.append(self.phi)
        # last layer
        self.stack.append(torch.nn.Linear(m, out_N))

    def forward(self, x):
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x