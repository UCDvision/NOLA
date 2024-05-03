import pdb

import torch
import torch.nn as nn
import torch.nn.init as init
import math
import numpy as np

from utils.utils import normalize_int, unnormalize_int


class GenTheta(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, c, out_dim, in_dim, seed):

        Bs, B = A.shape
        T = torch.zeros(Bs, out_dim, in_dim).to(A.device)
        l = B // c
        #         rand_seed = torch.randint(int(1e10), (1,))
        torch.manual_seed(seed)
        # pdb.set_trace()
        for i in range(c):
            A_ = A[:, i * l:(i + 1) * l]
            W_ = torch.zeros(l, out_dim, in_dim, device=A.device)
            init.kaiming_uniform_(W_, a=math.sqrt(5))
            # T += torch.einsum('nb,boi->noi', A_.type(torch.float16), W_.type(torch.float16)).float()
            T += torch.einsum('nb,boi->noi', A_, W_).float()

        params = torch.autograd.Variable(torch.tensor([c, out_dim, in_dim, seed]))
        ctx.save_for_backward(A, params)
        #         torch.manual_seed(rand_seed)
        return T

    @staticmethod
    def backward(ctx, grad_output):
        A, params = ctx.saved_tensors
        Bs, B = A.shape
        c, out_dim, in_dim, seed = params
        #         rand_seed = torch.randint(int(1e10), (1,))
        torch.manual_seed(seed)
        DA = torch.empty(Bs, 0).to(grad_output.device)
        l = B // c

        for i in range(c):
            W_ = torch.zeros(l, out_dim, in_dim, device=A.device)
            init.kaiming_uniform_(W_, a=math.sqrt(5))
            W_ = W_.permute(1, 2, 0)
            DA = torch.cat((DA, torch.einsum('nd,dl->nl', grad_output.flatten(1), W_.reshape(-1, l))), dim=1)

        #         torch.manual_seed(rand_seed)
        return DA, None, None, None, None

# 5717416, DeiT-Tiny
# .25 1736104
# .1
class Linear(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    # weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, param_ratio=0.25) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.genTetha = GenTheta.apply
        self.num_basis = int(in_features*out_features*param_ratio)
        self.alpha = nn.Parameter(torch.rand(self.num_basis)*0.1-0.05)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # self.alpha.uniform_(-0.05, -0.05)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            init.uniform_(self.bias, -bound, bound)


    # def forward(self, input: Tensor, seed) -> Tensor:
    def forward(self, input, seed):
        # pdb.set_trace()
        B = input.shape[0]
        alpha = self.alpha.unsqueeze(0).repeat(B,1)
        weight = self.genTetha(alpha, 1, self.out_features, self.in_features, seed)
        return (input @ weight.transpose(-1, -2)) + self.bias if self.bias is not None else (input @ weight.transpose(-1, -2))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, num_basis={}'.format(
            self.in_features, self.out_features, self.num_basis
        )
    
    
# NOLA (Optimize B):
    
    
class NOLA(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_basis = 10000, 
        **kwargs
    ):
        super(NOLA, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.N = in_features*out_features 
        self.rank = 16
        self.d = 128
        self.D = self.N//self.d
        scale_B = 1.0/self.rank
        scale_A = 1.0/self.num_basis
        self.c = 10
        self.scale = self.c*scale_B * scale_A
        
        self.perm = nn.Parameter(torch.tensor(np.random.permutation(self.N)), requires_grad=False)
        
        self.seed = nn.Parameter(torch.randint(int(1e10), (1,)), requires_grad=False)
        # torch.manual_seed(self.seed)
        self.A = nn.Parameter(torch.zeros(num_basis, self.D, self.rank), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(self.rank, self.d), requires_grad=False)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self.alpha = nn.Parameter(torch.zeros(num_basis))
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, num_basis={}'.format(
            self.in_features, self.out_features, self.num_basis)

    def forward(self, x: torch.Tensor):
        # pdb.set_trace()
        weight = torch.einsum('b,bdr->dr', self.alpha, self.A)
        weight = self.scale*(weight @ self.B)
        # 0.05 
        weight = weight.flatten()
        weight = weight[self.perm]
        weight = weight.reshape(self.in_features, self.out_features)
        
        return (x @ weight)  
    
    
# NOLA (Pranc B):
    
class NOLA2(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        ka=1024,
        kb=1024,
        rank=64,
        d=0,
        seed=None,
        **kwargs
    ):
        super(NOLA2, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.ka = ka
        self.kb = kb
        self.N = in_features*out_features 
        self.rank = rank
        self.d = d
        if d == 0:
            self.d = out_features
        self.D = self.N//self.d
        scale_B = 1.0/self.rank
        self.c = 1.0
        self.scale = self.c * scale_B

        # self.perm = nn.Parameter(torch.tensor(np.random.permutation(self.N)), requires_grad=False)

        if seed is None:
            self.seed = nn.Parameter(torch.randint(int(1e10), (1,)), requires_grad=False)
        else:
            self.seed = seed
        self.A = nn.Parameter(torch.zeros(self.ka, self.D, self.rank), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(self.kb, self.rank, self.d), requires_grad=False)

        # This seed must match in training and testing
        torch.manual_seed(0)

        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))

        self.nola_alpha = nn.Parameter(torch.ones(ka)*(1/self.ka), requires_grad=True)
        self.nola_beta = nn.Parameter(torch.ones(kb)*(1/self.kb), requires_grad=True)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, num_basis_A={}, num_basis_B={}'.format(
            self.in_features, self.out_features, self.ka, self.kb)

    def forward(self, x: torch.Tensor):
        # if self.training or dice:
        A = torch.einsum('b,bdr->dr', self.nola_alpha, self.A)
        B = torch.einsum('b,bdr->dr', self.nola_beta, self.B)

        out = self.scale * ((x @ A) @ B)
        return out
