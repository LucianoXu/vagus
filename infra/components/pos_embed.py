
import torch
from torch import nn
from torch.nn import functional as F


class RoPE(nn.Module):
    '''
    Share the module to avoid redundancy.
    '''
    mcos: torch.Tensor
    msin: torch.Tensor

    def __init__(self, dim: int, head_dim: int, context_len: int):
        super().__init__()

        assert dim % head_dim == 0
        assert head_dim > 0
        assert head_dim % 2 == 0
        assert context_len > 0

        self.dim = dim
        self.context_len = context_len
        self.head_dim = head_dim

        self.arm_dim = self.head_dim // 2

        self.prepared_L = 0

        self.register_buffer('mcos', torch.zeros(()), persistent=False)
        self.register_buffer('msin', torch.zeros(()), persistent=False)

        self.prepare_m(self.context_len)


    def prepare_m(self, L: int):

        if L > self.prepared_L:
            device : torch.device = self.mcos.device
            dtype : torch.dtype = self.mcos.dtype

            # phase must be computed in float32
            idxk = torch.arange(0, self.arm_dim, device=device, dtype=torch.float32) / self.arm_dim
            phase = torch.outer(torch.arange(0, L, device=device, dtype=torch.float32), torch.pow(10000, -idxk,))

            # mcos : (L, dim/2)
            self.register_buffer('mcos', torch.cos(phase).to(dtype), persistent=False)

            # msin : (L, dim/2)
            self.register_buffer('msin', torch.sin(phase).to(dtype), persistent=False)

            self.prepared_L = L


    def forward(self, x: torch.Tensor):

        assert x.shape[-1] == self.head_dim
        
        # x : (..., l, d)

        if x.shape[-2] > self.prepared_L:
            raise ValueError("Context length exceeded. Please call RoPE.prepare_m to enlarge.")

        # slice the matrix
        mcos = self.mcos[:x.shape[-2], ...]
        msin = self.msin[:x.shape[-2], ...]

        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x = x * mcos[..., None] + torch.stack((-x[..., 1], x[..., 0]), dim=-1) * msin[..., None]
        x = x.reshape(*x.shape[:-2], -1)

        return x