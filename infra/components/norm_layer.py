import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        assert dim > 0
        assert eps > 0

        self.eps = eps
        self.dim = dim
        self.gamma = nn.Parameter(data = torch.ones(self.dim))

    def forward(self, x):
        # x : (..., d)
        orig_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms 
        x = x.to(orig_dtype)
        x = x * self.gamma
        return x
