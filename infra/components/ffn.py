
import torch
from torch import nn

import math

class FFN(nn.Module):
    '''
    FFN using SwiGLU gating.
    '''
    def __init__(
            self, 
            dim: int, 
            hidden_dim: int,
            init_std: float = 0.02,
            layer_count: int | None = None,
        ):
        super().__init__()

        assert dim > 0
        assert hidden_dim > 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.l1 = nn.Linear(
            in_features=self.dim,
            out_features=2*self.hidden_dim,
            bias = False
        )

        # 2*self.hidden_dim output incorporates the gating multiplication in one kernel

        self.silu = nn.SiLU()

        self.l2 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.dim,
            bias = False
        ) 

        # Transformer++ init
        nn.init.normal_(self.l1.weight, std=init_std)
        l2_std = init_std / math.sqrt(2 * layer_count) if layer_count else init_std
        nn.init.normal_(self.l2.weight, std=l2_std)


    def forward(self, x):
        x1, xgate = torch.chunk(self.l1(x), 2, dim=-1)
        x = self.silu(x1) * xgate
        x = self.l2(x)
        return x
