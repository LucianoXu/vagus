
import torch
from torch import nn

class ShortConv(nn.Module):
    """Depthwise causal 1-D conv over the sequence (fla-style short conv)."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()

        assert dim > 0
        assert kernel > 0

        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x):
        # x: (B, L, D) -> causal pad on the left
        y = self.conv(nn.functional.pad(x.transpose(1, 2), (self.kernel - 1, 0)))
        return y.transpose(1, 2)

    def direct_conv(self, x):
        '''
        Direct conv without padding.
        x: (B, L, D)
        '''
        y = self.conv(x.transpose(1, 2))
        return y.transpose(1, 2)