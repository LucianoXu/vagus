import torch
from torch import nn

try:
    from fla.modules.conv.causal_conv1d import causal_conv1d  # type: ignore
    HAS_FLA_CONV = True
except ImportError:   # CUDA/Triton-only package, like the fla scan kernels
    causal_conv1d = None  # type: ignore[assignment]
    HAS_FLA_CONV = False


class ShortConv(nn.Module):
    """Depthwise causal 1-D conv over the sequence (fla-style short conv).

    Two paths, one semantics and one parameter (the nn.Conv1d weight,
    (D, 1, K), so a checkpoint moves between them unchanged):

      - nn.Conv1d on a left-padded transpose. Portable; on CUDA it is a
        cuDNN depthwise kernel over a (B, D, L+K-1) copy, three tensor
        round trips (pad, conv, transpose) plus a separate activation.
      - fla's causal_conv1d Triton kernel, which reads (B, L, D)
        directly, needs no padded copy and folds the activation in.
        Used when `fused=True` and an `activation` is passed, on a
        CUDA half-precision input; it is the same convolution, so the
        two agree up to bf16 accumulation order. Off by default: a
        caller that has not measured it keeps the portable path.

    `activation` ('silu' or None) is applied after the convolution
    either way — passing it here rather than calling F.silu on the
    result is what lets the fused kernel absorb it.
    """

    def __init__(self, dim: int, kernel: int, fused: bool = False):
        super().__init__()

        assert dim > 0
        assert kernel > 0

        self.kernel = kernel
        self.fused = fused
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def _fused_ok(self, x, activation) -> bool:
        return (self.fused and HAS_FLA_CONV and activation is not None
                and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16))

    def forward(self, x, activation: str | None = None):
        # x: (B, L, D) -> causal pad on the left
        if self._fused_ok(x, activation):
            assert causal_conv1d is not None
            y, _ = causal_conv1d(x, self.conv.weight.squeeze(1), activation=activation)
            return y
        y = self.conv(nn.functional.pad(x.transpose(1, 2), (self.kernel - 1, 0)))
        y = y.transpose(1, 2)
        return _act(y, activation)

    def direct_conv(self, x, activation: str | None = None):
        '''
        Direct conv without padding.
        x: (B, L, D)
        '''
        y = self.conv(x.transpose(1, 2))
        return _act(y.transpose(1, 2), activation)


def _act(y, activation: str | None):
    if activation is None:
        return y
    assert activation in ('silu', 'swish'), activation
    return nn.functional.silu(y)
