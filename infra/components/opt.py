import torch
from torch import nn
from torch.nn import functional as F

try:
    from fla.modules.conv.causal_conv1d import causal_conv1d  # type: ignore
    HAS_FLA_CONV = True
except ImportError:   # CUDA/Triton-only package, like the fla scan kernels
    causal_conv1d = None  # type: ignore[assignment]
    HAS_FLA_CONV = False

CONV_IMPLS = ('conv1d', 'shift', 'fla')


class ShortConv(nn.Module):
    """Depthwise causal 1-D conv over the sequence (fla-style short conv).

    Three paths, one semantics and one parameter (the nn.Conv1d weight,
    (D, 1, K), so a checkpoint moves between them unchanged):

      'conv1d'  nn.Conv1d on a left-padded transpose. Portable, and the
                obvious way to write it, but on CUDA it is a cuDNN
                depthwise call over a (B, D, L+K-1) copy with a
                transpose either side — four tensor round trips for K
                multiply-adds per element. In a compiled HAX1 step the
                convolution and its backward are the largest non-GEMM
                item in the profile, ahead of the KDA scan kernels.
      'shift'   the same convolution written as what it is: K shifted
                multiply-adds over (B, L, D). No extern call, no padded
                copy, no transpose — and being plain elementwise work it
                fuses with the SiLU, the reshape and the L2 norm around
                it, so inductor emits one kernel where 'conv1d' needed
                several plus a cuDNN launch. Reads x K times instead of
                once, but the shifted reads are adjacent and stay in
                cache. Meant for the compiled path; in eager the K pads
                are real copies.
      'fla'     fla's causal_conv1d Triton kernel, which reads
                (B, L, D) directly and folds the activation in. Needs
                CUDA + half precision + an `activation`. Measured slower
                end to end than 'conv1d' under torch.compile (the custom
                autograd Function breaks the graph where a cuDNN call
                did not), so it is not the default.

    `activation` ('silu' or None) is applied after the convolution on
    every path — passing it here rather than calling F.silu on the
    result is what lets 'fla' absorb it, and costs the others nothing.
    """

    def __init__(self, dim: int, kernel: int, impl: str = 'conv1d'):
        super().__init__()

        assert dim > 0
        assert kernel > 0
        assert impl in CONV_IMPLS, f'{impl!r} not in {CONV_IMPLS}'

        self.kernel = kernel
        self.impl = impl
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x, activation: str | None = None):
        '''x: (B, L, D) -> (B, L, D), causal (output t sees inputs <= t).'''
        if (self.impl == 'fla' and HAS_FLA_CONV and activation is not None
                and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
            assert causal_conv1d is not None
            y, _ = causal_conv1d(x, self.conv.weight.squeeze(1), activation=activation)
            return y
        if self.impl == 'shift':
            return _act(self._shift(x), activation)
        y = self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0)))
        return _act(y.transpose(1, 2), activation)

    def _shift(self, x):
        '''y[t] = sum_i w[:, K-1-i] * x[t-i], i = 0..K-1 — the same sum
        nn.Conv1d computes over the left-padded input.'''
        w = self.conv.weight.squeeze(1).to(x.dtype)          # (D, K)
        K = self.kernel
        y = x * w[:, -1]
        for i in range(1, K):
            y = y + F.pad(x[:, :-i], (0, 0, i, 0)) * w[:, K - 1 - i]
        return y

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
    return F.silu(y)
