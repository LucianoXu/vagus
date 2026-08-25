'''
Triton implementation of the interleaved-pair RoPE forward, numerically
matching infra/components/pos_embed.RoPE:

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

Requires CUDA + triton; import HAS_TRITON to test availability before use.
'''

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rope_fwd(x_ptr, cos_ptr, sin_ptr, out_ptr,
                  L, arm, D, BLOCK: tl.constexpr):
        # one program per row of length D; rows are laid out (..., L, D)
        # contiguously, so the position index is row % L
        row = tl.program_id(0)
        pos = row % L
        base = row * D

        offs = tl.arange(0, BLOCK)
        mask = offs < arm

        x0 = tl.load(x_ptr + base + 2 * offs, mask=mask)
        x1 = tl.load(x_ptr + base + 2 * offs + 1, mask=mask)
        c = tl.load(cos_ptr + pos * arm + offs, mask=mask)
        s = tl.load(sin_ptr + pos * arm + offs, mask=mask)

        tl.store(out_ptr + base + 2 * offs, x0 * c - x1 * s, mask=mask)
        tl.store(out_ptr + base + 2 * offs + 1, x0 * s + x1 * c, mask=mask)


def rope_triton(x: torch.Tensor, mcos: torch.Tensor, msin: torch.Tensor) -> torch.Tensor:
    '''x: (..., L, D) on cuda; mcos/msin: (>=L, D//2) as prepared by RoPE.'''
    assert HAS_TRITON, 'triton is not installed'
    assert x.is_cuda, 'rope_triton requires a CUDA tensor'

    L, D = x.shape[-2], x.shape[-1]
    arm = D // 2
    orig_shape = x.shape

    x = x.contiguous().view(-1, D)
    mcos = mcos[:L].contiguous()
    msin = msin[:L].contiguous()
    out = torch.empty_like(x)

    n_rows = x.shape[0]
    BLOCK = triton.next_power_of_2(arm)
    _rope_fwd[(n_rows,)](x, mcos, msin, out, L, arm, D, BLOCK=BLOCK)
    return out.view(orig_shape)
