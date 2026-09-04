# Checkpoint I/O for the two on-disk formats the project produces.
#
# full  — what train/main.py writes (ckpt-STEP.pt, recent.pt): model +
#         optimizer + loader + rng + resolved config. Resume material.
# slim  — model-final.pt: model_name, model_args, model (state_dict), step,
#         tokens_seen. Inference and archival material; ~1/3 the bytes.
#
# Both share model_name / model_args / model, so load_model accepts either
# and the generator never sees the difference. export_slim is the path
# from full to slim (the training loop calls it on its final checkpoint;
# it also serves older runs whose model-final.pt was made by hand).

from pathlib import Path

import torch
from torch import nn

from . import build_model
from ..utils import atomic_write

SLIM_KEYS = ('model_name', 'model_args', 'model', 'step', 'tokens_seen')

_DTYPES = {
    'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16,
}


def _as_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    return _DTYPES[dtype]


def is_full_checkpoint(state: dict) -> bool:
    return 'optimizer' in state


def load_checkpoint(path: str | Path, map_location='cpu') -> dict:
    '''The raw dict of either format. weights_only=False because the full
    format carries numpy RNG state and the loader state.'''
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model(path: str | Path, device: str | torch.device = 'cpu',
               dtype: str | torch.dtype | None = None) -> tuple[nn.Module, dict]:
    '''Rebuild the model from a full or slim checkpoint.

    Returns (model, meta) with meta = {model_name, model_args, step,
    tokens_seen, format}. dtype None keeps the stored precision (fp32
    master weights); 'bfloat16' casts for inference. The model comes back
    in eval mode with grads enabled — the generator runs under no_grad,
    and continual learning can train it as is.'''
    state = load_checkpoint(path)
    model = build_model(state['model_name'], state['model_args'])
    model.load_state_dict(state['model'])
    dt = _as_dtype(dtype)
    model = model.to(device=device, dtype=dt) if dt is not None else model.to(device)
    model.eval()
    meta = {
        'model_name': state['model_name'],
        'model_args': state['model_args'],
        'step': state.get('step'),
        'tokens_seen': state.get('tokens_seen'),
        'format': 'full' if is_full_checkpoint(state) else 'slim',
    }
    return model, meta


def slim_state(state: dict, dtype: str | torch.dtype | None = None) -> dict:
    '''The slim dict of a full (or slim) checkpoint dict, optionally
    casting the floating-point weights.'''
    dt = _as_dtype(dtype)
    weights = {
        k: (v.to(dt) if dt is not None and v.is_floating_point() else v)
        for k, v in state['model'].items()
    }
    out = {k: state[k] for k in SLIM_KEYS if k in state and k != 'model'}
    out['model'] = weights
    return out


def export_slim(src: str | Path, dst: str | Path,
                dtype: str | torch.dtype | None = None) -> Path:
    '''Write the slim checkpoint of `src` to `dst` (atomic). dtype None
    keeps fp32 — the lossless archival form; pass 'bfloat16' for a
    half-size inference copy.'''
    state = load_checkpoint(src)
    dst = Path(dst)
    atomic_write(dst, lambda f: torch.save(slim_state(state, dtype), f))
    return dst
