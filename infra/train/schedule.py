# LR schedules defined on run progress (0..1), returning a multiplier of
# the peak lr. Progress-fraction parameterization means changing batch
# size or token budget rescales the schedule instead of silently
# deforming it, and shortening a run (marin's contingency move) is just
# a smaller steps_total.

import math
from functools import partial
from typing import Callable


def cosine(progress: float, *, warmup: float = 0.01, min_ratio: float = 0.0) -> float:
    '''Linear warmup then cosine decay to min_ratio. The Transformer++
    benchmark shape (Mamba App. E.2: cosine to a small floor).'''
    if progress < warmup:
        return progress / warmup
    p = (progress - warmup) / (1 - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def linear(progress: float, *, warmup: float = 0.01, min_ratio: float = 0.05) -> float:
    '''Linear warmup then linear decay to min_ratio. The marin recipe shape.'''
    if progress < warmup:
        return progress / warmup
    p = (progress - warmup) / (1 - warmup)
    return 1 + (min_ratio - 1) * p


def wsd(progress: float, *, warmup: float = 0.01, decay: float = 0.2,
        min_ratio: float = 0.0) -> float:
    '''Warmup-Stable-Decay: flat at peak, linear decay over the last
    `decay` fraction. The stable-phase checkpoints can branch into
    multiple cooldowns, so one pretrain serves several experiments.'''
    if progress < warmup:
        return progress / warmup
    if progress < 1 - decay:
        return 1.0
    p = (progress - (1 - decay)) / decay
    return 1 + (min_ratio - 1) * p


SCHEDULES = {'cosine': cosine, 'linear': linear, 'wsd': wsd}


def build_schedule(name: str, args: dict) -> Callable[[float], float]:
    fn = partial(SCHEDULES[name], **args)
    for p in (0.0, 0.5, 1.0):   # fail loudly now on bad args, not at step 1
        fn(p)
    return fn
