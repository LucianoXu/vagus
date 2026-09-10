# The procedures that turn (w, m) into w*. Both work on the subject's
# model in place — the caller snapshots the weights before and restores
# them after (the eval task does) — and return a witness of what they
# did.
#
# replay_kl — the definition made an algorithm:
#   1. read X on a fresh stream; m = the state that leaves
#   2. replay: K continuations of N tokens sampled from (w, m)
#      (the model dreaming on from its memory)
#   3. teacher logits of every replay token under (w, m), fixed
#   4. student: the same model on a fresh stream (m0), reading
#      start_id + replay; minimise KL(teacher || student) per token,
#      mixed with the ordinary LM loss on windows of the training store
#      (weight lm_mix), for `steps` AdamW steps over all weights
#   Replay tokens after a BOS the model emitted are masked: the document
#   the memory conditions ended there.
#
#   Multi-hop (hops = levels of the student's memory, ending at 0): the
#   memory is removed in steps m -> m' -> ... -> m0 (memory.py chooses
#   how — uniform scaling, layer by layer, by singular value), the
#   student at each hop reads the replay from the degraded memory
#   through the differentiable forward (GDNLM.forward(states=...)) and
#   is trained to match the teacher; the last hop is the blank-memory
#   student above. teacher='fixed' keeps (w, m) as the teacher
#   throughout; 'chain' re-samples the replay at each hop from the
#   current weights at the previous hop's memory, the literal reading
#   of "compare (w, m) with (w', m')" hop by hop.
#
# ntp_x — the baseline consolidation must beat: the same optimiser
#   budget spent on next-token loss over X itself (the context read
#   with no memory involved), same LM mix. If replay_kl cannot beat
#   this, the memory contributed nothing beyond the text it saw.
#
# Two regularisers keep the model from collapsing onto one document:
#   lm_mix     plain LM loss on random windows of the training store
#   retain_kl  KL(original model || student) on random windows of the
#              store, the original weights kept as a frozen teacher —
#              "change nothing the context did not ask for", a tighter
#              anchor than the LM loss, which only says "stay good".
#
# lr_schedule: const, or cosine / linear decay to zero after `warmup`
# steps (a per-item fine-tune is a small training run; a decay lets a
# larger peak lr end at a settled point).
#
# Weights train in fp32 (an AdamW step at lr 1e-5 vanishes in bf16),
# under bf16 autocast on CUDA; the model returns to its original dtype
# before the caller scores it.

import copy
import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..dataset.loader import TokenStore
from ..inference import Generator, SamplingConfig
from ..optimizer import build_adamw
from .memory import DEGRADATIONS, degrade

METHODS = ('replay_kl', 'ntp_x')


@dataclass
class SleepConfig:
    method: str = 'replay_kl'
    # replay
    n_samples: int = 32            # K continuations
    sample_len: int = 256          # N tokens each
    temperature: float = 1.0
    top_p: float | None = None
    sample_batch: int = 32         # continuations sampled in parallel
    # optimisation
    steps: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    batch: int = 8                 # replay rows per step (replay_kl); X chunks per step (ntp_x)
    chunk_len: int = 512           # ntp_x: X is cut into chunks of this length
    grad_clip: float | None = 1.0
    lr_schedule: str = 'const'     # const | cosine | linear (decay to 0 after warmup)
    warmup: int = 0                # linear warmup steps from 0 to lr
    # regularisers: loss = distill + lm_mix * CE(windows) + retain_kl * KL(original || student)(windows)
    lm_mix: float = 0.5
    retain_kl: float = 0.0
    lm_batch: int = 4
    lm_len: int = 512
    retain_windows: int = 8        # fixed windows scored before / after (forgetting witness)
    # multi-hop curriculum (replay_kl): student memory level per hop, last must be 0
    hops: tuple[float, ...] = (0.0,)
    degrade: str = 'scale'         # memory.DEGRADATIONS
    teacher: str = 'fixed'         # fixed | chain
    seed: int = 0
    eval_chunk: int = 8            # rows per no-grad pass when scoring the replay set

    def __post_init__(self):
        assert self.method in METHODS, f'method {self.method!r} not in {METHODS}'
        assert self.lr_schedule in ('const', 'cosine', 'linear'), self.lr_schedule
        assert 0 <= self.warmup <= self.steps
        self.hops = tuple(float(h) for h in self.hops)   # type: ignore[assignment]
        assert self.hops and self.hops[-1] == 0.0 and all(0.0 <= h <= 1.0 for h in self.hops), self.hops
        assert all(a > b for a, b in zip(self.hops, self.hops[1:])), f'hops must decrease: {self.hops}'
        assert self.degrade in DEGRADATIONS, self.degrade
        assert self.teacher in ('fixed', 'chain'), self.teacher
        self.betas = tuple(self.betas)   # type: ignore[assignment]  # yaml gives a list

    def asdict(self) -> dict:
        return asdict(self)


def _expand(obj, K: int):
    '''A batch-1 exported state repeated K times along the batch dim
    (tensors of any rank >= 1; ints and None pass through).'''
    if torch.is_tensor(obj):
        if obj.dim() == 0:
            return obj
        assert obj.shape[0] == 1, f'expected a batch-1 state, got {tuple(obj.shape)}'
        return obj.expand(K, *obj.shape[1:])
    if isinstance(obj, dict):
        return {k: _expand(v, K) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_expand(v, K) for v in obj)
    return obj


def _param_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def lr_factor(step: int, steps: int, schedule: str, warmup: int) -> float:
    '''Multiplier of the peak lr at `step` (0-based): linear warmup over
    `warmup` steps, then const, or cosine / linear decay reaching 0 at
    the last step.'''
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == 'const' or steps - warmup <= 1:
        return 1.0
    t = (step - warmup) / (steps - 1 - warmup)          # 0 at the first post-warmup step, 1 at the last
    if schedule == 'linear':
        return 1.0 - t
    return 0.5 * (1.0 + math.cos(math.pi * t))


class Sleeper:

    def __init__(self, gen: Generator, store: TokenStore, cfg: SleepConfig,
                 log: Callable[[str], None] = lambda s: None):
        self.gen = gen
        self.model: torch.nn.Module = gen.model   # type: ignore[assignment]  # Decodable + nn.Module
        self.store = store
        self.cfg = cfg
        self.log = log
        self.device = gen.device
        assert gen.start_id is not None, 'consolidation needs a start token'
        self.start_id: int = gen.start_id
        self.rng = np.random.default_rng(cfg.seed)
        self.autocast = (self.device.type == 'cuda')
        self.original: torch.nn.Module | None = None   # frozen copy of the weights, for retain_kl

    def _original(self) -> torch.nn.Module:
        '''The subject's original weights, copied once (in the model's
        serving dtype) the first time a retain term needs them; the
        caller restores the model between items, so the copy stays valid.'''
        if self.original is None:
            self.original = copy.deepcopy(self.model).eval()
            for p in self.original.parameters():
                p.requires_grad_(False)
        return self.original

    def _retain_kl(self, win: torch.Tensor) -> torch.Tensor:
        '''KL(original || student) averaged over the positions of win[:, :-1].'''
        with torch.no_grad():
            t = self._original()(win[:, :-1])
        s_ = self._forward(win[:, :-1])
        return self._kl(s_, t, torch.ones(win.shape[0], win.shape[1] - 1, dtype=torch.bool, device=self.device))

    # --- windows of the store (LM mix and retention witness) -----------

    def _windows(self, n: int, L: int) -> torch.Tensor:
        weights = np.asarray(self.store.shard_tokens, dtype=np.float64)
        weights /= weights.sum()
        rows = []
        for _ in range(n):
            s = int(self.rng.choice(len(weights), p=weights))
            st = int(self.rng.integers(0, self.store.shard_tokens[s] - (L + 1)))
            rows.append(self.store.read_window(s, st, L + 1).astype(np.int64))
        return torch.from_numpy(np.stack(rows)).to(self.device)

    def _lm_loss(self, win: torch.Tensor) -> torch.Tensor:
        '''Mean CE of win[:, 1:] given win[:, :-1] (the window's own
        first token stands in for a start token, as training does).'''
        logits = self._forward(win[:, :-1])
        return F.cross_entropy(logits.float().flatten(0, 1), win[:, 1:].flatten())

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return self.model(tokens)
        return self.model(tokens)

    # --- the memory and its replay -------------------------------------

    @torch.no_grad()
    def read(self, x: torch.Tensor) -> dict:
        '''Fresh stream, read x (Lx,) -> the exported (batch-1) state m.'''
        self.gen.reset(1, max_len=1 + int(x.shape[0]) + self.cfg.sample_len)
        self.gen.prefill_ids(x[None].to(self.device))
        return self.gen.export_state()

    @torch.no_grad()
    def replay(self, m: dict) -> tuple[torch.Tensor, torch.Tensor]:
        '''K continuations (K, N) sampled from (w, m), and the teacher
        logits (K, N, V) of every continuation token under (w, m), in
        the model's dtype.'''
        cfg = self.cfg
        K, N = cfg.n_samples, cfg.sample_len
        max_len = int(m['fed_len']) + 1 + N
        samples, teacher = [], []
        seed0 = int(self.rng.integers(0, 2 ** 31))
        for b0 in range(0, K, cfg.sample_batch):
            b = min(cfg.sample_batch, K - b0)
            state = {**m, 'cache': _expand(m['cache'], b), 'pending': _expand(m['pending'], b),
                     'batch_size': b}
            self.gen.load_state(state, max_len=max_len)
            s = self.gen.gen_ids(SamplingConfig(max_new_tokens=N, temperature=cfg.temperature,
                                                top_p=cfg.top_p, stop_ids=(), seed=seed0 + b0))
            assert s.shape == (b, N), s.shape
            # teacher logits: the memory again, then the sampled tokens teacher-forced
            self.gen.load_state(state, max_len=max_len)
            assert self.gen.pending is not None
            block = torch.cat([self.gen.pending[:, None], s[:, :-1]], dim=1)
            t = self.model.decode_step(block, return_logits=True)
            assert t is not None
            samples.append(s)
            teacher.append(t)
        return torch.cat(samples), torch.cat(teacher)

    def _valid(self, s: torch.Tensor) -> torch.Tensor:
        '''(K, N) bool: positions up to and including the first
        start_id the model emitted (the document ends there).'''
        is_bos = (s == self.start_id).long()
        return (is_bos.cumsum(1) - is_bos) == 0

    def _student_input(self, s: torch.Tensor) -> torch.Tensor:
        bos = torch.full((s.shape[0], 1), self.start_id, dtype=torch.int64, device=self.device)
        return torch.cat([bos, s[:, :-1]], dim=1)

    def _student_logits(self, s: torch.Tensor, mem: dict | None) -> torch.Tensor:
        '''Logits predicting s (b, N) from the student: a fresh stream
        (start_id + s) when mem is None, else the stream continuing
        from mem's pending token with mem's matrix states as entry
        states (the differentiable forward).'''
        if mem is None:
            return self._forward(self._student_input(s))
        b = s.shape[0]
        pending = mem['pending'].expand(b)
        block = torch.cat([pending[:, None], s[:, :-1]], dim=1)
        caches = _expand(mem['cache'], b)['blocks']
        if self.autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return self.model(block, caches=caches)
        return self.model(block, caches=caches)

    def _kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            valid: torch.Tensor) -> torch.Tensor:
        '''Mean over valid positions of KL(teacher || student), fp32.'''
        pt = teacher_logits.float().log_softmax(-1)
        ps = student_logits.float().log_softmax(-1)
        kl = (pt.exp() * (pt - ps)).sum(-1)
        return (kl * valid).sum() / valid.sum().clamp(min=1)

    @torch.no_grad()
    def _replay_kl(self, s: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor,
                   mem: dict | None = None) -> float:
        tot, cnt = 0.0, 0
        for b0 in range(0, s.shape[0], self.cfg.eval_chunk):
            sl = slice(b0, b0 + self.cfg.eval_chunk)
            n = int(valid[sl].sum())
            if n == 0:
                continue
            tot += float(self._kl(self._student_logits(s[sl], mem), teacher[sl], valid[sl])) * n
            cnt += n
        return tot / max(cnt, 1)

    # --- the procedures --------------------------------------------------

    def consolidate(self, x: torch.Tensor) -> dict:
        '''x (Lx,) int64. Trains the model in place; returns the witness.'''
        cfg = self.cfg
        x = x.to(self.device, dtype=torch.int64)
        orig_dtype = _param_dtype(self.model)
        retain = self._windows(cfg.retain_windows, cfg.lm_len) if cfg.retain_windows else None

        m = s = teacher = valid = None
        if cfg.method == 'replay_kl':
            m = self.read(x)
            s, teacher = self.replay(m)
            valid = self._valid(s)

        self.model.float()
        for p in self.model.parameters():
            p.requires_grad_(True)
        opt = build_adamw(self.model, {'lr': cfg.lr, 'betas': list(cfg.betas),
                                       'weight_decay': cfg.weight_decay})
        witness: dict = {'method': cfg.method, 'x_len': int(x.shape[0]), 'config': cfg.asdict()}
        if cfg.retain_kl > 0:
            self._original()
        with torch.no_grad():
            if retain is not None:
                witness['retain_before'] = float(self._lm_loss(retain))
                if cfg.retain_kl > 0:
                    witness['retain_kl_before'] = float(self._retain_kl(retain))
            if cfg.method == 'replay_kl':
                assert s is not None and teacher is not None and valid is not None
                witness['kl_before'] = self._replay_kl(s, teacher, valid)
                witness['replay_valid_frac'] = float(valid.float().mean())

        # the hop plan: (student memory level, steps); ntp_x is one hop at level 0
        levels = list(cfg.hops) if cfg.method == 'replay_kl' else [0.0]
        per_hop = [cfg.steps // len(levels)] * len(levels)
        per_hop[-1] += cfg.steps - sum(per_hop)
        bos = torch.full((1, 1), self.start_id, dtype=torch.int64, device=self.device)
        xin = torch.cat([bos, x[None]], dim=1)          # start + X: X's tokens are the targets
        losses: list[float] = []
        hops: list[dict] = []
        step = 0
        for h, (level, n_steps) in enumerate(zip(levels, per_hop)):
            mem = None
            if cfg.method == 'replay_kl':
                assert m is not None
                mem = degrade(m, level, cfg.degrade) if level > 0 else None
                if cfg.teacher == 'chain' and h > 0:
                    # the previous hop's student, (w_h, m_{h-1}), becomes the teacher
                    prev = degrade(m, levels[h - 1], cfg.degrade)
                    with torch.no_grad():
                        s, teacher = self.replay(prev)
                    valid = self._valid(s)
                assert s is not None and teacher is not None and valid is not None
                hop = {'level': level, 'steps': n_steps}
                with torch.no_grad():
                    hop['kl_before'] = self._replay_kl(s, teacher, valid, mem)
            else:
                hop = {'level': 0.0, 'steps': n_steps}
            for _ in range(n_steps):
                f = lr_factor(step, cfg.steps, cfg.lr_schedule, cfg.warmup)
                for g in opt.param_groups:
                    g['lr'] = cfg.lr * f
                opt.zero_grad(set_to_none=True)
                if cfg.method == 'replay_kl':
                    assert s is not None and teacher is not None and valid is not None
                    idx = torch.from_numpy(self.rng.choice(s.shape[0], size=min(cfg.batch, s.shape[0]),
                                                           replace=False)).to(self.device)
                    distill = self._kl(self._student_logits(s[idx], mem), teacher[idx], valid[idx])
                else:
                    # chunks of start+X; a chunk's first token is its context, as _lm_loss
                    L, C = xin.shape[1], cfg.chunk_len + 1
                    if L <= C:
                        chunks = xin.expand(cfg.batch, L)
                    else:
                        starts = self.rng.integers(0, L - C + 1, size=cfg.batch)
                        chunks = torch.stack([xin[0, st:st + C] for st in starts])
                    distill = self._lm_loss(chunks)
                loss = distill
                if cfg.lm_mix > 0 or cfg.retain_kl > 0:
                    win = self._windows(cfg.lm_batch, cfg.lm_len)
                    if cfg.lm_mix > 0:
                        loss = loss + cfg.lm_mix * self._lm_loss(win)
                    if cfg.retain_kl > 0:
                        loss = loss + cfg.retain_kl * self._retain_kl(win)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
                losses.append(float(distill.detach()))
                step += 1
            if cfg.method == 'replay_kl':
                assert s is not None and teacher is not None and valid is not None
                with torch.no_grad():
                    hop['kl_after'] = self._replay_kl(s, teacher, valid, mem)
            hops.append(hop)
        witness['distill_loss'] = losses
        witness['hops'] = hops

        with torch.no_grad():
            if retain is not None:
                witness['retain_after'] = float(self._lm_loss(retain))
                if cfg.retain_kl > 0:
                    witness['retain_kl_after'] = float(self._retain_kl(retain))
            if cfg.method == 'replay_kl':
                assert s is not None and teacher is not None and valid is not None
                witness['kl_after'] = self._replay_kl(s, teacher, valid)
        del opt
        for p in self.model.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.model.to(orig_dtype)
        return witness

    def decode_replay(self, s: torch.Tensor, n: int = 2) -> list[str]:
        '''The first n replay rows as text, for the record.'''
        if self.gen.tokenizer is None:
            return []
        return self.gen.decode(s[:n])
