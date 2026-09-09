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
# ntp_x — the baseline consolidation must beat: the same optimiser
#   budget spent on next-token loss over X itself (the context read
#   with no memory involved), same LM mix. If replay_kl cannot beat
#   this, the memory contributed nothing beyond the text it saw.
#
# Weights train in fp32 (an AdamW step at lr 1e-5 vanishes in bf16),
# under bf16 autocast on CUDA; the model returns to its original dtype
# before the caller scores it.

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..dataset.loader import TokenStore
from ..inference import Generator, SamplingConfig
from ..optimizer import build_adamw

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
    # LM mix: loss = distill + lm_mix * CE(windows of the store)
    lm_mix: float = 0.5
    lm_batch: int = 4
    lm_len: int = 512
    retain_windows: int = 8        # fixed windows scored before / after (forgetting witness)
    seed: int = 0
    eval_chunk: int = 8            # rows per no-grad pass when scoring the replay set

    def __post_init__(self):
        assert self.method in METHODS, f'method {self.method!r} not in {METHODS}'
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
        return obj.expand(K, *obj.shape[1:]).clone()
    if isinstance(obj, dict):
        return {k: _expand(v, K) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_expand(v, K) for v in obj)
    return obj


def _param_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


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

    def _kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            valid: torch.Tensor) -> torch.Tensor:
        '''Mean over valid positions of KL(teacher || student), fp32.'''
        pt = teacher_logits.float().log_softmax(-1)
        ps = student_logits.float().log_softmax(-1)
        kl = (pt.exp() * (pt - ps)).sum(-1)
        return (kl * valid).sum() / valid.sum().clamp(min=1)

    @torch.no_grad()
    def _replay_kl(self, s: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor) -> float:
        tot, cnt = 0.0, 0
        for b0 in range(0, s.shape[0], self.cfg.eval_chunk):
            sl = slice(b0, b0 + self.cfg.eval_chunk)
            n = int(valid[sl].sum())
            if n == 0:
                continue
            tot += float(self._kl(self._forward(self._student_input(s[sl])), teacher[sl], valid[sl])) * n
            cnt += n
        return tot / max(cnt, 1)

    # --- the procedures --------------------------------------------------

    def consolidate(self, x: torch.Tensor) -> dict:
        '''x (Lx,) int64. Trains the model in place; returns the witness.'''
        cfg = self.cfg
        x = x.to(self.device, dtype=torch.int64)
        orig_dtype = _param_dtype(self.model)
        retain = self._windows(cfg.retain_windows, cfg.lm_len) if cfg.retain_windows else None

        s = teacher = valid = None
        if cfg.method == 'replay_kl':
            m = self.read(x)
            s, teacher = self.replay(m)
            valid = self._valid(s)
            del m

        self.model.float()
        for p in self.model.parameters():
            p.requires_grad_(True)
        opt = build_adamw(self.model, {'lr': cfg.lr, 'betas': list(cfg.betas),
                                       'weight_decay': cfg.weight_decay})
        witness: dict = {'method': cfg.method, 'x_len': int(x.shape[0]), 'config': cfg.asdict()}
        with torch.no_grad():
            if retain is not None:
                witness['retain_before'] = float(self._lm_loss(retain))
            if cfg.method == 'replay_kl':
                assert s is not None and teacher is not None and valid is not None
                witness['kl_before'] = self._replay_kl(s, teacher, valid)
                witness['replay_valid_frac'] = float(valid.float().mean())

        bos = torch.full((1, 1), self.start_id, dtype=torch.int64, device=self.device)
        xin = torch.cat([bos, x[None]], dim=1)          # start + X: X's tokens are the targets
        losses = []
        for step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            if cfg.method == 'replay_kl':
                assert s is not None and teacher is not None and valid is not None
                idx = torch.from_numpy(self.rng.choice(s.shape[0], size=min(cfg.batch, s.shape[0]),
                                                       replace=False)).to(self.device)
                distill = self._kl(self._forward(self._student_input(s[idx])), teacher[idx], valid[idx])
            else:
                # chunks of start+X; a chunk's first token is its context, as _lm_loss
                L = xin.shape[1]
                starts = self.rng.choice(max(L - cfg.chunk_len, 0) + 1, size=cfg.batch)
                chunks = torch.stack([xin[0, st:st + cfg.chunk_len + 1] if L > cfg.chunk_len else xin[0]
                                      for st in starts])
                distill = self._lm_loss(chunks)
            loss = distill
            if cfg.lm_mix > 0:
                loss = loss + cfg.lm_mix * self._lm_loss(self._windows(cfg.lm_batch, cfg.lm_len))
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(distill.detach()))
        witness['distill_loss'] = losses

        with torch.no_grad():
            if retain is not None:
                witness['retain_after'] = float(self._lm_loss(retain))
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
