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
#   Prioritised replay (replay_from): 'end' starts every continuation
#   from the memory after all of X; 'uniform' / 'surprise' start them
#   from the memory at positions inside X — X cut into replay_bin-token
#   bins, a bin drawn uniformly or with probability proportional to its
#   summed delta-rule residual (GDNLM.surprise: what the memory failed
#   to predict about what it stored, the CA1 comparator's signal),
#   standardised across the document's bins and exponentiated with
#   surprise_power. The biological replay's bias towards novel and
#   surprising episodes, made a sampling weight. On LAX1 the high
#   residuals sit at section breaks, topic changes and rare names.
#
# Joint modes (mode != 'onehop'): the student's memory is a variable
#   too. (w', m') starts at the teacher point (w, m) and is pushed by a
#   *predictive* penalty  lam * KL( S(w',m') || S(w',m0) )  — how much
#   the residual memory still changes the prediction (the norm of m is
#   a gauge the per-head RMSNorm readout cannot see, so |m' - m0| would
#   be spent on nothing) — while an anchor term keeps the prediction
#   itself in place. The path from (w, m) to (w*, ~m0) is then chosen by
#   gradient descent, not by a hand-made degradation (memory.py).
#     joint_fixed     anchor = the frozen teacher: KL(T || S(w',m'))
#     joint_reanchor  anchor = the student's own prediction one step
#                     earlier (a proximal step in KL geometry — at the
#                     current point the KL is second order and its
#                     gradient zero, so the anchor must lag): the
#                     chain that a fully online scheme needs; its cost
#                     is read off the drift KL(T || S(w',m')).
#   mem_param 'rows' releases per key channel, m'_l = m_l * sigmoid(g_l)
#   with g a (H, dk) vector per layer — the geometry of the gate's own
#   forgetting, well conditioned under the readout's scale gauge, and
#   the learned g is the release order; 'free' trains every entry.
#   penalty_ref 'student' judges emptiness with the live weights,
#   KL(S(w',m') || S(w',m0)) — a redundancy condition w' alone can
#   satisfy (the first sweep: the memory never moved, w' made the two
#   branches agree, one hop in disguise); 'original' judges it with the
#   frozen original weights, KL(S(w0,m') || S(w0,m0)), gradient to m'
#   only, so the memory is drained by a term w' cannot touch and the
#   anchor forces w' to absorb what it loses.
#   anchor_ema > 0 replaces the one-step-lag anchor of joint_reanchor by
#   an EMA copy of (w', m') with that decay: a long-horizon anchor
#   between lag-1 (0) and the frozen teacher (1); lag-1 under Adam
#   integrated the penalty's bias into 0.28 nat of drift.
#   At the end the memory is cleared as before; the witness carries the
#   trajectory of penalty / drift / blank-student KL.
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
    replay_from: str = 'end'       # end | uniform | surprise: where in X the replays start
    replay_bin: int = 64           # tokens per bin for uniform / surprise starts
    surprise_power: float = 1.0    # bin weight = exp(power * z), z the standardised bin residual sum
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
    # joint modes: the memory as a variable (see the header)
    mode: str = 'onehop'           # onehop | joint_fixed | joint_reanchor
    lam: float = 1.0               # weight of the predictive penalty KL(S(w',m') || S(w',m0))
    mem_param: str = 'rows'        # rows | free
    mem_lr: float = 0.2            # Adam lr of the memory parameters (rows: on the logit g; free: on entries)
    mem_init: float = 6.0          # rows: g init, sigmoid(6) = 0.9975
    penalty_ref: str = 'student'   # student | original: whose weights judge the memory's emptiness
    anchor_ema: float = 0.0        # joint_reanchor: EMA decay of the anchor copy (0 = one-step lag)
    probe_every: int = 16          # steps between trajectory probes (penalty / drift / blank KL)
    layer_probe: bool = False      # per-layer read KL at the end (24 x replay-set forwards)
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
        assert self.replay_from in ('end', 'uniform', 'surprise'), self.replay_from
        assert self.replay_from == 'end' or self.hops == (0.0,), 'positional replay is one-hop only'
        assert self.replay_bin >= 1                    # surprise_power < 0 favours the least surprising bins
        assert self.mode in ('onehop', 'joint_fixed', 'joint_reanchor'), self.mode
        assert self.mem_param in ('rows', 'free'), self.mem_param
        if self.mode != 'onehop':
            assert self.method == 'replay_kl' and self.hops == (0.0,) and self.replay_from == 'end', \
                'joint modes: replay_kl, one hop, end replay'
            assert self.lam >= 0 and self.mem_lr >= 0 and self.probe_every >= 1
            assert self.penalty_ref in ('student', 'original'), self.penalty_ref
            assert 0.0 <= self.anchor_ema < 1.0, self.anchor_ema
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


class MemoryParam:
    '''The student's memory as trainable parameters over an exported
    (batch-1) state m. rows: S'_l = S_l * sigmoid(g_l), g_l (1, H, dk, 1);
    free: S'_l itself. The short-conv caches and the pending token are
    carried through unchanged.'''

    def __init__(self, m: dict, how: str, init: float):
        self.how = how
        self.pending = m['pending']
        self.extras = [{k: v for k, v in b['att'].items() if k != 'state'} for b in m['cache']['blocks']]
        self.base = [b['att']['state'].detach().clone() for b in m['cache']['blocks']]
        if how == 'rows':
            self.params = [torch.full((*S.shape[:3], 1), init, dtype=S.dtype, device=S.device, requires_grad=True)
                           for S in self.base]
        else:
            self.params = [S.clone().requires_grad_(True) for S in self.base]

    def states(self) -> list[torch.Tensor]:
        if self.how == 'rows':
            return [S * torch.sigmoid(g) for S, g in zip(self.base, self.params)]
        return list(self.params)

    def state(self) -> dict:
        return {'pending': self.pending,
                'cache': {'blocks': [{'att': {'state': S, **ex}} for S, ex in zip(self.states(), self.extras)]}}

    def profile(self) -> dict:
        '''Per-layer summary of what was released.'''
        with torch.no_grad():
            if self.how == 'rows':
                keep = [torch.sigmoid(g) for g in self.params]
                return {'keep_mean': [float(k.mean()) for k in keep],
                        'keep_frac_below_half': [float((k < 0.5).float().mean()) for k in keep],
                        # weighted by the row norms the memory actually holds
                        'keep_weighted': [float((k[..., 0] * S.norm(dim=-1)).sum() / S.norm(dim=-1).sum().clamp(min=1e-9))
                                          for k, S in zip(keep, self.base)]}
            return {'rel_change': [float((P - S).norm() / S.norm().clamp(min=1e-9)) for P, S in zip(self.params, self.base)],
                    'norm_ratio': [float(P.norm() / S.norm().clamp(min=1e-9)) for P, S in zip(self.params, self.base)]}


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

    @torch.no_grad()
    def replay_from_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        '''K continuations started from the memory at positions inside
        x (Lx,), the start bin of each drawn by cfg.replay_from, plus
        the teacher logits under the memory at that position and a
        record: the per-bin surprise profile and the bins drawn.'''
        cfg = self.cfg
        L = int(x.shape[0])
        nb = max(L // cfg.replay_bin, 1)
        edges = [min((b + 1) * cfg.replay_bin, L) for b in range(nb)]
        edges[-1] = L                                              # the remainder joins the last bin
        info: dict = {'bins': nb, 'bin_len': cfg.replay_bin}
        if cfg.replay_from == 'surprise':
            prof = self.model.surprise(x[None].to(self.device)).mean(dim=(2, 3))[0]   # (L,)  # type: ignore[operator]
            lo = 0
            weights = []
            for hi in edges:
                weights.append(float(prof[lo:hi].sum()))
                lo = hi
            # bin sums vary by a few percent around a large floor (most layers
            # never predict most of a value), so the weight is exp(power * z)
            # with z the bin's standardised sum; the first bin is excluded —
            # its residuals are high only because the memory is still empty
            a = np.asarray(weights, dtype=np.float64)
            info['surprise'] = [round(v, 4) for v in weights]
            if nb > 1:
                z = (a[1:] - a[1:].mean()) / (a[1:].std() + 1e-9)
                w = np.concatenate([[0.0], np.exp(cfg.surprise_power * z)])
            else:
                w = np.ones(1)
        else:
            w = np.ones(nb)
        w = w / w.sum()
        info['weights'] = [round(float(v), 4) for v in w]
        draws = self.rng.choice(nb, size=cfg.n_samples, p=w)
        counts = np.bincount(draws, minlength=nb)
        info['draws'] = counts.tolist()
        # walk x once, exporting the memory at the end of every drawn bin
        N = cfg.sample_len
        self.gen.reset(1, max_len=1 + L + N)
        pos = 0
        seed0 = int(self.rng.integers(0, 2 ** 31))
        samples, teacher = [], []
        for b in range(nb):
            hi = edges[b]
            if hi > pos:
                self.gen.prefill_ids(x[None, pos:hi].to(self.device))
                pos = hi
            n = int(counts[b])
            if n == 0:
                continue
            m = self.gen.export_state()
            for b0 in range(0, n, cfg.sample_batch):
                bb = min(cfg.sample_batch, n - b0)
                state = {**m, 'cache': _expand(m['cache'], bb), 'pending': _expand(m['pending'], bb),
                         'batch_size': bb}
                self.gen.load_state(state, max_len=hi + 1 + N)
                s_ = self.gen.gen_ids(SamplingConfig(max_new_tokens=N, temperature=cfg.temperature,
                                                     top_p=cfg.top_p, stop_ids=(), seed=seed0 + b * 1000 + b0))
                assert s_.shape == (bb, N), s_.shape
                self.gen.load_state(state, max_len=hi + 1 + N)
                assert self.gen.pending is not None
                block = torch.cat([self.gen.pending[:, None], s_[:, :-1]], dim=1)
                t = self.model.decode_step(block, return_logits=True)
                assert t is not None
                samples.append(s_)
                teacher.append(t)
            # resume the walk from the full memory at hi
            self.gen.load_state(m, max_len=1 + L + N)
        return torch.cat(samples), torch.cat(teacher), info

    def _valid(self, s: torch.Tensor) -> torch.Tensor:
        '''(K, N) bool: positions up to and including the first
        start_id the model emitted (the document ends there).'''
        is_bos = (s == self.start_id).long()
        return (is_bos.cumsum(1) - is_bos) == 0

    def _student_input(self, s: torch.Tensor) -> torch.Tensor:
        bos = torch.full((s.shape[0], 1), self.start_id, dtype=torch.int64, device=self.device)
        return torch.cat([bos, s[:, :-1]], dim=1)

    def _student_logits(self, s: torch.Tensor, mem: dict | None,
                        model: torch.nn.Module | None = None) -> torch.Tensor:
        '''Logits predicting s (b, N) from the student (or `model`): a
        fresh stream (start_id + s) when mem is None, else the stream
        continuing from mem's pending token with mem's matrix states as
        entry states (the differentiable forward).'''
        model = self.model if model is None else model
        if mem is None:
            block = self._student_input(s)
            caches = None
        else:
            b = s.shape[0]
            pending = mem['pending'].expand(b)
            block = torch.cat([pending[:, None], s[:, :-1]], dim=1)
            caches = _expand(mem['cache'], b)['blocks']
        if self.autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return model(block, caches=caches)
        return model(block, caches=caches)

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

    def consolidate(self, x: 'torch.Tensor | list[torch.Tensor]') -> dict:
        '''One sleep. x: a context (Lx,) int64, or a list of contexts —
        several memories consolidated together, their replays pooled
        (interleaved replay, as a night's sleep replays the day's
        episodes), with cfg.steps per context so the per-document
        budget is the single-context one. Multi-hop curricula are
        single-context only. Trains the model in place; returns the
        witness.'''
        cfg = self.cfg
        xs = [x] if torch.is_tensor(x) else list(x)
        xs = [t.to(self.device, dtype=torch.int64) for t in xs]
        assert len(xs) == 1 or cfg.hops == (0.0,), 'multi-hop curricula are single-context only'
        orig_dtype = _param_dtype(self.model)
        retain = self._windows(cfg.retain_windows, cfg.lm_len) if cfg.retain_windows else None

        m = s = teacher = valid = None
        replay_info = []
        if cfg.method == 'replay_kl':
            ss, ts = [], []
            for t in xs:
                if cfg.replay_from == 'end':
                    m = self.read(t)
                    a, b = self.replay(m)
                else:
                    a, b, info = self.replay_from_x(t)
                    replay_info.append(info)
                ss.append(a)
                ts.append(b)
            s, teacher = torch.cat(ss), torch.cat(ts)
            valid = self._valid(s)
            if len(xs) > 1 or cfg.replay_from != 'end':
                m = None                        # no single end memory for intermediate hops
        x = xs[0]

        self.model.float()
        for p in self.model.parameters():
            p.requires_grad_(True)
        opt = build_adamw(self.model, {'lr': cfg.lr, 'betas': list(cfg.betas),
                                       'weight_decay': cfg.weight_decay})
        witness: dict = {'method': cfg.method, 'x_len': int(x.shape[0]), 'n_contexts': len(xs),
                         'config': cfg.asdict()}
        if replay_info:
            witness['replay'] = replay_info
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

        if cfg.mode != 'onehop':
            assert m is not None and s is not None and teacher is not None and valid is not None
            self._joint(m, s, teacher, valid, retain, opt, witness)
            del opt
            for p in self.model.parameters():
                p.requires_grad_(False)
                p.grad = None
            self.model.to(orig_dtype)
            return witness

        # the hop plan: (student memory level, steps); ntp_x is one hop at level 0
        total = cfg.steps * len(xs)
        levels = list(cfg.hops) if cfg.method == 'replay_kl' else [0.0]
        per_hop = [total // len(levels)] * len(levels)
        per_hop[-1] += total - sum(per_hop)
        bos = torch.full((1, 1), self.start_id, dtype=torch.int64, device=self.device)
        xins = [torch.cat([bos, t[None]], dim=1) for t in xs]   # start + X: X's tokens are the targets
        losses: list[float] = []
        hops: list[dict] = []
        step = 0
        for h, (level, n_steps) in enumerate(zip(levels, per_hop)):
            mem = None
            if cfg.method == 'replay_kl':
                if level > 0 or (cfg.teacher == 'chain' and h > 0):
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
                f = lr_factor(step, total, cfg.lr_schedule, cfg.warmup)
                for g in opt.param_groups:
                    g['lr'] = cfg.lr * f
                opt.zero_grad(set_to_none=True)
                if cfg.method == 'replay_kl':
                    assert s is not None and teacher is not None and valid is not None
                    idx = torch.from_numpy(self.rng.choice(s.shape[0], size=min(cfg.batch, s.shape[0]),
                                                           replace=False)).to(self.device)
                    distill = self._kl(self._student_logits(s[idx], mem), teacher[idx], valid[idx])
                else:
                    # chunks of start+X (one context per row, drawn at random among the
                    # contexts); a chunk's first token is its context, as _lm_loss
                    C = cfg.chunk_len + 1
                    rows = []
                    for _ in range(cfg.batch):
                        xin = xins[int(self.rng.integers(0, len(xins)))]
                        L = xin.shape[1]
                        if L <= C:
                            rows.append(F.pad(xin[0], (0, C - L), value=self.start_id) if len(xins) > 1 else xin[0])
                        else:
                            st = int(self.rng.integers(0, L - C + 1))
                            rows.append(xin[0, st:st + C])
                    distill = self._lm_loss(torch.stack(rows))
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

    def _joint(self, m: dict, s: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor,
               retain: torch.Tensor | None, opt, witness: dict) -> None:
        '''The joint modes (see the header). Trains the model and the
        memory parameters in place; fills the witness.'''
        cfg = self.cfg
        mp = MemoryParam(m, cfg.mem_param, cfg.mem_init)
        opt.add_param_group({'params': mp.params, 'lr': cfg.mem_lr, 'weight_decay': 0.0, 'name': 'memory'})
        K = s.shape[0]
        batches = [torch.from_numpy(self.rng.choice(K, size=min(cfg.batch, K), replace=False)).to(self.device)
                   for _ in range(cfg.steps)]
        orig = self._original() if cfg.penalty_ref == 'original' else None
        blank0 = None
        if orig is not None:
            # S(w0, m0) on the whole replay set, fixed for the sleep
            with torch.no_grad():
                blank0 = torch.cat([self._student_logits(s[b0:b0 + cfg.eval_chunk], None, orig)
                                    for b0 in range(0, K, cfg.eval_chunk)])
        ema_model = ema_params = None
        if cfg.mode == 'joint_reanchor' and cfg.anchor_ema > 0:
            ema_model = copy.deepcopy(self.model).eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)
            ema_params = [p.detach().clone() for p in mp.params]

        def ema_state() -> dict:
            assert ema_params is not None
            saved = mp.params
            mp.params = ema_params
            try:
                return mp.state()
            finally:
                mp.params = saved

        def probe(step: int) -> dict:
            with torch.no_grad():
                mem = mp.state()
                row = {'step': step,
                       'penalty': self._pair_kl(s, mem, None),          # KL(S(w',m') || S(w',m0))
                       'drift': self._replay_kl(s, teacher, valid, mem),   # KL(T || S(w',m'))
                       'blank': self._replay_kl(s, teacher, valid)}        # KL(T || S(w',m0))
            return row

        traj = [probe(0)]
        anchor_next = teacher[batches[0]]                      # step 0: the teacher is the current point
        losses, penalties = [], []
        for step in range(cfg.steps):
            f = lr_factor(step, cfg.steps, cfg.lr_schedule, cfg.warmup)
            for g in opt.param_groups:
                g['lr'] = (cfg.mem_lr if g.get('name') == 'memory' else cfg.lr) * f
            opt.zero_grad(set_to_none=True)
            idx = batches[step]
            mem = mp.state()
            s_mem = self._student_logits(s[idx], mem)
            s_blank = self._student_logits(s[idx], None)
            anchor = teacher[idx] if cfg.mode == 'joint_fixed' else anchor_next
            distill = self._kl(s_mem, anchor, valid[idx])
            if orig is None:
                penalty = self._kl(s_blank, s_mem, valid[idx])     # KL(S(w',m') || S(w',m0)), both sides live
            else:
                assert blank0 is not None
                o_mem = self._student_logits(s[idx], mem, orig)     # frozen w0 reads the live memory
                penalty = self._kl(blank0[idx], o_mem, valid[idx])  # KL(S(w0,m') || S(w0,m0)), grad to m' only
            loss = distill + cfg.lam * penalty
            if cfg.lm_mix > 0 or cfg.retain_kl > 0:
                win = self._windows(cfg.lm_batch, cfg.lm_len)
                if cfg.lm_mix > 0:
                    loss = loss + cfg.lm_mix * self._lm_loss(win)
                if cfg.retain_kl > 0:
                    loss = loss + cfg.retain_kl * self._retain_kl(win)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + mp.params, cfg.grad_clip)
            if cfg.mode == 'joint_reanchor' and ema_model is None and step + 1 < cfg.steps:
                # one-step lag: this point's prediction (before the update) on the next batch
                with torch.no_grad():
                    anchor_next = self._student_logits(s[batches[step + 1]], mp.state())
            opt.step()
            if cfg.mode == 'joint_reanchor' and step + 1 < cfg.steps:
                with torch.no_grad():
                    if ema_model is not None:
                        assert ema_params is not None
                        d = cfg.anchor_ema
                        for pe, p in zip(ema_model.parameters(), self.model.parameters()):
                            pe.mul_(d).add_(p.detach(), alpha=1 - d)
                        for pe, p in zip(ema_params, mp.params):
                            pe.mul_(d).add_(p.detach(), alpha=1 - d)
                        anchor_next = self._student_logits(s[batches[step + 1]], ema_state(), ema_model)
            losses.append(float(distill.detach()))
            penalties.append(float(penalty.detach()))
            if (step + 1) % cfg.probe_every == 0 and step + 1 < cfg.steps:
                traj.append(probe(step + 1))
        traj.append(probe(cfg.steps))
        witness['distill_loss'] = losses
        witness['penalty_loss'] = penalties
        witness['trajectory'] = traj
        witness['kl_after'] = traj[-1]['blank']
        witness['penalty_after'] = traj[-1]['penalty']
        witness['drift_after'] = traj[-1]['drift']
        witness['release'] = mp.profile()
        del ema_model, ema_params, blank0
        witness['hops'] = [{'level': 'joint', 'steps': cfg.steps, 'kl_before': traj[0]['blank'],
                            'kl_after': traj[-1]['blank']}]
        if cfg.layer_probe:
            witness['layer_read_kl'] = self._layer_probe(s, mp)
        with torch.no_grad():
            if retain is not None:
                witness['retain_after'] = float(self._lm_loss(retain))
                if cfg.retain_kl > 0:
                    witness['retain_kl_after'] = float(self._retain_kl(retain))

    @torch.no_grad()
    def _pair_kl(self, s: torch.Tensor, mem_p: dict | None, mem_q: dict | None) -> float:
        '''KL( S(w', mem_p) || S(w', mem_q) ) over the valid positions of the replay set.'''
        valid = self._valid(s)
        tot, cnt = 0.0, 0
        for b0 in range(0, s.shape[0], self.cfg.eval_chunk):
            sl = slice(b0, b0 + self.cfg.eval_chunk)
            n = int(valid[sl].sum())
            if n == 0:
                continue
            tot += float(self._kl(self._student_logits(s[sl], mem_q), self._student_logits(s[sl], mem_p),
                                  valid[sl])) * n
            cnt += n
        return tot / max(cnt, 1)

    @torch.no_grad()
    def _layer_probe(self, s: torch.Tensor, mp: MemoryParam) -> list[float]:
        '''Per layer, KL( S(w',m') || S(w', m' with that layer blank) ): what each
        layer's residual memory still contributes to the prediction.'''
        full = mp.state()
        out = []
        for l in range(len(mp.base)):
            blk = copy.deepcopy(full['cache']['blocks'])
            blk[l]['att']['state'] = torch.zeros_like(blk[l]['att']['state'])
            out.append(self._pair_kl(s, full, {'pending': full['pending'], 'cache': {'blocks': blk}}))
        return out

    def decode_replay(self, s: torch.Tensor, n: int = 2) -> list[str]:
        '''The first n replay rows as text, for the record.'''
        if self.gen.tokenizer is None:
            return []
        return self.gen.decode(s[:n])
