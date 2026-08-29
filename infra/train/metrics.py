# Training observability, owned end to end by Monitor: cadence, windowed
# derived values (loss EMA, throughput, grad-norm memory), metric hooks,
# tensorboard writing and console lines. A training loop reports raw
# per-step facts through Monitor.observe() and does nothing else.
#
# The hooks are modular: each is fn(MetricCtx) -> dict of scalars;
# FAST_METRICS run every log interval, SLOW_METRICS on a sparser cadence
# (they touch all parameters). Add/remove by editing the lists. Defaults
# follow the marin hero-run dashboard, minus MoE-specific items.
#
# Architecture-specific observation belongs to the MODEL, via the optional
# convention (same design language as param_groups: the model declares its
# specialities, infrastructure merges and executes):
#
#     def metric_hooks(self) -> dict:      # both keys optional
#         return {'fast': [...], 'slow': [...]}
#
# The trainer appends these after the defaults. Hooks read what they need
# from MetricCtx — typically ctx.model's own parameters/buffers (e.g. a
# linear-attention model logging per-layer decay-gate stats, an MoE model
# logging routing entropy); .detach() tensors before reducing to floats,
# like metric_param_norms does. Returned dicts are merged by key, so a
# hook is added or removed without any other party knowing.

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist


@dataclass
class MetricCtx:
    '''The contract between a training loop and the metric hooks: fill
    what you have, hooks skip what is missing/zero.'''
    # static per run
    model: Any = None                # unwrapped root module
    optimizer: Any = None
    world_size: int = 1
    device_type: str = 'cpu'
    param_count: int = 0
    peak_tflops: float | None = None # per-device; enables mfu
    attn_flops_per_tok: float = 0.0  # 12*L*d*ctx term; 0 = 6N only
    # per log window
    loss: float = 0.0
    loss_ema: float | None = None
    z_term: float | None = None      # z-loss term inside loss (None = untracked)
    lr: float = 0.0
    grad_norm: float = 0.0
    prev_grad_norm: float = 0.0
    tokens_seen: int = 0
    tokens_per_s: float = 0.0
    step_ms: float = 0.0
    last_batch: Any = None           # most recent input ids (one rank's micro-
                                     # batch), for probe-style slow hooks


def metric_core(c: MetricCtx) -> dict:
    out = {'loss': c.loss, 'loss_ema': c.loss_ema, 'lr': c.lr,
           'grad_norm': c.grad_norm, 'tokens': c.tokens_seen}
    if c.z_term is not None:
        # 'loss' is the combined objective; only loss_ce is comparable
        # across runs with different z_loss settings
        out['loss_z'] = c.z_term
        out['loss_ce'] = c.loss - c.z_term
    if c.prev_grad_norm:
        out['grad_norm_step_ratio'] = c.grad_norm / c.prev_grad_norm
    return out


def metric_throughput(c: MetricCtx) -> dict:
    out = {'tokens_per_s': c.tokens_per_s, 'step_ms': c.step_ms}
    if c.peak_tflops:
        # PaLM-style estimate: 6N per token plus attention matmuls
        flops_per_tok = 6 * c.param_count + c.attn_flops_per_tok
        achieved = flops_per_tok * c.tokens_per_s / c.world_size
        out['mfu'] = achieved / (c.peak_tflops * 1e12)
    if c.device_type == 'cuda':
        out['max_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    return out


def metric_param_norms(c: MetricCtx) -> dict:
    '''Per-optimizer-group L2 norms (marin: constant/drifting norms are a
    health signal), plus the embedding norm on its own.'''
    out = {}
    for g in c.optimizer.param_groups:
        sq = sum(float(p.detach().float().pow(2).sum()) for p in g['params'])
        out[f"pnorm/{g['name']}"] = math.sqrt(sq)
    emb = getattr(c.model, 'embedding', None)
    if emb is not None:
        out['pnorm/embedding'] = float(emb.weight.detach().float().norm())
    return out


def metric_update_ratio(c: MetricCtx) -> dict:
    '''Per-group ||update|| / ||param|| at the current lr (marin health
    checklist; the direct check that Muon and AdamW step sizes stay
    coordinated under one shared lr). AdamW groups are computed from the
    optimizer state (post-step m/v ~ the next update); Muon groups use the
    closed form of the orthogonalized update — exact up to the
    Newton-Schulz singular spread (~[0.7, 1.3]) under lr_adjust 'rms',
    where update RMS = 0.2*lr by construction. Weight-decay contribution
    ignored in both.'''
    out = {}
    state = c.optimizer.state          # MuonAdamW: merged read-only view
    for g in c.optimizer.param_groups:
        lr = g['lr']
        upd_sq = pnorm_sq = 0.0
        for p in g['params']:
            pnorm_sq += float(p.detach().float().pow(2).sum())
            s = state.get(p)
            if not s:
                continue               # before the first step
            if 'exp_avg' in s:         # AdamW
                beta1, beta2 = g['betas']
                t = float(s['step'])
                m_hat = s['exp_avg'].float() / (1 - beta1 ** t)
                v_hat = s['exp_avg_sq'].float() / (1 - beta2 ** t)
                upd_sq += lr * lr * float(
                    (m_hat / (v_hat.sqrt() + g['eps'])).pow(2).sum())
            elif 'momentum_buffer' in s:   # Muon: analytic estimate
                m, n = p.shape
                if g['lr_adjust'] == 'rms':
                    upd_sq += (0.2 * lr) ** 2 * p.numel()
                else:                  # 'shape': ~unit spectral norm output
                    scale = max(1.0, m / n) ** 0.5
                    upd_sq += (lr * scale) ** 2 * min(m, n)
        if pnorm_sq > 0 and upd_sq > 0:
            out[f"upd_ratio/{g['name']}"] = math.sqrt(upd_sq / pnorm_sq)
    return out


FAST_METRICS = [metric_core, metric_throughput]
SLOW_METRICS = [metric_param_norms, metric_update_ratio]


class Monitor:
    '''
    The whole observation path behind one call. Construct on EVERY rank:
    observe() all-reduces the reported loss, and that collective is only
    safe because the cadence decision (step % interval) is deterministic
    and identical across ranks. Writing and printing happen only where
    enabled: pass tb_dir on the main rank alone, and a log_fn that is
    already rank-gated.

    observe(step, loss, grad_norm, tokens_seen, final, z) expects the
    per-rank mean loss of the step as a device tensor; `final` forces a
    log event off-cadence (last step, stop signal). `z` is the per-rank
    mean z-loss term inside that loss (device tensor, or None when
    untracked) — its presence must be uniform across ranks and steps, as
    it changes the collective.
    '''

    def __init__(
            self,
            ctx: MetricCtx,
            tokens_per_step: int,
            steps_total: int,
            *,
            log_interval: int,
            slow_interval: int,
            tb_dir=None,
            meta_text: str | None = None,
            log_fn: Callable = print,
            print_every: int = 10,      # console line every N log events
            fast=FAST_METRICS,
            slow=SLOW_METRICS,
        ):
        self.ctx = ctx
        self.tokens_per_step = tokens_per_step
        self.steps_total = steps_total
        self.log_interval = log_interval
        self.slow_interval = slow_interval
        self.log_fn = log_fn
        self.print_every = print_every
        self.fast = fast
        self.slow = slow

        self.writer = None
        if tb_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(tb_dir)
                if meta_text:
                    self.writer.add_text('meta', meta_text)
            except Exception as e:   # tensorboard absent on the bare cluster env
                log_fn(f'[warn] tensorboard unavailable ({e}); scalars not logged')

        self._t = time.perf_counter()

    def observe(self, step: int, loss: torch.Tensor, grad_norm,
                tokens_seen: int, final: bool = False,
                z: torch.Tensor | None = None):
        if step % self.log_interval != 0 and not final:
            return
        if dist.is_available() and dist.is_initialized():
            if z is not None:          # one collective for both scalars
                pack = torch.stack([loss, z])
                dist.all_reduce(pack, op=dist.ReduceOp.AVG)
                loss, z = pack[0], pack[1]
            else:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        c = self.ctx
        now = time.perf_counter()
        c.loss = float(loss)
        c.z_term = float(z) if z is not None else None
        c.loss_ema = (c.loss if c.loss_ema is None
                      else 0.9 * c.loss_ema + 0.1 * c.loss)
        c.prev_grad_norm, c.grad_norm = c.grad_norm, float(grad_norm)
        c.lr = c.optimizer.param_groups[0]['lr']
        c.tokens_seen = tokens_seen
        c.step_ms = (now - self._t) / self.log_interval * 1e3
        c.tokens_per_s = self.tokens_per_step * self.log_interval / (now - self._t)
        self._t = now

        scalars = {}
        for fn in self.fast:
            scalars |= fn(c)
        if step % self.slow_interval == 0:
            for fn in self.slow:
                scalars |= fn(c)
        if self.writer is not None:
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        if step % (self.log_interval * self.print_every) == 0 or final:
            self.log_fn(f'step {step:>7,}/{self.steps_total:,} | '
                        f'loss {c.loss:.4f} (ema {c.loss_ema:.4f}) | '
                        f'lr {c.lr:.2e} | gnorm {c.grad_norm:.2f} | '
                        f'{c.tokens_per_s/1e6:.2f} Mtok/s')

    def close(self):
        if self.writer is not None:
            self.writer.close()
