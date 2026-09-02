# Management-aware continued training (Tier 1 of the unified-block plan).
#
# A sibling of main.py, not a refactor: main.py's loop is the audited
# path under the 15B registry runs and stays untouched. This entry point
# adds exactly three things and shares the rest by import:
#   1. init_ckpt — start a FRESH run (fresh optimizer/schedule/loader)
#      from the weights of a finished run's checkpoint (continued
#      pretraining), instead of main.py's own-run-dir resume.
#   2. manage: 'triage' — the forward is the differentiable
#      block-streaming path of components/unified.py (atoms + P=1 pool,
#      evict/demote by the error-law scores, hard selection, BPTT with
#      per-cell checkpointing). Loss stays next-token CE on all
#      positions; positions beyond the first blocks read a managed cache,
#      which is the closed-loop training signal.
#      manage: 'evict' — the eviction-only path of components/evict.py
#      (branch evict-only, 2026-09-02): same contract, no pool/demotion,
#      fused SDPA readout, compiled cell; knobs score / gumbel_tau /
#      lookahead / use_checkpoint below.
#   3. manage: 'none' — the stateless forward, byte-identical batches:
#      the paired control arm that separates "2B more tokens" from
#      "management-aware 2B tokens". Both arms live here so the loop,
#      data order and loss path are shared code, not shared intent.
#
# Resume works as in main.py (same run dir => continue); init_ckpt is
# only consulted on a fresh start.

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..components.losses import make_ce
from ..components.evict import EvictCfg, Health as EvictHealth
from ..components.evict import stream_hidden as stream_evict
from ..components.unified import Health, ManageCfg, stream_hidden
from ..dataset.loader import TokenStore, WindowLoader
from ..models import build_model
from ..optimizer import build_optimizer
from ..utils import git_state
from .main import (TrainConfig, _YamlLoader, find_resume, save_checkpoint,
                   setup_distributed, unwrap, _stop_handler)
from .metrics import FAST_METRICS, SLOW_METRICS, MetricCtx, Monitor
from .schedule import build_schedule
import signal


@dataclass
class ManagedConfig(TrainConfig):
    init_ckpt: str | None = None     # weights to continue from (fresh run)
    # Policy family name (2026-09-01, engram PLANNING.md): the dynamic
    # management algorithm is 'triage' (this repo implements the
    # evict+demote sub-family; 'full-triage' adds merge). 'unified' is
    # kept as a read alias so in-flight recipes/jobs keep working.
    manage: str = 'triage'           # 'triage' | 'evict' | 'none' ('unified' = alias)
    block_len: int = 256
    budget: int = 512
    ring_window: int = 32
    demote: bool = True
    lam: float = 1.0 / 1024      # v2 position-measure discount rate
    pool_gate: str = 'static'    # 'static' (v2) | 'stepped' (v3 per-band gate)
    pool_write: str = 'hebbian'  # 'hebbian' | 'delta' (v4 whitened slopes)
    pool_norm: str = 'ledger'    # v2.2 write normalization ('ledger' | 'off')
    manage_every: int = 1        # sprint lever: manage every k-th boundary
    compile_cell: bool = False   # sprint: torch.compile the block cell
    slope_eps: float = 1e-3      # v2 P0/P1 slope-degeneracy threshold
    # evict-only path (manage: evict)
    score: str = 'lin'           # 'lin' | 'sq' | 'p2' (components/evict.py)
    gumbel_tau: float = 0.0      # > 0: perturbed top-k during training
    lookahead: int = 0           # transport horizon beyond the block end
    use_checkpoint: bool = True  # per-cell activation checkpointing
    # kill condition (uct v2 restart plan item 1): if set, the pre-clip
    # grad norm at step 100 must be <= this, else checkpoint + exit(3)
    gnorm_gate: float | None = None
    # drop the manifest's last shard from training — the eval holdout
    # (budget_ppl.py picks entries[-1] by the same rule)
    holdout_last_shard: bool = True

    @classmethod
    def from_yaml(cls, path):        # same model_recipe merging as main
        path = Path(path)
        raw = yaml.load(open(path, encoding='utf-8'), _YamlLoader) or {}
        if 'model_recipe' in raw:
            mpath = (path.parent / raw.pop('model_recipe')).resolve()
            model = yaml.load(open(mpath, encoding='utf-8'), _YamlLoader) or {}
            assert set(model) <= {'model_name', 'model_args'}
            assert not (set(model) & set(raw))
            raw |= model
        return cls(**raw)


class StreamWrapper(nn.Module):
    '''DDP-dispatchable forward for a streaming path (calling inner
    submodules directly would bypass DDP's reducer hookup). `fn` is the
    path's stream_hidden, `health_cls` its counter class.'''

    def __init__(self, model, cfg, fn=stream_hidden, health_cls=Health,
                 use_checkpoint: bool = True):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.fn = fn
        self.health_cls = health_cls
        self.use_checkpoint = use_checkpoint
        self.health = health_cls()

    def forward(self, tokens):
        return self.fn(self.model, tokens, self.cfg, manage=True,
                       use_checkpoint=self.use_checkpoint, health=self.health)


def train(config: ManagedConfig):
    rank, world, device, is_dist = setup_distributed(config)
    is_main = rank == 0
    if config.manage == 'unified':   # legacy alias -> canonical name
        config.manage = 'triage'
    assert config.manage in ('triage', 'evict', 'none')
    if config.manage != 'none':
        assert not config.compile, \
            'streaming paths do not use model.compile_blocks; set compile: false'
        assert config.context_len % config.block_len == 0

    if world != config.world_size:
        raise ValueError(f'launched world_size={world} != recipe {config.world_size}')
    declared = (config.world_size * config.batch_size
                * config.grad_accum_steps * config.context_len)
    if config.global_batch_tokens is not None \
            and declared != config.global_batch_tokens:
        raise ValueError('global_batch_tokens mismatch')

    def log(*a):
        if is_main:
            print(f'[{datetime.now().strftime("%H:%M:%S")}]', *a, flush=True)

    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGUSR1, _stop_handler)
    import infra.train.main as _m
    _m._STOP = False

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.tf32 and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    code = git_state()
    commit8 = (code['commit'] or 'nogit')[:8]
    run_dir = Path(config.out_root) / f'{config.run_name}-{commit8}'
    resume_from = find_resume(run_dir) if run_dir.exists() else None
    if run_dir.exists() and resume_from is None and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} exists without checkpoints')

    shards = config.data_shards
    if config.holdout_last_shard:
        assert shards is None, 'holdout_last_shard with explicit data_shards'
        all_entries = TokenStore(config.data_dir).entries
        shards = [e['file'] for e in all_entries[:-1]]
        log(f'holdout (excluded from training): {all_entries[-1]["file"]}')
    store = TokenStore(config.data_dir, shards=shards)
    loader = WindowLoader(store, config.context_len, config.batch_size,
                          seed=config.seed, rank=rank, world_size=world)
    tokens_per_step = world * config.batch_size * config.grad_accum_steps * config.context_len
    steps_total = int(config.train_tokens) // tokens_per_step

    model = build_model(config.model_name, config.model_args).to(device)
    if resume_from is None and config.init_ckpt:
        state = torch.load(config.init_ckpt, map_location='cpu', weights_only=False)
        args_in_ckpt = state.get('model_args') or state.get('config', {}).get('model_args')
        if args_in_ckpt is not None:
            assert args_in_ckpt == config.model_args, \
                'init_ckpt model_args differ from recipe'
        model.load_state_dict(state.get('model', state))
        log(f'initialized weights from {config.init_ckpt} '
            f'(source step {state.get("step", "?")})')
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(config.optimizer_name, model, config.optimizer_args)
    sched = build_schedule(config.schedule_name, config.schedule_args)

    log(f'run {run_dir.name} | manage={config.manage} budget={config.budget} '
        f'block={config.block_len} score={config.score} '
        f'gumbel_tau={config.gumbel_tau} compile_cell={config.compile_cell} '
        f'use_checkpoint={config.use_checkpoint} | device {device} x{world} | '
        f'{param_count:,} params | data {store.total_tokens:,} tokens')
    log(f'plan: {steps_total:,} steps x {tokens_per_step:,} tokens/step')

    if is_main and resume_from is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.yaml').write_text(
            yaml.safe_dump(asdict(config), sort_keys=False))
        (run_dir / 'meta.json').write_text(json.dumps({
            'created': datetime.now().astimezone().isoformat(timespec='seconds'),
            'code': code, 'world_size': world, 'device': str(device),
            'torch': torch.__version__, 'config': asdict(config),
            'data': {'dir': str(Path(config.data_dir).resolve()),
                     'shards': [e['file'] for e in store.entries],
                     'total_tokens': store.total_tokens},
            'derived': {'steps_total': steps_total,
                        'tokens_per_step': tokens_per_step,
                        'param_count': param_count},
        }, indent=2))

    step, tokens_seen = 0, 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=False)
        assert state['config']['model_args'] == config.model_args
        unwrap(model).load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        loader.load_state_dict(state['loader'])
        torch.set_rng_state(state['rng']['torch'].cpu())
        if state['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state['rng']['cuda']])
        np.random.set_state(state['rng']['numpy'])
        step, tokens_seen = state['step'], state['tokens_seen']
        log(f'resumed from {resume_from.name} at step {step:,}')

    mcfg = ManageCfg(block_len=config.block_len, budget=config.budget,
                     ring_window=config.ring_window, demote=config.demote,
                     lam=config.lam, slope_eps=config.slope_eps,
                     pool_gate=config.pool_gate,
                     pool_write=config.pool_write,
                     pool_norm=config.pool_norm,
                     manage_every=config.manage_every,
                     compile_cell=config.compile_cell)
    if config.manage == 'triage':
        net: nn.Module = StreamWrapper(model, mcfg)
    elif config.manage == 'evict':
        ecfg = EvictCfg(block_len=config.block_len, budget=config.budget,
                        ring_window=config.ring_window,
                        lookahead=config.lookahead, score=config.score,
                        gumbel_tau=config.gumbel_tau,
                        manage_every=config.manage_every,
                        compile_cell=config.compile_cell)
        net = StreamWrapper(model, ecfg, stream_evict, EvictHealth,
                            use_checkpoint=config.use_checkpoint)
    else:
        if config.compile and hasattr(model, 'compile_blocks'):
            model.compile_blocks()  # type: ignore
        net = model
    if is_dist:
        net = DDP(net, device_ids=[device.index], output_device=device.index)

    amp_dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[config.dtype]
    autocast = (torch.autocast(device.type, dtype=amp_dtype)
                if amp_dtype is not torch.float32 else nullcontext())

    margs = config.model_args
    ctx = MetricCtx(model=unwrap(model), optimizer=optimizer, world_size=world,
                    device_type=device.type, param_count=param_count,
                    peak_tflops=config.peak_tflops,
                    attn_flops_per_tok=(12 * margs['layer_count'] * margs['dim']
                                        * config.context_len))
    monitor = Monitor(ctx, tokens_per_step, steps_total,
                      log_interval=config.log_interval,
                      slow_interval=config.slow_interval,
                      fast=FAST_METRICS, slow=SLOW_METRICS,
                      tb_dir=run_dir / 'tb' if is_main else None,
                      log_fn=log)

    head_weight = unwrap(model).head.weight  # type: ignore
    ce_fn = make_ce(config.ce_impl, z_loss=config.z_loss or 0.0,
                    chunk_rows=config.ce_chunk_rows,
                    compute_dtype=amp_dtype if amp_dtype is not torch.float32 else None)

    batches = iter(loader)
    net.train()
    last_recent = time.time()
    log('training...')
    try:
        while step < steps_total:
            step += 1
            mult = sched(step / steps_total)
            for g in optimizer.param_groups:
                g['lr'] = g['base_lr'] * mult

            optimizer.zero_grad(set_to_none=True)
            loss_acc = torch.zeros((), device=device)
            for micro in range(config.grad_accum_steps):
                x, y = next(batches)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                sync = (net.no_sync()  # type: ignore
                        if is_dist and micro < config.grad_accum_steps - 1
                        else nullcontext())
                with sync, autocast:
                    if config.manage != 'none':
                        hidden = net(x)
                    else:
                        hidden = net(x, return_hidden=True)
                    loss = ce_fn(hidden, head_weight, y)
                    (loss / config.grad_accum_steps).backward()
                loss_acc += loss.detach()
                ctx.last_batch = x

            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), config.grad_clip or float('inf'))
            optimizer.step()
            tokens_seen += tokens_per_step

            stop = _m._STOP
            if is_dist:
                flag = torch.tensor(float(stop), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop = bool(flag.item())

            monitor.observe(step, loss_acc / config.grad_accum_steps,
                            grad_norm, tokens_seen,
                            final=stop or step == steps_total)
            if config.gnorm_gate is not None and step == 100 \
                    and float(grad_norm) > config.gnorm_gate:
                w = unwrap(net)
                forensics = (w.health.as_dict()
                             if isinstance(w, StreamWrapper) else {})
                log(f'GNORM GATE FAILED: {float(grad_norm):.4g} > '
                    f'{config.gnorm_gate} at step 100 — a second amplifier '
                    f'exists; stopping (uct v2 restart plan item 1) | '
                    f'forensics: {forensics}')
                if is_main:
                    save_checkpoint(run_dir, 'recent', step, tokens_seen,
                                    model, optimizer, loader, config)
                monitor.close()
                if is_dist:
                    dist.destroy_process_group()
                raise SystemExit(3)
            if is_main and config.manage != 'none' \
                    and step % config.slow_interval == 0:
                w = unwrap(net)
                if isinstance(w, StreamWrapper):
                    log(f'health {w.health.as_dict()}')
                    w.health = w.health_cls()

            if is_main and config.permanent_ckpt_interval \
                    and step % config.permanent_ckpt_interval == 0 and step < steps_total:
                save_checkpoint(run_dir, 'permanent', step, tokens_seen,
                                model, optimizer, loader, config)
            if is_main and time.time() - last_recent > config.recent_ckpt_minutes * 60:
                save_checkpoint(run_dir, 'recent', step, tokens_seen,
                                model, optimizer, loader, config)
                last_recent = time.time()
                log(f'recent checkpoint at step {step:,}')
            if stop:
                log(f'stop at step {step:,}; checkpointing')
                break

        if is_main:
            kind = 'recent' if _m._STOP else 'permanent'
            p = save_checkpoint(run_dir, kind, step, tokens_seen,
                                model, optimizer, loader, config)
            log(f'final {kind} checkpoint: {p.name} | {tokens_seen/1e9:.3f}B tokens')
    finally:
        monitor.close()
        if is_dist:
            dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args = ap.parse_args()
    train(ManagedConfig.from_yaml(args.config))


if __name__ == '__main__':
    main()
