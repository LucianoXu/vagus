# Pretraining entry point.
#
# Design contract:
# - model/optimizer/schedule are picked by name + args dict, fed to
#   factories, so swapping architectures or optimizers is config-only.
# - resume is the primary path and fresh start its special case: the run
#   directory holds everything (resolved config, witnessed metadata,
#   tensorboard events, checkpoints), and restarting the same command
#   continues exactly (loader state + RNG + optimizer restored).
# - stop condition is a token budget; the schedule lives on progress
#   fractions, so batch-size changes rescale rather than deform it.
# - no validation loop (sub-epoch regime: every batch is unseen data);
#   fresh-data train loss + its EMA is the generalization signal.

import argparse
import json
import os
import re
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..components.losses import make_ce
from ..dataset.loader import TokenStore, WindowLoader
from ..models import build_model
from ..optimizer import build_optimizer
from ..utils import atomic_write, git_state
from .metrics import FAST_METRICS, SLOW_METRICS, MetricCtx, Monitor
from .schedule import build_schedule


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# YAML 1.1 parses `1e-4` (no dot) as a *string*; with args now passed
# through as plain dicts there is no dataclass layer to coerce it back.
# Register the full float form as an implicit resolver once, globally.
_FLOAT_RE = re.compile(r'''^[-+]?(
    (\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? | \d+[eE][-+]?\d+
    )$''', re.X)


class _YamlLoader(yaml.SafeLoader):
    pass


_YamlLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float', _FLOAT_RE, list('-+0123456789.'))


@dataclass
class TrainConfig:
    run_name: str
    data_dir: str

    # model: name selects the class, args go verbatim into its constructor
    model_name: str = 'TransformerPP'
    model_args: dict = field(default_factory=dict)

    # optimizer (adamw: lr/betas/weight_decay; decay group = matrices,
    # no-decay group = everything of dim < 2)
    optimizer_name: str = 'adamw'
    optimizer_args: dict = field(default_factory=lambda: dict(
        lr=3.0e-4, betas=(0.9, 0.95), weight_decay=0.1))

    schedule_name: str = 'cosine'
    schedule_args: dict = field(default_factory=lambda: dict(
        warmup=0.01, min_ratio=0.05))

    # data / budget
    data_shards: list | None = None      # subset by shard file name
    context_len: int = 2048              # loader window; model may allow more
    batch_size: int = 8                  # per-rank micro-batch
    grad_accum_steps: int = 1
    # Declared hardware layout. The loader's data order depends on
    # world_size (rank-interleaved slices), so the card count is part of
    # the recipe: launching with a different world size is a different
    # experiment and is refused at startup.
    world_size: int = 1
    # Declared invariant: world * batch_size * grad_accum * context_len
    # must equal this when set. Catches editing one factor and forgetting
    # the others.
    global_batch_tokens: int | None = None
    train_tokens: int = 1_000_000_000    # stop condition (global tokens)
    seed: int = 42

    # loss shaping
    grad_clip: float | None = 1.0
    z_loss: float | None = None          # folded into the reported loss
    # cross-entropy path (losses.make_ce): 'liger' fused Triton kernel
    # (CUDA-only, ~-5.5GB peak at 340M shapes for ~+3% step time),
    # 'chunked' dependency-free fused fallback, 'full' plain logits path.
    # Numerically interchangeable at bf16 rounding (bench_ce.py).
    ce_impl: str = 'liger'
    ce_chunk_rows: int = 4096            # 'chunked' only

    # system
    device: str = 'auto'                 # ignored under torchrun
    dtype: str = 'bfloat16'              # autocast dtype; 'float32' disables
    compile: bool = True
    tf32: bool = True

    # observability / archival
    out_root: str = 'runs'
    log_interval: int = 10               # steps: loss/lr/throughput scalars
    slow_interval: int = 100             # steps: param-norm scalars
    permanent_ckpt_interval: int | None = None   # steps; None = only final
    recent_ckpt_minutes: float = 30.0
    peak_tflops: float | None = None     # per-device; enables MFU logging

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'TrainConfig':
        '''A train recipe may reference a model recipe (a yaml holding only
        model_name/model_args) via `model_recipe: <path relative to this
        file>`; the two must not both define the model. The resolved config
        written into the run dir always carries the merged result.'''
        path = Path(path)
        raw = yaml.load(open(path, encoding='utf-8'), _YamlLoader) or {}
        if 'model_recipe' in raw:
            mpath = (path.parent / raw.pop('model_recipe')).resolve()
            model = yaml.load(open(mpath, encoding='utf-8'), _YamlLoader) or {}
            if not set(model) <= {'model_name', 'model_args'}:
                raise ValueError(f'{mpath} is not a pure model recipe')
            if set(model) & set(raw):
                raise ValueError('model defined in both train and model recipe')
            raw |= model
        return cls(**raw)   # unknown keys -> loud TypeError


# ---------------------------------------------------------------------------
# Distributed / environment
# ---------------------------------------------------------------------------

def setup_distributed(config: TrainConfig):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device(f'cuda:{local_rank}'), True

    if config.device != 'auto':
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return 0, 1, device, False


def unwrap(model: nn.Module) -> nn.Module:
    model = getattr(model, '_orig_mod', model)
    model = getattr(model, 'module', model)
    return model


_STOP = False


def _stop_handler(_signum, _frame):
    global _STOP
    _STOP = True


# ---------------------------------------------------------------------------
# Checkpointing: sparse permanent (ckpt-STEP.pt, kept) + rolling recent
# (recent.pt + recent-prev.pt, time-based) — both atomic via tmp+rename.
# ---------------------------------------------------------------------------

def save_checkpoint(run_dir: Path, kind: str, step: int, tokens_seen: int,
                    model, optimizer, loader, config: TrainConfig):
    state = {
        'step': step,
        'tokens_seen': tokens_seen,
        'model_name': config.model_name,
        'model_args': config.model_args,
        'model': unwrap(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'loader': loader.state_dict(),
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
        },
        'config': asdict(config),
    }
    path = run_dir / ('recent.pt' if kind == 'recent' else f'ckpt-{step:08d}.pt')
    # rotate before writing: if the write crashes, recent.pt is absent but
    # recent-prev.pt survives, and find_resume falls through to it
    if kind == 'recent' and path.exists():
        os.replace(path, run_dir / 'recent-prev.pt')
    atomic_write(path, lambda f: torch.save(state, f))
    return path


def find_resume(run_dir: Path) -> Path | None:
    for name in ('recent.pt', 'recent-prev.pt'):
        if (run_dir / name).exists():
            return run_dir / name
    ckpts = sorted(run_dir.glob('ckpt-*.pt'))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    global _STOP
    _STOP = False   # train() is library-callable; don't inherit a prior run's signal
    rank, world, device, is_dist = setup_distributed(config)
    is_main = rank == 0

    if world != config.world_size:
        raise ValueError(
            f'launched with world_size={world} but the recipe declares '
            f'world_size={config.world_size}. The data order depends on the '
            f'card count, so this would be a different experiment; edit the '
            f'recipe deliberately if the new layout is intended.')
    
    declared = (config.world_size * config.batch_size
                * config.grad_accum_steps * config.context_len)
    
    if config.global_batch_tokens is not None \
            and declared != config.global_batch_tokens:
        raise ValueError(
            f'world*batch*accum*ctx = {declared:,} does not match the declared '
            f'global_batch_tokens = {config.global_batch_tokens:,}')

    def log(*a):
        if is_main:
            print(f'[{datetime.now().strftime("%H:%M:%S")}]', *a, flush=True)

    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGUSR1, _stop_handler)

    torch.manual_seed(config.seed)          # same init on every rank
    np.random.seed(config.seed)
    if config.tf32 and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    code = git_state()
    commit8 = (code['commit'] or 'nogit')[:8]
    run_dir = Path(config.out_root) / f'{config.run_name}-{commit8}'
    resume_from = find_resume(run_dir) if run_dir.exists() else None
    if run_dir.exists() and resume_from is None and any(run_dir.iterdir()):
        raise FileExistsError(
            f'{run_dir} exists without checkpoints; refusing a silent overwrite')

    # data
    store = TokenStore(config.data_dir, shards=config.data_shards)
    loader = WindowLoader(store, config.context_len, config.batch_size,
                          seed=config.seed, rank=rank, world_size=world)
    tokens_per_step = world * config.batch_size * config.grad_accum_steps * config.context_len
    steps_total = int(config.train_tokens) // tokens_per_step
    assert steps_total > 0

    # model + optimizer + schedule
    model = build_model(config.model_name, config.model_args).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(config.optimizer_name, model, config.optimizer_args)
    sched = build_schedule(config.schedule_name, config.schedule_args)

    log(f'run {run_dir.name} | device {device} x{world} | '
        f'{param_count:,} params | data {store.total_tokens:,} tokens '
        f'({len(store.entries)} shards)')
    log(f'plan: {steps_total:,} steps x {tokens_per_step:,} tokens/step '
        f'= {steps_total * tokens_per_step / 1e9:.2f}B tokens '
        f'({steps_total * tokens_per_step / store.total_tokens:.2f} epochs)')

    if is_main and resume_from is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.yaml').write_text(
            yaml.safe_dump(asdict(config), sort_keys=False))
        (run_dir / 'meta.json').write_text(json.dumps({
            'created': datetime.now().astimezone().isoformat(timespec='seconds'),
            'code': code,
            'world_size': world,
            'device': str(device),
            'torch': torch.__version__,
            'config': asdict(config),
            'data': {
                # witnessed fact: the physical location actually read on
                # this machine (config may hold a symlinked relative name)
                'dir': str(Path(config.data_dir).resolve()),
                'source': store.manifest['source'],
                'tokenizer': {k: store.manifest['tokenizer'][k] for k in ('id', 'sha256')},
                'total_tokens': store.total_tokens,
                'shards': [e['file'] for e in store.entries],
            },
            'derived': {
                'steps_total': steps_total,
                'tokens_per_step': tokens_per_step,
                'param_count': param_count,
                'params_by_group': {k: sum(p.numel() for p in v)
                                    for k, v in unwrap(model).param_groups().items()},  # type: ignore
                'windows': loader.window_count,
                'batches_per_epoch': loader.batches_per_epoch,
            },
        }, indent=2))
        log(f'fresh run: wrote config.yaml + meta.json')

    step, tokens_seen = 0, 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=False)
        assert state['config']['model_args'] == config.model_args, \
            'checkpoint model_args differ from config'
        unwrap(model).load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        loader.load_state_dict(state['loader'])
        torch.set_rng_state(state['rng']['torch'].cpu())
        if state['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state['rng']['cuda']])
        np.random.set_state(state['rng']['numpy'])
        step, tokens_seen = state['step'], state['tokens_seen']
        log(f'resumed from {resume_from.name} at step {step:,} '
            f'({tokens_seen/1e9:.3f}B tokens)')

    if config.compile:
        t0 = time.perf_counter()
        if hasattr(model, 'compile_blocks'):
            model.compile_blocks()  # type: ignore
        else:
            model = cast(nn.Module, torch.compile(model))
        log(f'compile requested ({time.perf_counter() - t0:.1f}s setup; '
            f'first step pays the real cost)')
    if is_dist:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    amp_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}[config.dtype]
    autocast = (torch.autocast(device.type, dtype=amp_dtype)
                if amp_dtype is not torch.float32 else nullcontext())
    assert amp_dtype is not torch.float16, 'fp16 (GradScaler) path not implemented'

    margs = config.model_args
    ctx = MetricCtx(
        model=unwrap(model), optimizer=optimizer, world_size=world,
        device_type=device.type, param_count=param_count,
        peak_tflops=config.peak_tflops,
        attn_flops_per_tok=(12 * margs['layer_count'] * margs['dim'] * config.context_len
                            if {'layer_count', 'dim'} <= margs.keys() else 0))
    # constructed on every rank (its loss all-reduce is a collective);
    # only rank 0 gets the tb writer, and log() is already rank-gated.
    # models contribute architecture-specific hooks via the optional
    # metric_hooks() convention (see metrics.py).
    hooks = getattr(unwrap(model), 'metric_hooks', dict)()
    monitor = Monitor(
        ctx, tokens_per_step, steps_total,
        log_interval=config.log_interval, slow_interval=config.slow_interval,
        fast=FAST_METRICS + list(hooks.get('fast', [])),
        slow=SLOW_METRICS + list(hooks.get('slow', [])),
        tb_dir=run_dir / 'tb' if is_main else None,
        meta_text=f'```json\n{(run_dir / "meta.json").read_text()}\n```'
                  if is_main else None,
        log_fn=log)

    # one uniform loss path: the model yields pre-head hidden states and
    # the head projection lives inside the loss fn (for 'full' that is the
    # same computation head() would have done)
    head_weight = unwrap(model).head.weight  # type: ignore
    ce_fn = make_ce(
        config.ce_impl, z_loss=config.z_loss or 0.0,
        chunk_rows=config.ce_chunk_rows,
        compute_dtype=amp_dtype if amp_dtype is not torch.float32 else None)

    batches = iter(loader)
    model.train()
    last_recent = time.time()
    log('training...')

    try:
        while step < steps_total:
            step += 1
            progress = step / steps_total
            mult = sched(progress)
            for g in optimizer.param_groups:
                g['lr'] = g['base_lr'] * mult

            optimizer.zero_grad(set_to_none=True)
            loss_acc = torch.zeros((), device=device)
            for micro in range(config.grad_accum_steps):
                x, y = next(batches)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                sync = (model.no_sync()  # type: ignore
                        if is_dist and micro < config.grad_accum_steps - 1
                        else nullcontext())
                with sync, autocast:
                    hidden = model(x, return_hidden=True)
                    loss = ce_fn(hidden, head_weight, y)
                    (loss / config.grad_accum_steps).backward()
                loss_acc += loss.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip or float('inf'))
            optimizer.step()
            tokens_seen += tokens_per_step

            stop = _STOP
            if is_dist:
                flag = torch.tensor(float(stop), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop = bool(flag.item())

            monitor.observe(step, loss_acc / config.grad_accum_steps,
                            grad_norm, tokens_seen,
                            final=stop or step == steps_total)

            if is_main and config.permanent_ckpt_interval \
                    and step % config.permanent_ckpt_interval == 0 and step < steps_total:
                p = save_checkpoint(run_dir, 'permanent', step, tokens_seen,
                                    model, optimizer, loader, config)
                log(f'permanent checkpoint: {p.name}')
            if is_main and time.time() - last_recent > config.recent_ckpt_minutes * 60:
                save_checkpoint(run_dir, 'recent', step, tokens_seen,
                                model, optimizer, loader, config)
                last_recent = time.time()
                log(f'recent checkpoint at step {step:,}')

            if stop:
                log(f'stop signal received at step {step:,}; checkpointing and exiting')
                break

        if is_main:
            kind = 'recent' if _STOP else 'permanent'
            p = save_checkpoint(run_dir, kind, step, tokens_seen,
                                model, optimizer, loader, config)
            log(f'final {kind} checkpoint: {p.name} | '
                f'{tokens_seen/1e9:.3f}B tokens seen')
    finally:
        monitor.close()
        if is_dist:
            dist.destroy_process_group()   # no barrier: see job 27725880


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='path to a TrainConfig yaml')
    args = ap.parse_args()
    print(f'[config] loading {args.config} '
          f'(sci-notation resolver active: bare 1e-4 parses as float)')
    train(TrainConfig.from_yaml(args.config))


if __name__ == '__main__':
    main()
