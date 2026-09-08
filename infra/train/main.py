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
#
# Parallelism, one knob (shard_size), two regimes:
# - shard_size 1: DDP. Every rank holds the full fp32 model, grads and
#   optimizer state; Muon shards only its Newton-Schulz work.
# - shard_size N: FSDP2 over a 2-D DeviceMesh (replicate, shard) =
#   (world / N, N). Params, grads and optimizer state are row-sharded
#   over the N ranks of a shard group (fp32 masters; a bf16 copy is
#   all-gathered per block for compute) and replicated across groups.
#   With N = the GPUs of one node this is HSDP: the per-block all-gathers
#   stay on NVLink and only the gradient reduce crosses the network,
#   once per optimizer step. Data parallelism spans the whole world in
#   both regimes (the loader shards by global rank), so a recipe's data
#   order depends on world_size alone, not on shard_size.

import argparse
import json
import os
import re
import signal
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP

from ..components.losses import make_ce
from ..dataset.loader import TokenStore, WindowLoader
from ..models import build_model
from ..models.io import export_slim
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


_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
           'float32': torch.float32}


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

    # parallelism (see module header): ranks per FSDP2 shard group. 1 =
    # DDP; the GPUs of one node = HSDP. world_size must be a multiple.
    shard_size: int = 1
    # FSDP2 mixed precision for the block matrices: the dtype of their
    # all-gathered compute copy ('bfloat16' halves the all-gather bytes
    # and is what autocast would cast them to anyway; 'float32' keeps
    # the fp32 masters in compute) and of the gradient reduction. The
    # 1-D params (norm gammas, gates), the embedding/head and the
    # residual stream stay fp32 in both cases, so the arithmetic is the
    # DDP + autocast arithmetic exactly.
    fsdp_param_dtype: str = 'bfloat16'
    fsdp_reduce_dtype: str = 'float32'
    # Free each block's gathered compute copy after its forward (True:
    # re-gather in backward, FSDP2's default) or keep it until backward
    # (False: one all-gather per block per micro-step instead of two, at
    # the cost of holding every block's bf16 copy — 0.7 GB at 340M).
    fsdp_reshard_after_forward: bool = True

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
    # explicit run directory; default out_root/<run_name>-<commit8>. Set
    # it to continue a run's directory deliberately across code commits
    # (the default name changes with the commit, and a resubmit from a
    # newer checkout would otherwise start a fresh run beside it).
    run_dir: str | None = None
    log_interval: int = 10               # steps: loss/lr/throughput scalars
    slow_interval: int = 100             # steps: param-norm scalars
    permanent_ckpt_interval: int | None = None   # steps; None = only final
    recent_ckpt_minutes: float = 30.0
    peak_tflops: float | None = None     # per-device; enables MFU logging

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] = ()) -> 'TrainConfig':
        '''A train recipe may reference a model recipe (a yaml holding only
        model_name/model_args) via `model_recipe: <path relative to this
        file>`; the two must not both define the model. The resolved config
        written into the run dir always carries the merged result.

        `overrides` are `key=value` strings for top-level fields (the
        value parsed as yaml: `shard_size=4`, `fsdp_param_dtype=float32`,
        `run_dir=runs/x`), for launch-time variations of one recipe such
        as smoke runs; an experiment's recipe file stays the record.'''
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
        for item in overrides:
            key, sep, value = item.partition('=')
            if not sep or not key:
                raise ValueError(f'override {item!r} is not key=value')
            raw[key] = yaml.load(value, _YamlLoader)
        return cls(**raw)   # unknown keys -> loud TypeError


# ---------------------------------------------------------------------------
# Distributed / environment
# ---------------------------------------------------------------------------

def setup_distributed(config: TrainConfig):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
            if not dist.is_initialized():      # a caller may own the group
                dist.init_process_group(backend='nccl', device_id=device)
        else:                              # CPU (gloo): the parity tests
            device = torch.device('cpu')
            if not dist.is_initialized():
                dist.init_process_group(backend='gloo')
        return rank, world, device, True

    if config.device != 'auto':
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return 0, 1, device, False


def build_mesh(config: TrainConfig, device: torch.device, world: int) -> DeviceMesh | None:
    '''The 2-D (replicate, shard) DeviceMesh of the FSDP2 regime, None
    for DDP. Rank r sits at (r // shard_size, r % shard_size): consecutive
    ranks form a shard group, which under torchrun is one node.'''
    if config.shard_size <= 1:
        return None
    if world % config.shard_size:
        raise ValueError(f'world_size {world} is not a multiple of shard_size {config.shard_size}')
    return init_device_mesh(device.type, (world // config.shard_size, config.shard_size),
                            mesh_dim_names=('replicate', 'shard'))


class LMLoss(nn.Module):
    '''FSDP2 root module: forward computes the loss. The fused CE reads
    the (tied) head weight directly, and a parameter is only unsharded
    inside the forward of the FSDP module that owns it, so the loss has
    to be computed inside the root's forward rather than in the loop.
    `replicated` lists the params FSDP ignores (the 1-D ones): plain
    fp32 tensors on every rank whose grads the loop all-reduces itself.'''

    def __init__(self, module: nn.Module, ce_fn):
        super().__init__()
        self.module = module
        self.ce_fn = ce_fn
        self.replicated: list[nn.Parameter] = []

    def forward(self, x, y):
        hidden = self.module(x, return_hidden=True)
        return self.ce_fn(hidden, self.module.head.weight, y)


def apply_fsdp(root: LMLoss, mesh: DeviceMesh, config: TrainConfig) -> None:
    '''One FSDP2 unit per block plus the root (embedding/head). Block
    matrices compute in fsdp_param_dtype; the root keeps fp32 (embedding
    output = the residual stream, head weight for the CE); block inputs
    are not cast; and every 1-D parameter (norm gammas, gate biases,
    A_log) is left out of FSDP altogether — replicated fp32, 75k
    elements at 340M — because a bf16 compute copy would round exactly
    the quantities autocast leaves in fp32 (norms, softplus gates). The
    arithmetic is then DDP + autocast's, with the all-gathers in bf16.
    Call after compile: the FSDP hooks then wrap the compiled forward.'''
    param_dtype = _DTYPES[config.fsdp_param_dtype]
    reduce_dtype = _DTYPES[config.fsdp_reduce_dtype]
    blocks_mp = MixedPrecisionPolicy(
        param_dtype=None if param_dtype is torch.float32 else param_dtype,
        reduce_dtype=reduce_dtype, cast_forward_inputs=False)
    root_mp = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=reduce_dtype,
                                   cast_forward_inputs=False)
    root.replicated = [p for p in root.parameters() if p.dim() < 2]
    ignored = set(root.replicated)
    for blk in root.module.blocks:  # type: ignore[union-attr]
        fully_shard(blk, mesh=mesh, mp_policy=blocks_mp, ignored_params=ignored,
                    reshard_after_forward=config.fsdp_reshard_after_forward)
    fully_shard(root, mesh=mesh, mp_policy=root_mp, ignored_params=ignored)


def reduce_replicated(root: LMLoss, mesh: DeviceMesh) -> None:
    '''Average the grads of the FSDP-ignored params over the whole mesh
    (one flat all-reduce), the sync FSDP does not do for them.'''
    grads = [p.grad for p in root.replicated if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.flatten() for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    torch._foreach_copy_(grads, list(flat.split([g.numel() for g in grads])))  # type: ignore[attr-defined]


def unwrap(model: nn.Module) -> nn.Module:
    model = getattr(model, '_orig_mod', model)
    model = getattr(model, 'module', model)
    return model


def optimizer_members(optimizer) -> list[torch.optim.Optimizer]:
    '''The torch optimizers behind the factory's object (a composite
    such as MuonAdamW surfaces its members), for the DCP state-dict API.'''
    return list(getattr(optimizer, 'members', [optimizer]))


@contextmanager
def unsharded(model: nn.Module, compute=None):
    '''All FSDP2 units' params materialised as plain tensors, so probe
    code can call submodules directly; resharded on exit. The gathered
    copies are in the units' compute dtype, so a bf16 policy needs the
    probes under autocast (`compute`: a context manager) — the same
    arithmetic the training forward runs.'''
    units = [m for m in model.modules() if isinstance(m, FSDPModule)]
    for m in units:
        m.unshard()
    try:
        with (compute if compute is not None else nullcontext()):
            yield
    finally:
        for m in units:
            m.reshard()


def clip_grad_norm(model: nn.Module, max_norm: float | None) -> torch.Tensor:
    '''Global grad norm, clipped in place when max_norm is set. DTensor-
    aware: the FSDP2 sharded grads' partial norms are completed over
    the mesh and combined with the replicated (plain) grads' norm
    before the one clip coefficient is applied to both.'''
    params = [p for p in model.parameters() if p.grad is not None]
    sharded = [p for p in params if isinstance(p.grad, DTensor)]
    plain = [p for p in params if not isinstance(p.grad, DTensor)]
    sq = torch.zeros((), device=params[0].grad.device if params else 'cpu')  # type: ignore[union-attr]
    for group in (sharded, plain):
        if group:
            n = torch.nn.utils.get_total_norm(cast(list[torch.Tensor], [p.grad for p in group]), 2.0)
            if isinstance(n, DTensor):
                n = n.full_tensor()
            sq = sq + n.float() ** 2
    total = sq.sqrt()
    if max_norm is not None:
        for group in (sharded, plain):
            if group:
                torch.nn.utils.clip_grads_with_norm_(group, max_norm, total)
    return total


_STOP = False


def _stop_handler(_signum, _frame):
    global _STOP
    _STOP = True


# ---------------------------------------------------------------------------
# Checkpointing: sparse permanent (ckpt-STEP.pt, kept) + rolling recent
# (recent.pt + recent-prev.pt, time-based) — both atomic via tmp+rename.
# One file format for both regimes: full (unsharded) fp32 tensors on
# CPU. Under FSDP2 every rank takes part in the gathers but only rank 0
# receives the result (DCP's cpu_offload semantics) and writes it; on
# resume only rank 0 reads the file and the DCP loaders broadcast from
# it. The optimizer state is then in the DCP (FQN-keyed) layout, tagged
# so that resume picks the matching loader.
# ---------------------------------------------------------------------------

_FULL = StateDictOptions(full_state_dict=True, cpu_offload=True)
_FROM_RANK0 = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)


def _strip(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def gather_state(model, optimizer, is_fsdp: bool) -> tuple[dict, dict, str]:
    '''(model state, optimizer state, format). Collective under FSDP,
    where the dicts come back filled on rank 0 and empty elsewhere.'''
    if not is_fsdp:
        return unwrap(model).state_dict(), optimizer.state_dict(), 'native'
    msd = _strip(get_model_state_dict(model, options=_FULL), 'module.')
    osd = get_optimizer_state_dict(model, optimizer_members(optimizer), options=_FULL)
    return msd, osd, 'dcp'


def load_state(model, optimizer, state: dict, is_fsdp: bool) -> None:
    '''Inverse of gather_state. Under FSDP a collective: rank 0 passes
    the checkpoint dict, the others an empty dict.'''
    if not is_fsdp:
        fmt = state.get('optimizer_format', 'native')
        assert fmt == 'native', f'{fmt} checkpoint cannot be loaded into the DDP regime'
        unwrap(model).load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        return
    if state:
        fmt = state.get('optimizer_format', 'native')
        assert fmt == 'dcp', f'{fmt} checkpoint cannot be loaded into the FSDP regime'
    msd = {'module.' + k: v for k, v in state.get('model', {}).items()}
    set_model_state_dict(model, msd, options=_FROM_RANK0)
    set_optimizer_state_dict(model, optimizer_members(optimizer),
                             optim_state_dict=state.get('optimizer', {}), options=_FROM_RANK0)


def save_checkpoint(run_dir: Path, kind: str, step: int, tokens_seen: int,
                    model, optimizer, loader, config: TrainConfig,
                    is_fsdp: bool = False, is_main: bool = True):
    msd, osd, fmt = gather_state(model, optimizer, is_fsdp)   # collective under FSDP
    if not is_main:
        return None
    state = {
        'step': step,
        'tokens_seen': tokens_seen,
        'model_name': config.model_name,
        'model_args': config.model_args,
        # identity of the token stream's vocabulary, so a checkpoint can be
        # assembled for generation without outside knowledge
        # (Generator.from_checkpoint); same fields as meta.json
        'tokenizer': {k: loader.store.manifest['tokenizer'][k] for k in ('id', 'sha256')},
        'model': msd,
        'optimizer': osd,
        'optimizer_format': fmt,
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

def train(config: TrainConfig, teardown: bool = True):
    '''teardown=False leaves the process group up for a caller that
    trains more than once per process (the parity tests).'''
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

    mesh = build_mesh(config, device, world)
    is_fsdp = mesh is not None
    replicate_size = mesh.size(0) if mesh is not None else 1

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
    run_dir = (Path(config.run_dir) if config.run_dir
               else Path(config.out_root) / f'{config.run_name}-{commit8}')
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

    amp_dtype = _DTYPES[config.dtype]
    autocast = (torch.autocast(device.type, dtype=amp_dtype)
                if amp_dtype is not torch.float32 else nullcontext())
    assert amp_dtype is not torch.float16, 'fp16 (GradScaler) path not implemented'

    # one uniform loss path: the model yields pre-head hidden states and
    # the head projection lives inside the loss fn (for 'full' that is the
    # same computation head() would have done)
    ce_fn = make_ce(
        config.ce_impl, z_loss=config.z_loss or 0.0,
        chunk_rows=config.ce_chunk_rows,
        compute_dtype=amp_dtype if amp_dtype is not torch.float32 else None)

    # model: build -> compile -> wrap -> optimizer. The optimizer must see
    # the wrapped model's parameters (FSDP2 replaces them with DTensors).
    model: nn.Module = build_model(config.model_name, config.model_args).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    params_by_group = {k: sum(p.numel() for p in v)
                       for k, v in model.param_groups().items()}  # type: ignore[operator]

    if config.compile:
        t0 = time.perf_counter()
        if hasattr(model, 'compile_blocks'):
            model.compile_blocks()  # type: ignore
        else:
            model = cast(nn.Module, torch.compile(model))
        log(f'compile requested ({time.perf_counter() - t0:.1f}s setup; '
            f'first step pays the real cost)')
    if is_fsdp:
        assert mesh is not None
        model = LMLoss(model, ce_fn)
        apply_fsdp(model, mesh, config)
    elif is_dist:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    optimizer = build_optimizer(config.optimizer_name, unwrap(model), config.optimizer_args)
    sched = build_schedule(config.schedule_name, config.schedule_args)

    layout = (f'HSDP mesh {tuple(mesh.shape)} (replicate, shard)' if mesh is not None
              else 'DDP' if is_dist else 'single')
    log(f'run {run_dir.name} | device {device} x{world} [{layout}] | '
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
            'shard_size': config.shard_size,
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
                'params_by_group': params_by_group,
                'windows': loader.window_count,
                'batches_per_epoch': loader.batches_per_epoch,
            },
        }, indent=2))
        log(f'fresh run: wrote config.yaml + meta.json')

    step, tokens_seen = 0, 0
    if resume_from is not None:
        # DDP: every rank reads the file. FSDP: rank 0 reads, the weights
        # go out through the DCP broadcast and the small rest by object
        # broadcast.
        state: dict = {}
        if is_main or not is_fsdp:
            state = torch.load(resume_from, map_location='cpu', weights_only=False)
            assert state['config']['model_args'] == config.model_args, \
                'checkpoint model_args differ from config'
        load_state(model, optimizer, state, is_fsdp)
        small = [{k: state[k] for k in ('step', 'tokens_seen', 'loader', 'rng')} if state else None]
        if is_fsdp:
            dist.broadcast_object_list(small, src=0)
        rest = small[0]
        assert rest is not None
        loader.load_state_dict(rest['loader'])
        torch.set_rng_state(rest['rng']['torch'].cpu())
        if rest['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rest['rng']['cuda']])
        np.random.set_state(rest['rng']['numpy'])
        step, tokens_seen = rest['step'], rest['tokens_seen']
        del state, rest
        log(f'resumed from {resume_from.name} at step {step:,} '
            f'({tokens_seen/1e9:.3f}B tokens)')

    ctx = MetricCtx(
        model=unwrap(model), optimizer=optimizer, world_size=world,
        device_type=device.type, param_count=param_count,
        peak_tflops=config.peak_tflops,
        attn_flops_per_tok=unwrap(model).attn_flops_per_token(config.context_len),  # type: ignore[attr-defined]
        slow_context=(lambda: unsharded(
            model, autocast if config.fsdp_param_dtype != 'float32' else None))
            if is_fsdp else None)
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

    head_weight = unwrap(model).head.weight  # type: ignore[union-attr]  (DDP loss path)
    if config.z_loss and not ce_fn.tracks_z:
        log('[warn] z-loss active but this liger-kernel cannot report the z '
            'term separately (needs return_z_loss, >= 0.5.2); training is '
            'unaffected, loss_ce/loss_z curves disabled')

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
            z_acc = torch.zeros((), device=device) if ce_fn.tracks_z else None
            for micro in range(config.grad_accum_steps):
                x, y = next(batches)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                last = micro == config.grad_accum_steps - 1
                if is_fsdp:
                    # every micro-step reduce-scatters into the sharded fp32
                    # grad (NVLink); the cross-group all-reduce runs once,
                    # on the last one
                    if replicate_size > 1:
                        model.set_requires_all_reduce(last)  # type: ignore[union-attr]
                    with autocast:
                        loss = model(x, y)
                        (loss / config.grad_accum_steps).backward()
                else:
                    sync = model.no_sync() if is_dist and not last else nullcontext()  # type: ignore
                    with sync, autocast:
                        hidden = model(x, return_hidden=True)
                        loss = ce_fn(hidden, head_weight, y)
                        (loss / config.grad_accum_steps).backward()
                loss_acc += loss.detach()
                if z_acc is not None:
                    assert ce_fn.last_z is not None   # tracks_z contract
                    z_acc += ce_fn.last_z
                ctx.last_batch = x   # for probe-style slow hooks (a
                                     # reference, not a copy)

            if is_fsdp:
                assert mesh is not None
                reduce_replicated(model, mesh)  # type: ignore[arg-type]
            grad_norm = clip_grad_norm(model, config.grad_clip)
            optimizer.step()
            tokens_seen += tokens_per_step

            # rank-consensus decisions: the stop flag (any rank's signal)
            # and the time-based recent checkpoint (any rank's clock) —
            # under FSDP the save itself is a collective, so every rank
            # must take the same branch
            want_recent = time.time() - last_recent > config.recent_ckpt_minutes * 60
            flags = torch.tensor([float(_STOP), float(want_recent)], device=device)
            if is_dist:
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            stop, want_recent = bool(flags[0].item()), bool(flags[1].item())

            monitor.observe(step, loss_acc / config.grad_accum_steps,
                            grad_norm, tokens_seen,
                            final=stop or step == steps_total,
                            z=None if z_acc is None
                              else z_acc / config.grad_accum_steps)

            if config.permanent_ckpt_interval \
                    and step % config.permanent_ckpt_interval == 0 and step < steps_total:
                p = save_checkpoint(run_dir, 'permanent', step, tokens_seen,
                                    model, optimizer, loader, config, is_fsdp, is_main)
                log(f'permanent checkpoint: {p.name if p else ""}')
            if want_recent:
                save_checkpoint(run_dir, 'recent', step, tokens_seen,
                                model, optimizer, loader, config, is_fsdp, is_main)
                last_recent = time.time()
                log(f'recent checkpoint at step {step:,}')

            if stop:
                log(f'stop signal received at step {step:,}; checkpointing and exiting')
                break

        kind = 'recent' if _STOP else 'permanent'
        p = save_checkpoint(run_dir, kind, step, tokens_seen,
                            model, optimizer, loader, config, is_fsdp, is_main)
        if is_main:
            assert p is not None
            log(f'final {kind} checkpoint: {p.name} | '
                f'{tokens_seen/1e9:.3f}B tokens seen')
            if kind == 'permanent':
                # the inference/archival artifact (weights + args, fp32):
                # what load_model and the registry's model-final.pt mean
                export_slim(p, run_dir / 'model-final.pt')
                log('wrote model-final.pt (slim)')
    finally:
        monitor.close()
        if is_dist and teardown:
            dist.destroy_process_group()   # no barrier: see job 27725880


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='path to a TrainConfig yaml')
    ap.add_argument('overrides', nargs='*', help='key=value top-level field overrides')
    args = ap.parse_args()
    print(f'[config] loading {args.config} {" ".join(args.overrides)} '
          f'(sci-notation resolver active: bare 1e-4 parses as float)')
    train(TrainConfig.from_yaml(args.config, args.overrides))


if __name__ == '__main__':
    main()
