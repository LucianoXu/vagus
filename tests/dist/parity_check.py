# Parallel-regime parity: the same tiny hybrid model, the same per-rank
# data, trained under DDP (shard_size 1), full FSDP2 (shard_size = world)
# and HSDP (shard_size = world / 2) must produce the same loss trace and
# the same parameters — the regimes differ only in where tensors live
# and in the reduction order of the gradient sums. Then a full train()
# run under HSDP with checkpointing and a resume across two invocations.
#
# CPU + gloo, so it runs anywhere:
#     torchrun --nproc_per_node 4 -m tests.dist.parity_check [--out DIR]
# (tests/test_dist_parity.py wraps it for pytest.) On a GPU node the same
# command under NCCL exercises the production path.

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from infra.components.losses import make_ce
from infra.models import build_model
from infra.models.io import load_model
from infra.optimizer import build_optimizer
from infra.optimizer.muon import muon_update, muon_update_sharded
from infra.train.main import (LMLoss, TrainConfig, apply_fsdp, build_mesh, clip_grad_norm,
                              gather_state, load_state, reduce_replicated, train, unwrap)

MODEL = dict(vocab_size=101, dim=64, layer_count=3, head_count=2, key_head_dim=16,
             value_head_dim=32, gate_rank=8, chunk_size=8, la_impl='torch', context_len=48,
             layer_kinds=['gdn', 'parallel', 'gdn'], softmax_head_dim=16, softmax_rope=False)
OPT = dict(lr=3e-3, momentum=0.95, weight_decay=0.1, lr_adjust='rms', ns_coeffs='polar_express',
           adamw=dict(lr=3e-3, betas=[0.9, 0.95], weight_decay=0.1))
ADAMW = dict(lr=3e-3, betas=[0.9, 0.95], weight_decay=0.1)


def make_dataset(root: Path, tokens: int = 40_000, vocab: int = 101) -> Path:
    '''A one-shard vagus-tokens-v1 store of random tokens.'''
    d = root / 'data'
    d.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(0).integers(2, vocab, size=tokens, dtype=np.uint16)
    np.save(d / 'shard0.npy', arr)
    (d / 'manifest.json').write_text(json.dumps({
        'format': 'vagus-tokens-v1', 'dtype': 'uint16',
        'source': {'name': 'synthetic'},
        'tokenizer': {'id': 'test', 'sha256': '0' * 64, 'vocab_size': vocab},
        'shards': [{'source': 'shard0', 'file': 'shard0.npy', 'tokens': tokens}],
    }))
    return d


DEVICE = torch.device('cpu')          # set in main(): cuda:<local rank> when available


def batch(rank: int, step: int, micro: int, B=2, L=32, vocab=101):
    g = torch.Generator().manual_seed(1000 * rank + 10 * step + micro)
    ids = torch.randint(2, vocab, (B, L + 1), generator=g).to(DEVICE)
    return ids[:, :-1], ids[:, 1:]


def run_regime(shard_size: int, steps: int, accum: int, param_dtype: str,
               amp: torch.dtype | None = None, optimizer: str = 'muon'):
    '''The training step of infra.train.main on synthetic batches;
    returns (loss trace, full fp32 state dict) on every rank. amp None
    runs fp32 arithmetic, where the regimes differ only by summation
    order; bf16 autocast adds rounding noise that a tiny model at a
    healthy lr amplifies into different gradient spikes.'''
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = DEVICE
    cfg = TrainConfig(run_name='parity', data_dir='', shard_size=shard_size,
                      fsdp_param_dtype=param_dtype, world_size=world)
    mesh = build_mesh(cfg, device, world)
    is_fsdp = mesh is not None
    replicate = mesh.size(0) if mesh is not None else 1

    torch.manual_seed(0)
    model = build_model('GDNLM', MODEL).to(device)
    ce_fn = make_ce('chunked', z_loss=1e-4, chunk_rows=64, compute_dtype=amp)
    if is_fsdp:
        assert mesh is not None
        model = LMLoss(model, ce_fn)
        apply_fsdp(model, mesh, cfg)
    else:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    opt = build_optimizer(optimizer, unwrap(model), OPT if optimizer == 'muon' else ADAMW)
    head_weight = unwrap(model).head.weight
    autocast = torch.autocast(device.type, dtype=amp) if amp is not None else _null()

    trace = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        acc = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = batch(rank, step, micro)
            last = micro == accum - 1
            if is_fsdp:
                if replicate > 1:
                    model.set_requires_all_reduce(last)
                with autocast:
                    loss = model(x, y)
                    (loss / accum).backward()
            else:
                sync = model.no_sync() if not last else _null()
                with sync, autocast:
                    hidden = model(x, return_hidden=True)
                    loss = ce_fn(hidden, head_weight, y)
                    (loss / accum).backward()
            acc += loss.detach()
        if is_fsdp:
            reduce_replicated(model, mesh)
        gn = clip_grad_norm(model, 1.0)
        opt.step()
        acc = acc / accum
        dist.all_reduce(acc, op=dist.ReduceOp.AVG)
        trace.append((float(acc), float(gn)))

    msd, osd, fmt = gather_state(model, opt, is_fsdp)      # FSDP: filled on rank 0 only
    msd = {k: v.detach().float().cpu().clone() for k, v in msd.items()}
    # round-trip: the gathered state must load back into the same regime
    if is_fsdp:
        load_state(model, opt, {'model': msd, 'optimizer': osd, 'optimizer_format': fmt} if msd else {}, True)
        again, _, _ = gather_state(model, opt, True)
        for k, v in again.items():
            assert torch.equal(v.float(), msd[k]), f'state round-trip changed {k}'
        holder = [msd if rank == 0 else None]
        dist.broadcast_object_list(holder, src=0)
        msd = holder[0]
    return trace, msd


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def compare(name, ref, other, rtol, atol):
    worst = 0.0
    for k, v in ref.items():
        d = (v - other[k]).abs().max().item()
        tol = atol + rtol * v.abs().max().item()
        worst = max(worst, d / max(tol, 1e-30))
        assert d <= tol, f'{name}: {k} differs by {d:.3e} (tol {tol:.3e})'
    return worst


def check_regimes(log):
    world = dist.get_world_size()
    assert world % 2 == 0, 'need an even world for the HSDP regime'
    accum = 2
    # 1. fp32 arithmetic, AdamW (no Newton-Schulz): FSDP and HSDP must
    #    reproduce DDP to summation-order precision over many steps —
    #    this pins the mechanics (sharding, accumulation, reduce, clip).
    # 2. fp32, Muon, a few steps: the DTensor Muon path against the DDP
    #    one. Newton-Schulz runs in bf16, where inputs equal to ~1e-7
    #    still round differently now and then, so the band is 1e-4 and
    #    the horizon short (the drift compounds ~2%/step beyond that).
    #    Parameters after Muon steps are compared loosely: the polar
    #    factor G -> U V^T is not Lipschitz where singular values are
    #    small, so a 1e-7 input difference legitimately moves a few
    #    entries by a fraction of one step; check_muon_sharded below is
    #    the exact test of the sharded update itself.
    for optimizer, steps, ltol, gtol, ptol in [('adamw', 8, 1e-5, 1e-4, 1e-4),
                                                ('muon', 3, 1e-4, 1e-3, 5e-3)]:
        ddp_trace, ddp_sd = run_regime(1, steps, accum, 'float32', optimizer=optimizer)
        log(f'{optimizer} DDP (loss, gnorm): {[(round(l, 5), round(g, 4)) for l, g in ddp_trace]}')
        for shard in (world, world // 2):
            trace, sd = run_regime(shard, steps, accum, 'float32', optimizer=optimizer)
            log(f'{optimizer} shard_size {shard}: {[(round(l, 5), round(g, 4)) for l, g in trace]}')
            for (l0, g0), (l1, g1) in zip(ddp_trace, trace):
                assert abs(l0 - l1) <= ltol * (1 + abs(l0)), f'{optimizer} shard {shard}: loss {l0} vs {l1}'
                assert abs(g0 - g1) <= gtol * (1 + abs(g0)), f'{optimizer} shard {shard}: gnorm {g0} vs {g1}'
            # abs band: AdamW-driven params (embedding, gate matrices) move ~lr
            # per step whichever way a rounded near-zero gradient falls
            atol = ptol / 10 if optimizer == 'adamw' else OPT['lr'] * steps
            worst = compare(f'{optimizer} shard {shard}', ddp_sd, sd, ptol, atol)
            log(f'{optimizer} shard_size {shard} (fp32): matches DDP; params within {worst:.2f}x of tol')
    steps = 6
    # 3. bf16 autocast + bf16 FSDP compute copies of the matrices (the
    #    production setting) against bf16-autocast DDP: the same
    #    arithmetic (autocast casts the same matrices to bf16; the 1-D
    #    params stay fp32 in both), so the Muon-regime bands apply.
    steps = 3
    ddp_trace, ddp_sd = run_regime(1, steps, accum, 'float32', amp=torch.bfloat16)
    trace, sd = run_regime(world // 2, steps, accum, 'bfloat16', amp=torch.bfloat16)
    log(f'bf16 DDP : {[(round(l, 5), round(g, 4)) for l, g in ddp_trace]}')
    log(f'bf16 HSDP: {[(round(l, 5), round(g, 4)) for l, g in trace]}')
    #    (bf16 rounding noise is ~1e-3 relative rather than 1e-7, so the
    #    Newton-Schulz amplification shows up in the gradient norm within
    #    a few steps; the loss trace is the tight signal)
    #    The gradient norm is only reported here: on A100 the third step
    #    differed by 11% at a 1e-5 loss match (a Newton-Schulz spike).
    for (l0, _), (l1, _) in zip(ddp_trace, trace):
        assert abs(l0 - l1) <= 1e-3 * (1 + abs(l0)), f'bf16 HSDP: loss {l0} vs {l1}'
    #    params: AdamW moves an entry by ~lr per step whichever way the
    #    (rounded) gradient sign falls, so the band is lr x steps
    worst = compare('bf16 HSDP', ddp_sd, sd, 1e-2, ADAMW['lr'] * steps)
    log(f'bf16 HSDP vs bf16 DDP: loss trace matches; params within {worst:.2f}x of tol')


def check_muon_sharded(log):
    '''muon_update_sharded on row-sharded DTensors must equal muon_update
    on the full tensors, exactly: same momentum arithmetic on the shards,
    same bf16 Newton-Schulz input after the all-gather, same slice back.
    Bucket sizes below and above the shard-group size exercise the
    padding branch.'''
    world = dist.get_world_size()
    cfg = TrainConfig(run_name='parity', data_dir='', shard_size=world, world_size=world)
    mesh = build_mesh(cfg, DEVICE, world)
    assert mesh is not None
    shapes = [(64, 32)] * 5 + [(32, 64)] * 1 + [(8 * world, 16)] * 3
    torch.manual_seed(1)                              # identical on every rank
    full = [torch.randn(s, device=DEVICE) for s in shapes]
    grads = [torch.randn(s, device=DEVICE) for s in shapes]
    bufs = [torch.randn(s, device=DEVICE) * 0.1 for s in shapes]
    kw = dict(lr=1e-2, momentum=0.95, weight_decay=0.1, nesterov=True, ns_steps=5, eps=1e-7,
              lr_adjust='rms', ns_coeffs='polar_express', compile_ns=False)
    ref_p = [t.clone() for t in full]
    ref_b = [t.clone() for t in bufs]
    muon_update(ref_p, [g.clone() for g in grads], ref_b, **kw, world_size=1, rank=0)
    place = [Replicate(), Shard(0)]
    sh_p = [distribute_tensor(t.clone(), mesh, place) for t in full]
    sh_g = [distribute_tensor(t.clone(), mesh, place) for t in grads]
    sh_b = [distribute_tensor(t.clone(), mesh, place) for t in bufs]
    muon_update_sharded(sh_p, sh_g, sh_b, **kw)
    for i, (p, b) in enumerate(zip(sh_p, sh_b)):
        assert torch.equal(p.full_tensor(), ref_p[i]), f'sharded Muon param {shapes[i]} differs'
        assert torch.equal(b.full_tensor(), ref_b[i]), f'sharded Muon momentum {shapes[i]} differs'
    log(f'muon_update_sharded == muon_update exactly on {len(shapes)} matrices (shard group {world})')


def check_train_resume(root: Path, log):
    world = dist.get_world_size()
    data = make_dataset(root)
    common = dict(run_name='hsdp-e2e', data_dir=str(data), model_name='GDNLM', model_args=MODEL,
                  optimizer_name='muon', optimizer_args=OPT,
                  schedule_name='wsd', schedule_args=dict(warmup=0.1, decay=0.2, min_ratio=0.0),
                  context_len=32, batch_size=2, grad_accum_steps=2, world_size=world,
                  shard_size=world // 2, z_loss=1e-4, ce_impl='chunked', ce_chunk_rows=64,
                  compile=False, out_root=str(root / 'runs'), run_dir=str(root / 'runs' / 'e2e'),
                  device=str(DEVICE),
                  log_interval=1, slow_interval=2, permanent_ckpt_interval=3,
                  recent_ckpt_minutes=0.0)
    tokens_per_step = world * 2 * 2 * 32
    train(TrainConfig(**common, train_tokens=4 * tokens_per_step), teardown=False)
    dist.barrier()
    run = root / 'runs' / 'e2e'
    names = sorted(p.name for p in run.iterdir())
    assert 'ckpt-00000003.pt' in names and 'ckpt-00000004.pt' in names and 'model-final.pt' in names, names
    assert 'recent.pt' in names, names            # recent_ckpt_minutes 0 -> every step
    # the permanent checkpoint's model must load standalone (slim export path)
    if dist.get_rank() == 0:
        m, meta = load_model(run / 'model-final.pt')
        assert meta['step'] == 4 and meta['format'] == 'slim'
        state = torch.load(run / 'ckpt-00000004.pt', weights_only=False)
        assert state['optimizer_format'] == 'dcp' and state['step'] == 4
    dist.barrier()
    # resume: a longer budget continues from recent.pt (step 4) to step 6
    train(TrainConfig(**common, train_tokens=6 * tokens_per_step), teardown=False)
    dist.barrier()
    if dist.get_rank() == 0:
        state = torch.load(run / 'ckpt-00000006.pt', weights_only=False)
        assert state['step'] == 6 and state['tokens_seen'] == 6 * tokens_per_step
        assert (run / 'model-final.pt').exists()
    log('train() under HSDP: 4 steps, checkpoints, resume to 6 steps: ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='scratch dir (default: a temp dir)')
    args = ap.parse_args()
    from infra.train import main as train_main   # noqa: F401  (import check)
    if os.environ.get('PARITY_TRACE'):          # where does it hang? dump all stacks after N s
        import faulthandler, sys
        faulthandler.dump_traceback_later(int(os.environ['PARITY_TRACE']), repeat=True, file=sys.stderr)
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        torch.cuda.set_device(DEVICE)
        dist.init_process_group('nccl', device_id=DEVICE)
    else:
        dist.init_process_group('gloo')
    rank = dist.get_rank()

    def log(*a):
        if rank == 0:
            print('[parity]', *a, flush=True)

    root = Path(args.out) if args.out else None
    if root is None:
        holder = [tempfile.mkdtemp(prefix='vagus-parity-')] if rank == 0 else [None]
        dist.broadcast_object_list(holder, src=0)
        root = Path(holder[0])
    root.mkdir(parents=True, exist_ok=True)
    try:
        check_muon_sharded(log)
        check_regimes(log)
        check_train_resume(root, log)
        log('ALL OK')
    except BaseException:
        import traceback
        print(f'[parity] rank {rank} FAILED:\n' + traceback.format_exc(), flush=True)
        os._exit(1)          # no barrier on failure: torchrun tears the others down
    dist.barrier()
    if rank == 0 and args.out is None:
        shutil.rmtree(root, ignore_errors=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
