# Evaluation entry point: recipe -> subjects x tasks -> record.
#
#   vagus-eval recipe/eval/x.yaml [key=value ...]
#
# Subject-major: each checkpoint is loaded once, every task runs on it,
# then it is freed; since a task's items depend only on (seed, args),
# pairing across subjects happens afterwards from the stored per-item
# arrays. The run directory is refused if it already holds files — like
# a train run, an eval run is never silently overwritten.

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ..config import apply_overrides, load_yaml
from ..inference import Generator
from ..utils import atomic_write, git_state
from .core import EvalCtx, Subject, TaskResult, sha256_file
from .pairs import paired
from .tasks import TASKS

_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}


@dataclass
class EvalConfig:
    eval_name: str
    # each: a checkpoint path or run dir (its model-final.pt), or a dict
    # {ckpt|run, label, tokenizer_id, run_meta}; run_meta points at the
    # training run's meta.json when it does not sit beside the checkpoint
    # (a slim copy), so the exposure witness can see the training store.
    # The first subject is the reference of the paired statistics.
    subjects: list
    tasks: list                          # [{name, args}], names from tasks.TASKS
    device: str = 'auto'
    dtype: str = 'bfloat16'
    seed: int = 0
    out_root: str = 'runs/eval'
    run_dir: str | None = None           # default out_root/<eval_name>-<commit8>
    registry_dir: str | None = 'registry/eval'   # a copy of the whole record; None = no copy
    n_boot: int = 2000

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> 'EvalConfig':
        return cls(**apply_overrides(load_yaml(path), overrides))


def resolve_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_subject(spec, device: torch.device, dtype: torch.dtype) -> Subject:
    spec = {'ckpt': spec} if isinstance(spec, str) else dict(spec)
    src = Path(spec.get('ckpt') or spec['run'])
    if src.is_dir():
        ckpt, label = src / 'model-final.pt', src.name
    else:
        ckpt, label = src, src.stem
    label = spec.get('label') or label
    gen, meta = Generator.from_checkpoint(ckpt, device=device, dtype=dtype,
                                         tokenizer_id=spec.get('tokenizer_id'), with_meta=True)
    run_meta = None
    beside = Path(spec['run_meta']) if spec.get('run_meta') else ckpt.parent / 'meta.json'
    if beside.exists():
        candidate = json.loads(beside.read_text())
        if 'config' in candidate and 'data' in candidate:   # a train run's record
            run_meta = candidate
        elif spec.get('run_meta'):
            raise ValueError(f'{beside} is not a train run meta.json')
    return Subject(label=label, path=ckpt.resolve(), sha256=sha256_file(ckpt), meta=meta,
                   generator=gen, run_meta=run_meta)


def _free(device: torch.device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def _write_json(path: Path, obj) -> None:
    atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def evaluate(config: EvalConfig) -> Path:
    code = git_state()
    commit8 = (code['commit'] or 'nogit')[:8]
    run_dir = Path(config.run_dir) if config.run_dir else Path(config.out_root) / f'{config.eval_name}-{commit8}'
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} exists; refusing a silent overwrite')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    log_file = open(run_dir / 'log.txt', 'a', encoding='utf-8')

    def log(msg: str):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    device = resolve_device(config.device)
    dtype = _DTYPES[config.dtype]
    for t in config.tasks:
        if t['name'] not in TASKS:
            raise KeyError(f'unknown task {t["name"]!r}; registered: {sorted(TASKS)}')
    started = datetime.now().astimezone()
    t_start = time.perf_counter()
    log(f'eval {run_dir.name} | device {device} {config.dtype} | seed {config.seed} | '
        f'{len(config.subjects)} subjects x {len(config.tasks)} tasks | code {commit8}'
        f'{" (dirty)" if code["dirty"] else ""}')

    results: dict = {t['name']: {'args': t.get('args') or {}, 'subjects': {}, 'pairs': {}}
                     for t in config.tasks}
    subjects_meta = []
    labels: list[str] = []
    for spec in config.subjects:
        t0 = time.perf_counter()
        subject = load_subject(spec, device, dtype)
        if subject.label in labels:
            raise ValueError(f'duplicate subject label {subject.label!r}')
        labels.append(subject.label)
        subjects_meta.append(subject.identity())
        log(f'subject {subject.label}: {subject.path.name} sha {subject.sha256[:8]} | '
            f'{subject.meta["model_name"]} {subject.param_count / 1e6:.1f}M | '
            f'step {subject.meta.get("step")} tokens {subject.meta.get("tokens_seen")} | '
            f'loaded in {time.perf_counter() - t0:.1f}s')
        for t in config.tasks:
            name, args = t['name'], t.get('args') or {}
            log(f'task {name} on {subject.label}')
            t0 = time.perf_counter()
            ctx = EvalCtx(subject, name, config.seed, device, dtype, log)
            res: TaskResult = TASKS[name](ctx, **args)
            elapsed = time.perf_counter() - t0
            if res.samples:
                sdir = run_dir / 'samples' / name
                sdir.mkdir(parents=True, exist_ok=True)
                with open(sdir / f'{subject.label}.jsonl', 'w', encoding='utf-8') as f:
                    for row in res.samples:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
            results[name]['subjects'][subject.label] = {
                'scalars': res.scalars,
                'items': {k: [round(float(x), 6) for x in v] for k, v in res.items.items()},
                'witness': res.witness,
                'elapsed_s': round(elapsed, 1),
            }
            log(f'task {name} on {subject.label} done in {elapsed:.1f}s')
            _write_json(run_dir / 'results.json', results)   # partial record survives a crash
        del subject
        _free(device)

    ref = labels[0]
    for name, block in results.items():
        ref_items = block['subjects'][ref]['items']
        for label in labels[1:]:
            other = block['subjects'][label]['items']
            block['pairs'][label] = {
                key: paired(ref_items[key], other[key], n_boot=config.n_boot, seed=config.seed)
                for key in ref_items if key in other and len(other[key]) == len(ref_items[key])
            }
            shown = [(k, p) for k, p in block['pairs'][label].items()
                     if not k.startswith(('gold_', 'copy_'))][:6]
            log(f'{name}: {label} - {ref}: ' + ' '.join(
                f'{k}={p["mean_diff"]:+.4f}[{p["ci95"][0]:+.4f},{p["ci95"][1]:+.4f}]' for k, p in shown))

    finished = datetime.now().astimezone()
    meta = {
        'created': started.isoformat(timespec='seconds'),
        'finished': finished.isoformat(timespec='seconds'),
        'elapsed_s': round(time.perf_counter() - t_start, 1),
        'code': code,
        'torch': torch.__version__,
        'device': str(device),
        'dtype': config.dtype,
        'seed': config.seed,
        'reference': ref,
        'subjects': subjects_meta,
        'tasks': config.tasks,
    }
    _write_json(run_dir / 'meta.json', meta)
    _write_json(run_dir / 'results.json', results)
    if config.registry_dir:
        rdir = Path(config.registry_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        _write_json(rdir / f'{run_dir.name}.json',
                    {'config': asdict(config), 'meta': meta, 'results': results})
        log(f'record copied to {rdir / (run_dir.name + ".json")}')
    log(f'done in {meta["elapsed_s"]}s: {run_dir}')
    log_file.close()
    return run_dir


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description='vagus evaluation')
    p.add_argument('recipe')
    p.add_argument('overrides', nargs='*', help='key=value (yaml-parsed), e.g. device=cpu')
    a = p.parse_args(argv)
    evaluate(EvalConfig.from_yaml(a.recipe, a.overrides))


if __name__ == '__main__':
    sys.exit(main())
