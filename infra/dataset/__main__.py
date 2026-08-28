# Data-prep entry point. Two subcommands, split because MPCDF compute
# nodes have no internet: `download` runs on a login node (network-bound,
# negligible CPU — a sanctioned login-node use), `tokenize` runs inside a
# SLURM CPU job (recipe/slurm/raven_tokenize.sbatch) and never touches the
# network. Both are idempotent, so the standard flow is
#
#   python -m infra.dataset download --sample 100BT --count 140   # login node
#   sbatch recipe/slurm/raven_tokenize.sbatch 100BT 140           # CPU job
#
# and re-running either resumes/extends. A count prefix is a valid smaller
# corpus (see fineweb_edu.download), so the dataset can be grown in place.

import argparse
import json
from pathlib import Path

from . import fineweb_edu
from .prepare import prepare


def _local_sources(sample: str, count: int, out_dir: Path) -> list[Path]:
    '''
    Name-sorted source prefix for `prepare`; no network. A source whose
    shard is already in the manifest may have had its parquet deleted
    (they are dropped after tokenization to free /ptmp) — prepare skips
    it by name, so a placeholder path stands in for it. Only sources
    that still need tokenizing must actually be on disk.
    '''
    dest = fineweb_edu._default_dest() / 'sample' / sample
    have = {p.name: p for p in dest.glob('*.parquet')}

    manifest_path = out_dir / 'manifest.json'
    done = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        done = {s['source'] for s in manifest['shards']
                if (out_dir / s['file']).exists() and (out_dir / s['idx']).exists()}

    names = sorted(set(have) | done)
    if len(names) < count:
        raise SystemExit(
            f"{len(names)} sources known ({len(done)} tokenized, "
            f"{len(have)} parquet in {dest}), need {count} — "
            f"run `python -m infra.dataset download` on a login node first")
    # a successful `download --count N` guarantees the name-sorted prefix
    # is gapless; tokenize is meant to run only after that succeeded
    return [have.get(n, dest / n) for n in names[:count]]


def main():
    p = argparse.ArgumentParser(prog='python -m infra.dataset')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('download', 'tokenize'):
        s = sub.add_parser(name)
        s.add_argument('--sample', default='100BT', choices=fineweb_edu.SAMPLES)
        s.add_argument('--count', type=int, required=True,
                       help='number of source shards (name-sorted prefix)')
    args = p.parse_args()

    if args.cmd == 'download':
        paths = fineweb_edu.download(args.sample, args.count)
        print(f"{len(paths)} shards present under {paths[0].parent}")
        return

    tokenizer_id = 'mistral32k'
    out_dir = (Path(__file__).resolve().parents[2] / 'data' / 'tokenized'
               / f'fineweb-edu-{args.sample}-{tokenizer_id}')
    manifest = prepare(
        sources=_local_sources(args.sample, args.count, out_dir),
        provenance={'repo_id': fineweb_edu.REPO_ID,
                    'revision': fineweb_edu.REVISION,
                    'sample': args.sample},
        out_dir=out_dir,
        tokenizer_id=tokenizer_id,
    )
    print(f"manifest: {manifest}")


if __name__ == '__main__':
    main()
