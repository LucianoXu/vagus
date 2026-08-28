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
from pathlib import Path

from . import fineweb_edu
from .prepare import prepare


def _local_sources(sample: str, count: int) -> list[Path]:
    '''name-sorted prefix of already-downloaded shards; no network.'''
    dest = fineweb_edu._default_dest() / 'sample' / sample
    have = sorted(dest.glob('*.parquet'))
    if len(have) < count:
        raise SystemExit(
            f"{dest} holds {len(have)} shards, need {count} — "
            f"run `python -m infra.dataset download` on a login node first")
    return have[:count]


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
        sources=_local_sources(args.sample, args.count),
        provenance={'repo_id': fineweb_edu.REPO_ID,
                    'revision': fineweb_edu.REVISION,
                    'sample': args.sample},
        out_dir=out_dir,
        tokenizer_id=tokenizer_id,
    )
    print(f"manifest: {manifest}")


if __name__ == '__main__':
    main()
