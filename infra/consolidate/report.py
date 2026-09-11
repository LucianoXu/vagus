# One table over consolidation records:
#   python -m infra.consolidate.report registry/eval/consolidate-gap256-*.json
# Per record (one subject, one sleep length): the sleep cell, ratio (the
# fraction of the context benefit lost; 0 perfect), kept by position
# bucket, the forgetting witness (retain LM loss delta and, when the
# retain-KL term ran, its KL delta), the fraction of items that kept
# anything, and the replay-KL fit. `--pair A B` adds the paired per-item
# nll2 difference between two records (same seed => same items).

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    rec = json.loads(Path(path).read_text())
    res = rec['results']['consolidation']
    out = []
    for label, r in res['subjects'].items():
        w = r['witness']
        for L in w.get('sleep_x_len', []):
            out.append((Path(path).name, label, int(L), r))
    return out


def row(name, label, L, r):
    sc, it, w = r['scalars'], r['items'], r['witness']
    sl = w.get('sleep', {})
    cell = (f"{sl.get('method', '?'):9s} lr={sl.get('lr', 0):<7g} mix={sl.get('lm_mix', 0):<4g} "
            f"rk={sl.get('retain_kl', 0):<4g} {sl.get('lr_schedule', 'const'):6s} w{sl.get('warmup', 0):<2d} "
            f"s{sl.get('steps', 0):<3d} hops={','.join(f'{h:g}' for h in sl.get('hops', [0.0])):12s} "
            f"{sl.get('degrade', '-')[:8]:8s} {sl.get('teacher', '-'):5s} "
            f"{sl.get('mode', 'onehop'):14s} lam={sl.get('lam', 0):<4g} {sl.get('mem_param', '-'):4s} mlr={sl.get('mem_lr', 0):<6g}")
    kept = []
    for b in [k[5:] for k in it if k.startswith('nll3_b')]:
        den = sc[f'nll3_{b}'] - sc[f'nll1_{b}@{L}']
        kept.append((sc[f'nll3_{b}'] - sc[f'nll2_{b}@{L}']) / den if abs(den) > 1e-9 else float('nan'))
    pos = float((np.array(it[f'benefit@{L}']) - np.array(it[f'lost@{L}']) > 0).mean())
    ret = sc.get(f'retain_delta@{L}', float('nan'))
    rk = ''
    cons = w.get('consolidations', {}).get(str(L), [])
    if cons and 'retain_kl_after' in cons[0]:
        rk = f"{np.mean([c['retain_kl_after'] - c['retain_kl_before'] for c in cons]):+.4f}"
    kl = f"{sc[f'kl_before@{L}']:.2f}->{sc[f'kl_after@{L}']:.2f}" if f'kl_before@{L}' in sc else ''
    hops = ''
    if cons and 'trajectory' in cons[0]:
        t = cons[0]['trajectory']
        pen = np.mean([[r['penalty'] for r in c['trajectory']] for c in cons], axis=0)
        dr = np.mean([[r['drift'] for r in c['trajectory']] for c in cons], axis=0)
        hops = ' pen ' + '>'.join(f'{v:.2f}' for v in pen) + ' drift ' + '>'.join(f'{v:.3f}' for v in dr)
    elif cons and len(cons[0].get('hops', [])) > 1:
        hops = ' hops ' + ' '.join(f"{h['level']:g}:{np.mean([c['hops'][i]['kl_before'] for c in cons]):.2f}->"
                                   f"{np.mean([c['hops'][i]['kl_after'] for c in cons]):.2f}"
                                   for i, h in enumerate(cons[0]['hops']))
    return (f"{label:5s} L={L:<5d} gap={w.get('gap', 0):<4d} {cell} | ratio {sc[f'ratio@{L}']:6.3f} kept "
            + '/'.join(f'{k:.2f}' for k in kept) + f" retainΔ {ret:+.4f} {rk:>8s} >0 {pos:.2f} KL {kl:>10s}{hops}"
            + f"   [{name}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('records', nargs='+')
    ap.add_argument('--pair', nargs=2, metavar=('A', 'B'), help='paired nll2 difference A - B (record paths)')
    a = ap.parse_args(argv)
    rows = [row(*t) for p in a.records for t in load(p)]
    print('\n'.join(sorted(rows)))
    if a.pair:
        (na, la, La, ra), (nb, lb, Lb, rb) = load(a.pair[0])[0], load(a.pair[1])[0]
        d = np.array(ra['items'][f'nll2@{La}']) - np.array(rb['items'][f'nll2@{Lb}'])
        print(f'\npaired nll2 {na} - {nb}: {d.mean():+.4f} (sd {d.std():.4f}), A better on {(d < 0).sum()}/{len(d)}')


if __name__ == '__main__':
    sys.exit(main())
