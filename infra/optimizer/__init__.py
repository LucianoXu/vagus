# Optimizers. The factory maps a model's param_groups() three-way split
# (muon / adamw_decay / adamw_no_decay) onto a concrete optimizer. Each
# impl registers one builder here; custom optimizers (muon, ...) get
# their own module in this package plus a registry line.
#
# Contract for every builder: the returned object exposes step /
# zero_grad / state_dict / load_state_dict / param_groups, and each param
# group carries a 'name' (metrics log per-group norms under it) and an
# 'lr'. build_optimizer stamps 'base_lr' onto every group afterwards —
# the schedule multiplies base_lr into lr each step, so a composite
# (e.g. muon + adamw) must surface the groups of all its members.

import torch

from . import muon


def build_adamw(model, args: dict) -> torch.optim.Optimizer:
    '''Plain AdamW baseline: muon + adamw_decay merged into one decay
    group, everything of dim < 2 without decay.'''
    groups = model.param_groups()
    args = dict(args)
    wd = args.pop('weight_decay', 0.0)
    if 'betas' in args:
        args['betas'] = tuple(args['betas'])
    param_groups = [
        {'params': groups['muon'] + groups['adamw_decay'],
         'weight_decay': wd, 'name': 'decay'},
        {'params': groups['adamw_no_decay'],
         'weight_decay': 0.0, 'name': 'no_decay'},
    ]
    return torch.optim.AdamW(param_groups, **args,
                             fused=torch.cuda.is_available())


def build_muon(model, args: dict) -> 'muon.MuonAdamW':
    '''Muon on the block matrices, AdamW on embeddings/head/vectors.
    args: lr / momentum / weight_decay (/ nesterov / ns_steps / lr_adjust)
    for Muon, plus an 'adamw' sub-dict passed to torch.optim.AdamW.'''
    groups = model.param_groups()
    args = dict(args)
    adamw_args = dict(args.pop('adamw'))
    if 'betas' in adamw_args:
        adamw_args['betas'] = tuple(adamw_args['betas'])
    wd = adamw_args.pop('weight_decay', 0.0)
    muon_opt = muon.Muon(
        [{'params': groups['muon'], 'name': 'muon'}], **args)
    adamw_opt = torch.optim.AdamW(
        [{'params': groups['adamw_decay'],
          'weight_decay': wd, 'name': 'decay'},
         {'params': groups['adamw_no_decay'],
          'weight_decay': 0.0, 'name': 'no_decay'}],
        **adamw_args, fused=torch.cuda.is_available())
    return muon.MuonAdamW(muon_opt, adamw_opt)


OPTIMIZERS = {'adamw': build_adamw, 'muon': build_muon}


def _check_coverage(model, opt) -> None:
    '''Every trainable param must land in exactly one group. A param a
    model's param_groups() forgets has no optimizer and silently never
    updates; this turns that into a construction-time error.'''
    counts: dict[int, int] = {}
    for g in opt.param_groups:
        for p in g['params']:
            counts[id(p)] = counts.get(id(p), 0) + 1
    names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    missing = [n for i, n in names.items() if i not in counts]
    dup = [n for i, n in names.items() if counts.get(i, 0) > 1]
    foreign = len([i for i in counts if i not in names])
    if missing or dup or foreign:
        raise ValueError(
            'optimizer param groups do not cover the model: '
            f'missing {missing}, in multiple groups {dup}, '
            f'{foreign} params not in the model')


def build_optimizer(name: str, model, args: dict) -> torch.optim.Optimizer:
    opt = OPTIMIZERS[name](model, args)
    _check_coverage(model, opt)
    for g in opt.param_groups:
        g['base_lr'] = g['lr']   # schedule multiplies this every step
    return opt
