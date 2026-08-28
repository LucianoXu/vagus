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


OPTIMIZERS = {'adamw': build_adamw}


def build_optimizer(name: str, model, args: dict) -> torch.optim.Optimizer:
    opt = OPTIMIZERS[name](model, args)
    for g in opt.param_groups:
        g['base_lr'] = g['lr']   # schedule multiplies this every step
    return opt
