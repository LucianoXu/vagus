# Task registry: a recipe's `tasks: [{name, args}]` selects by name, args
# go verbatim into the task function. A task is
#
#     def run(ctx: EvalCtx, **args) -> TaskResult
#
# and must draw its items from ctx.rng() / ctx.torch_seed() only, so the
# item order is the same for every subject (the runner pairs on it).

from . import consolidation, degeneration, holdout, recall_probe

TASKS = {
    'holdout': holdout.run,
    'degeneration': degeneration.run,
    'recall_probe': recall_probe.run,
    'consolidation': consolidation.run,
}
