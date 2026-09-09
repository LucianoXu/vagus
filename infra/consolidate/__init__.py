# Memory consolidation ("sleep"): moving what the fast memory holds
# into the slow memory.
#
# Fast memory m is the model's streaming state — the linear-attention
# matrix state, the KV cache of a softmax layer — anything export_cache
# returns; its blank state m0 is what reset_cache allocates. Slow memory
# w is the weights. For a model that has read a context and sits at
# (w, m), consolidation looks for w* such that
#
#     p(s | w, m)  ~  p(s | w*, m0)        s ~ p(. | w, m)
#
# — the model with its memory cleared, run from a fresh stream, should
# model the continuation distribution the way the model with the
# memory did; KL is the currency. After consolidation the memory can be
# emptied and the model continues "on top of" what it read.
#
# The unit test (probe.py + eval/tasks/consolidation.py): split a
# document into X | Y and score Y three ways —
#
#     nll1  read X then Y                       (memory intact)
#     nll2  read X, consolidate, clear, read Y  (memory moved to weights)
#     nll3  read Y alone                        (no memory)
#
# nll1 < nll3 is what context buys; a consolidation is good when
# nll2 ~ nll1, and (nll2 - nll1) / (nll3 - nll1) in [0, 1] is the
# fraction of that benefit lost.
#
# sleep.py holds the procedures that produce w*: replay_kl (the
# definition above made an algorithm — sample continuations from
# (w, m), distill the memory-conditioned logits into the weights of a
# fresh-stream student, mixed with ordinary LM loss so the model does
# not collapse onto the replay) and ntp_x (the baseline that must be
# beaten: plain next-token fine-tuning on X itself, which uses no
# memory at all).

from .items import Item, item_ids, sample_items
from .probe import bucket_means, nll_positions, score_continuation
from .sleep import SleepConfig, Sleeper

__all__ = ['Item', 'item_ids', 'sample_items', 'bucket_means', 'nll_positions',
           'score_continuation', 'SleepConfig', 'Sleeper']
