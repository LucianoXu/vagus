# Evaluation: an eval run is built like a train run — a yaml recipe (+
# `key=value` overrides), a run directory holding the resolved config, a
# witnessed meta.json and the results, refused silent overwrites — with
# "optimizer + data stream" replaced by "tasks x subjects".
#
#   Subject   one checkpoint resolved into (generator, meta, sha256) —
#             core.Subject; tasks see nothing below the Generator.
#   Task      fn(ctx, **args) -> TaskResult, registered by name in
#             tasks.TASKS; a new metric is one function + one line there.
#   EvalCtx   what a task may touch: the subject, device/dtype, RNGs
#             derived from (seed, task, salt) — never from the subject,
#             so every subject sees the same items and the runner can pair
#             them — and the token stores.
#   Record    runs/eval/<eval_name>-<commit8>/{config.yaml, meta.json,
#             results.json, samples/}, plus a copy of the whole record
#             under registry/eval/ (small, versioned).
#
# Every task runs on the Decodable protocol through the Generator
# (score_ids for teacher-forced logits, generate_ids for sampling), never
# on model.forward(): any Decodable — our models, an HF adapter — gets
# the same evaluation.
#
# Evaluation data is sampled by seed from the training store itself (no
# reserved held-out shards): in the sub-epoch regime single-exposure
# memorisation is negligible at these scales and paired comparisons
# share whatever bias remains; the holdout task records the expected
# exposure so a reader can judge. Point `shards:` elsewhere when a
# subject trained for several epochs or on other data.

from .core import EvalCtx, Subject, TaskResult
from .main import EvalConfig, evaluate

__all__ = ['EvalCtx', 'Subject', 'TaskResult', 'EvalConfig', 'evaluate']
