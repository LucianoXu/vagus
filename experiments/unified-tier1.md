# unified-tier1 — frozen floor vs management-aware continued training

Tier 1 of the 2026-08-31 plan (engram PLANNING/EXPERIMENTS context):
quantify on the SAX2 15B endpoint (a) the budget-PPL floor of the
frozen model under unified-block management, and (b) whether ~2B tokens
of management-aware continued training lift that floor beyond what
plain continued training explains.

Theory under test: engram `theory/KVCacheAsBoundedAssociativeMemory.md`
§4 (eviction identity), §5′ (type-change identity, P=1 scores,
centering), §7′(c′) (single typed atom list + P=1 moment pool, mass
column, denominator guardrail).

## Design

Four configurations, budgets m ∈ {256, 512, 1024}, eval lengths
{2048, 4096 (stress row — beyond trained context)}:

1. frozen SAX2, full cache — ceiling
2. frozen SAX2 + unified management — frozen floor
   (+ eviction-only ablation at m=512: is the pool paying?)
3. SAX2 + 2B management-aware CT (budget 512 during training),
   seeds 43/44 → evaluated as in (2)
4. SAX2 + 2B plain CT, same data/geometry/lr/seeds → evaluated as in
   (2). Separates "2B more tokens" from "management-aware 2B".

Eval: `infra/eval/budget_ppl.py` — held-out = last manifest shard
(excluded from both CT arms via `holdout_last_shard`; the original 15B
run saw ~13.5% of its windows once — a constant across all compared
checkpoints, so paired deltas are unaffected). NLL over positions ≥
256, fp32, per-sequence means kept for pairing.

## Implementation (branch unified-tier1)

- `infra/components/unified.py` — atom table (c, k, v) + P=1 moment
  pool per kv-head; readout combines explicit-denominator softmax over
  atoms with the centered pooled P=1 term; signed-pool denominator
  guardrail (fallback to atoms-only readout, occurrences counted).
  Management between 256-token blocks: transported ring ensemble
  (W=32, RoPE phase addition), per-atom exit = argmin of Monte-Carlo
  Lemma-3 (evict) vs Lemma-4 P=1 (demote) scores, top-k down to
  budget. Differentiable BPTT across blocks with per-(layer, block)
  activation checkpointing.
- `infra/train/managed.py` — continued-training entry point; both arms
  (manage: unified | none) share loop, data order, loss path.
  init_ckpt loads finished-run weights at fresh start.
- `infra/eval/budget_ppl.py`, recipes `recipe/train/{unified_ct2B_m512,
  plain_ct2B}_{s43,s44}.yaml`, sbatch `recipe/slurm/raven_managed.sbatch`,
  `raven_eval_ppl.sbatch`, tests `tests/test_unified.py`.

## Deviations from the sketched design (recorded per directive)

- **Hard selection, no soft γ / soft c.** Tier 1 trains the *model*
  under a fixed policy (closed-loop adaptation); gradients flow through
  every surviving contribution (atoms, masses, pool moments — demoted
  atoms keep gradients via the pool), but not through the discrete
  choice. Selection gradients belong to policy learning (E4 proper) and
  carry the §7′(h)(ii) risk this tier deliberately excludes.
- **Both exit scores use the squared (ΔD-consistent) forms**, evaluated
  by exact Monte-Carlo over the transported ring rather than the
  Gaussian main term: cross-family comparability by construction
  (2026-09-01 discussion: the two exits must price in the same
  currency). E1's finding that the *linear* form ranks eviction better
  is a within-family statement; revisit if exit choices look skewed.
- **Evicted/demoted atoms stay in the buffer (masked dead)** — the
  budget governs readout sparsity and the honest atom count, not
  physical memory (engram's masking-harness equivalence). Physical
  compaction is an engineering optimization out of Tier-1 scope.
- **Batch geometry 8×8 instead of SAX2's 16×4** (same global batch) in
  *both* CT arms, so the streaming state fits 40GB and the window→step
  mapping stays identical across arms.
- **Pool state fp32 with a=clamp(·,25) centering**; sink-scale logits
  cancel by centering (§5′-2); clamp activations are counted in health.
- **RoPE phase drift on pooled keys is unmitigated** (the per-band gate
  is axis-D queue item 7, out of scope); the pool therefore underprices
  decoherence — if demotion hurts at 4096 this is the first suspect.

## Gates

- Gate 0 (streaming ≡ softmax when not binding): `tests/test_unified.py`
  — max hidden diff < 2e-4 fp32 on a 3-layer tiny model, both
  manage=False and non-binding budget. PASSED locally 2026-09-01
  (6/6 tests, torch 2.13 cpu).
- Gate 1 (manage=none loss ≡ stateless loss): exact same-forward by
  construction in managed.py; cross-path equality asserted in tests.
  At scale: the first ~400 steps of plain-ct2B-s43 must continue from
  loss ≈ 2.42 without a spike. → see Runs.
- Gate 2 (managed pilot): first ~1B of unified-ct2B-m512-s43 — loss
  finite, no divergence, health counters sane (z-fallback ≪ 1,
  demote/evict both active). Full matrix only after this.

## Runs

Constraint discovered at submit time (2026-09-01 01:07): **Raven full
maintenance 2026-09-01 08:00 → 2026-09-07 08:00 (all nodes)**. All four
CT arms were therefore submitted with 6h walltime to fit the remaining
window; they checkpoint on SIGUSR1 900s before the limit and resume by
resubmitting the same command after maintenance. Plain arms should
finish 2B tonight (~2.5h at SAX2 throughput); unified arms cover as
much as eager streaming throughput allows and resume after 09-07.

| job | config | notes |
|---|---|---|
| 29835289 | eval-frozen: SAX2 model-final, ceiling + floors + m512-evict-only, L∈{2048,4096} | out: runs/eval/frozen_sax2.json |
| 29835293 | plain-ct2B-s43 (arm 4, gate-1-at-scale) | 6h wall |
| 29835294 | plain-ct2B-s44 (arm 4) | 6h wall |
| 29835295 | unified-ct2B-m512-s43 (arm 3, gate-2 pilot) | 6h wall |
| 29835296 | unified-ct2B-m512-s44 (arm 3) | 6h wall |

Deployment note: GitHub was unreachable from the workstation at
submit time; the branch reached raven as a git bundle over SSH
(`git bundle create ... c3436e3..unified-tier1` → fetch on raven). An
earlier bundle accidentally shipped a pre-amend commit carrying 2.6GB
of stignore-runs checkpoints; raven's checkout and object store were
reset and pruned (.git back to 1.6M), and `/stignore-*` is now in
.gitignore.

## After the maintenance window (2026-09-07 08:00+)

Resume any CT arm that did not reach step 3,814 (identical command
resumes from its recent checkpoint; drop -t once the queue is normal):

    cd ~/work/vagus
    sbatch -J plain-s43 recipe/slurm/raven_managed.sbatch recipe/train/plain_ct2B_s43.yaml
    sbatch -J plain-s44 recipe/slurm/raven_managed.sbatch recipe/train/plain_ct2B_s44.yaml
    sbatch -J uct-s43  recipe/slurm/raven_managed.sbatch recipe/train/unified_ct2B_m512_s43.yaml
    sbatch -J uct-s44  recipe/slurm/raven_managed.sbatch recipe/train/unified_ct2B_m512_s44.yaml

Then the trained-arm evaluation (fills rows 3–4 of the matrix):

    sbatch -J eval-ct recipe/slurm/raven_eval_ppl.sbatch \
      --ckpt runs/unified-ct2B-m512-s43-02cad06b/ckpt-00003814.pt \
      --ckpt runs/unified-ct2B-m512-s44-02cad06b/ckpt-00003814.pt \
      --ckpt runs/plain-ct2B-s43-02cad06b/ckpt-00003814.pt \
      --ckpt runs/plain-ct2B-s44-02cad06b/ckpt-00003814.pt \
      --data_dir data/tokenized/fineweb-edu-100BT-mistral32k \
      --out runs/eval/ct_arms.json

Progress check: `squeue -u yinxu` and
`tail runs/slurm-uct-s43-*.out` from `~/work/vagus` (login via `mpcdf`).

## Results

(budget-PPL table lands here)

- Frozen ceiling (2026-09-01, job 29835289): held-out L2048 full-cache
  nll 2.4244 / ppl 11.30 — matches the SAX2 training endpoint (2.4235),
  confirming the streaming full-cache path at scale (gate 0).
  Holdout shard: 013_00009.npy (274.7M tokens).
