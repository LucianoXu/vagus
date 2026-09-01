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

### Frozen SAX2 (job 29835289, complete — runs/eval/frozen_sax2.json)

Held-out shard 013_00009.npy (274.7M tokens); NLL in nats over
positions ≥ 256; m512e = eviction-only ablation.

| cell | full | m1024 | m512 | m256 | m512e |
|---|---|---|---|---|---|
| L2048 | 2.4244 | 2.4347 | 2.5863 | 3.6274 | **2.4370** |
| L4096 | 2.4591 | 2.4878 | 2.8709 | 4.7117 | **2.4398** |

Findings (2026-09-01):

1. **Gate 0 at scale**: full-cache streaming nll 2.4244 ≈ the SAX2
   training endpoint 2.4235 (fresh-data regime).
2. **The transported eviction score alone is a strong floor**: m512e
   sits 0.013 nat above full at L2048 — quarter-cache with near-ceiling
   quality — and at L4096 it is **below** full (2.4398 < 2.4591):
   light eviction beats full attention beyond the trained context
   (engram E6's compression-beats-full effect reproduced on SAX at
   340M; short effective context also dodges RoPE extrapolation).
3. **The P=1 pool as wired is a net harm under tight budgets**:
   evict+demote at m512 pays +0.15 nat over evict-only, the gap more
   than doubles at L4096 (+0.43), and the denominator-guardrail rate
   rises with length (0.2% → 0.7% at m512; 1.9% → 4.4% at m256).
   Interpretation: the demote score's ring ensemble is transported only
   to the block end (Δ ≤ 32), so it cannot see RoPE decoherence of the
   pooled linear term hundreds of positions later; high-mass atoms get
   demoted cheaply and their P=1 extrapolation later pollutes the
   partition function. This is the pre-registered first suspect (see
   Deviations) and is the empirical form of engram axis-D item 7
   (per-band gate / decoherence pricing). v2 direction: score demotion
   against a long-horizon transported ensemble (or analytic per-band
   decoherence), not the near ring.

### Gate 1 at scale — PASSED

plain-ct2B-s43, step 100: loss 2.4258 (ema 2.4241), gnorm 0.11,
0.23 Mtok/s — clean continuation of the 2.4235 endpoint, no warmup
spike. Plain arms complete 2B in ~2.4h (before the maintenance wall).

### Gate 2 — pilot verdict: stable loss, pathological gradients

Both unified arms at step 100 (02:07): loss s43 2.5964 (ema 2.5698),
s44 2.5466 (ema 2.5664) — finite, non-diverging, sitting ≈0.15 nat
above the plain arm at the same step (2.4258), i.e. exactly the frozen
m512 management gap before any adaptation. Formally gate 2 passes
(no divergence). BUT: **gnorm ~1.6e14 / 5.0e13** (plain arm: 0.11).
Root cause (same disease as the PPL finding, backward face): demoted
high-mean-logit atoms (sinks) enter the pool with weight
w = c·e^a ~ e^25; gradients through the pool moments back into k/v
carry the unbalanced factor e^(a − M_t) whenever the atom's absorbed
logit scale a exceeds the readout-time alive max M_t. grad_clip=1.0
keeps the run alive but the clipped update direction is dominated by
this garbage — closed-loop adaptation signal is drowned. Verdict:
tonight's unified segments are **pilot-only and will not be resumed**;
v2 relaunches both seeds fresh. The principled fix is the same one the
PPL result demands — price demotion on a long-horizon transported
ensemble, which makes sinks expensive to demote and keeps e^a inside
the pool bounded relative to the live partition function — plus a
numerical belt (bound pooled weights relative to a running Z̄, or
log-domain pool accumulators).

### Unified-arm throughput (tonight's run)

Eager streaming: ~95 s/step ≈ 0.0055 Mtok/s (4 GPUs at 51–71%,
25.8/40GB — the memory model held). Tonight's 6h window yields ~0.1B
tokens per unified arm: enough for gate 2 (loss sanity under
management) but not 2B. Post-maintenance v2 pass (correctness first, then speed) before
relaunching the arms: bf16 logit matmuls under autocast (the current
fp32 einsums bypass tensor cores; bf16 logits are the training-native
numerics — SDPA does the same), micro-batch back to 16, optionally
torch.compile of the block cell and W=16 scoring. Target ≥ 0.05
Mtok/s (2B ≈ 11h). Relaunch BOTH unified seeds fresh on v2 so the
pair shares one code state; plain arms stay valid as controls
(manage=none semantics untouched).
