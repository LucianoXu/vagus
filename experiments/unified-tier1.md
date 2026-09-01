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

| job | config | outcome |
|---|---|---|
| 29835289 | eval-frozen: SAX2 model-final, ceiling + floors + m512e, L∈{2048,4096} | COMPLETED 7m26s → runs/eval/frozen_sax2.json (copy: stignore-runs/eval/) |
| 29835293 | plain-ct2B-s43 (arm 4) | COMPLETED 2h27m, 2.000B, final loss 2.3765 (ema 2.3999), ckpt-00003814.pt |
| 29835294 | plain-ct2B-s44 (arm 4) | COMPLETED 2h27m, 2.000B, final loss 2.4310 (ema 2.3962), ckpt-00003814.pt |
| 29835295 | unified-ct2B-m512-s43 (arm 3 pilot) | FAILED at maintenance wall (5h44); **gate 2 not passed — chronic divergence** (loss 2.60@100 → 3.27@500, gnorm →2.6e16); weights polluted, archived, never resume |
| 29835296 | unified-ct2B-m512-s44 (arm 3 pilot) | same (FAILED 5h44) |
| 29840076 | eval-plainct: both arm-4 ckpts, same 10 cells | COMPLETED → runs/eval/plain_ct.json (copy: stignore-runs/eval/) |
| 29851927+28 | v2 evals: frozen + both plain ckpts, 11 cells each | COMPLETED → runs/eval/{frozen_sax2,plain_ct}_v2.json (mirrored) |
| 29852376 | evict-only m256e/m1024e × 2 lengths × 3 ckpts | COMPLETED → runs/eval/evict_only_extra.json (mirrored) |
| 29852392+93 | uct v2 s43/s44 relaunch (gnorm_gate 1.1, commit dae0f07) | queued |

Deployment note: GitHub was unreachable from the workstation at
submit time; the branch reached raven as a git bundle over SSH
(`git bundle create ... c3436e3..unified-tier1` → fetch on raven). An
earlier bundle accidentally shipped a pre-amend commit carrying 2.6GB
of stignore-runs checkpoints; raven's checkout and object store were
reset and pruned (.git back to 1.6M), and `/stignore-*` is now in
.gitignore.

## After the maintenance window (2026-09-07 08:00+)

Plain arms and their eval finished before the wall — nothing to
resume there. The unified arms are pilot-only (gradient pathology, see
Gate 2): **do not resubmit their recipes as-is.** The v2 sequence is:

1. Fix the demote pricing (long-horizon transported ensemble) + bound
   the pooled weights; bf16 matmuls / micro-batch 16 for throughput
   (target ≥ 0.05 Mtok/s). Re-run gates 0–2.
2. Relaunch BOTH unified seeds fresh on v2 (delete or archive the
   pilot run dirs first — same run_name resumes otherwise):
   `sbatch -J uct-s43 recipe/slurm/raven_managed.sbatch recipe/train/unified_ct2B_m512_s43.yaml` (and s44)
3. Eval the v2 arms:
   `sbatch -J eval-uct recipe/slurm/raven_eval_ppl.sbatch --ckpt runs/unified-ct2B-m512-s43-<commit>/ckpt-00003814.pt --ckpt runs/unified-ct2B-m512-s44-<commit>/ckpt-00003814.pt --data_dir data/tokenized/fineweb-edu-100BT-mistral32k --out runs/eval/unified_ct.json`

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

### Arm 4 — plain 2B CT, both seeds (job 29840076, complete)

Same 10 cells; seed-averaged NLL (s43/s44 agree to ≤0.01 everywhere
except the noisy m256 cells):

| cell | full | m1024 | m512 | m256 | m512e |
|---|---|---|---|---|---|
| L2048 | 2.3999 | 2.4096 | 2.5676 | 3.6932 | 2.4120 |
| L4096 | 2.4402 | 2.4724 | 2.8820 | 4.7840 | 2.4146 |

Reading (vs frozen): plain CT lowers the ceiling and the loose floors
by ≈ its train-loss gain (−0.025 nat, uniformly), i.e. the model got
generically better — but the **management gap (managed − full) does
not close; at tight budgets it widens**: m256@L2048 gap 1.203 → 1.293
(+0.09), m512@L4096 0.412 → 0.442 (+0.03), m512e gaps unchanged
(0.013/0.021 → 0.012/0.026). Plain continued training makes the model
lean *more* on the full cache. This is the control half of the Tier-1
question in its sharpest form: if the v2 management-aware arm closes
any of the gap, the credit belongs to management awareness — the
token-matched control moves the gap the other way.

### Gate 1 at scale — PASSED

plain-ct2B-s43, step 100: loss 2.4258 (ema 2.4241), gnorm 0.11,
0.23 Mtok/s — clean continuation of the 2.4235 endpoint, no warmup
spike. Plain arms complete 2B in ~2.4h (before the maintenance wall).

### Gate 2 — FAILED: chronic divergence (corrected 2026-09-01)

The original verdict ("stable loss, formally passes") was wrong — it
read a single point (step 100). The full trajectory of uct-s43:
loss 2.5964 (step 100, gnorm 1.6e14) → **3.2713 (step 500, gnorm
2.6e16)** — a monotone climb with growing gradient norms. Mechanism:
the rms-normalized optimizer + grad_clip=1.0 turn 1e14-scale garbage
gradients into a bounded-step **noise walk** — nothing explodes, but
every step drags the weights away from the checkpoint. Gate 2 is
recorded as **not passed (chronic divergence)**; both pilot arms ended
FAILED at the maintenance wall (05:44 elapsed) and their weights are
noise-polluted at every step. gnorm at the same step on the plain arm:
0.11.
Root cause (same disease as the PPL finding, backward face): demoted
high-mean-logit atoms (sinks) enter the pool with weight
w = c·e^a ~ e^25; gradients through the pool moments back into k/v
carry the unbalanced factor e^(a − M_t) whenever the atom's absorbed
logit scale a exceeds the readout-time alive max M_t. grad_clip=1.0
keeps the run alive but the clipped update direction is dominated by
this garbage — closed-loop adaptation signal is drowned and the
weights walk away from the init (corrected verdict above). Tonight's
unified segments are **pilot-only and must not be resumed**; v2
relaunches both seeds fresh (implemented 2026-09-01 as §6(a′)/5′-7/5′-8,
commit dbcc5fe).

**uct v2 restart plan** (hold until the frozen-v2 verdict is in):

1. **gnorm hard gate**: at step 100 the v2 unified arm's gnorm must be
   within 10× of the plain arm's (0.11 → gate at ≤ 1.1); otherwise
   stop immediately — a second amplifier exists beyond the pricing fix.
2. **stop-grad on the statistics**: (m_q, Σ_q, σ_j, γ_b, Z̄) are
   environment estimates, not model outputs — all detached; gradients
   flow only through the readout path and (if any) softened decisions.
   Status in the v2 code: already structural — `_measure_stats` and
   `_manage` run under `@torch.no_grad()`, the ring is detached at
   capture, and the write uses detached (μ, σ²) with gradients only
   through (c, k, v) — keep this as an asserted invariant at restart.
3. **init_ckpt = the SAX2 origin** (ckpt-00028610), never the pilot
   weights (every pilot step is noise-polluted). Mechanically
   guaranteed: run dirs are keyed by code commit (a v2 launch creates
   `unified-ct2B-m512-s4x-<v2commit>` and starts fresh from
   init_ckpt); the pilot dirs are archived under
   `runs/archive-pilot-v1/` to make accidental resume impossible.
   Add at restart: per-step guardrail-rate logging and a gnorm
   decomposition (pool-write path vs atom path) so a recurrence is
   locatable at a glance.

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

---

## v2 results (2026-09-01, commits dbcc5fe/d6f3a1f; jobs 29851927/29851928)

Same protocol as Tier-1, v2 policy (§6(a′) pricing + 5′-7/5′-8
projected pool; eviction score untouched). NLL, seed-averaged where
two seeds exist.

### Frozen SAX2, v1 → v2

| cell | v1 | v2 | Δ |
|---|---|---|---|
| L2048/m256 | 3.6274 | 2.4902 | **−1.137** |
| L2048/m512 | 2.5863 | 2.4472 | −0.139 |
| L2048/m1024 | 2.4347 | 2.4277 | −0.007 |
| L2048/m512e | 2.4370 | 2.4370 | 0 (bit-identical) |
| L4096/m256 | 4.7117 | 2.5596 | **−2.152** |
| L4096/m512 | 2.8709 | 2.4742 | −0.397 |
| L4096/m1024 | 2.4878 | 2.4371 | −0.051 |
| L4096/m512e | 2.4398 | 2.4398 | 0 (bit-identical) |

### plain-ct2B (both seeds), v2 policy

| cell | v2 (seed mean) | gap vs own full |
|---|---|---|
| L2048: full / m256 / m512 / m1024 / m512e | 2.3999 / 2.4636 / 2.4213 / 2.4026 / 2.4120 | — / +0.064 / +0.021 / +0.003 / +0.012 |
| L4096: full / m256 / m512 | 2.4402 / 2.5399 / 2.4458 | — / +0.100 / +0.006 |
| L4096: m1024 / m512e / m512_lam256 | 2.4114 / 2.4146 / 2.4671 | −0.029 / −0.026 / +0.027 |

### 5′-8 predictions, checked

1. **Guardrail collapse — confirmed.** z-fallback m512: 0.20%/0.75%
   (v1, L2048/L4096) → 0.013%/0.056%; m256: 1.9%/4.4% → 0.06%/0.18%.
   One to two orders of magnitude, at every budget. μ-clamp: never
   fired.
2. **m512 pooled vs evict-only — partially confirmed.** The sign of
   the harm flipped from catastrophic (+0.15/+0.43 behind evict-only)
   to marginal (+0.010 L2048 / +0.034 L4096 behind; plain-ct: +0.009 /
   +0.031). Not yet ≥ evict-only at m512; the per-budget baselines
   below settle where the pool pays.

### Pool vs evict-only, per budget (job 29852376, complete)

Pool deficit = nll(pooled) − nll(evict-only), v2 policy; seed means.

| | m256 | m512 | m1024 |
|---|---|---|---|
| frozen L2048 | +0.039 (2.4902/2.4508) | +0.010 (2.4472/2.4370) | +0.001 (2.4277/2.4270) |
| frozen L4096 | +0.107 (2.5596/2.4527) | +0.034 (2.4742/2.4398) | +0.004 (2.4371/2.4328) |
| plain L2048 | +0.038 (2.4636/2.4257) | +0.009 (2.4213/2.4120) | +0.000 (2.4026/2.4023) |
| plain L4096 | +0.112 (2.5399/2.4275) | +0.031 (2.4458/2.4146) | +0.003 (2.4114/2.4082) |

**Verdict: no budget tier where the pool nets positive on this corpus
at these lengths.** The deficit is monotone in budget tightness with
no crossover: neutral at loose budgets, mildly costly at tight ones —
tight budgets force higher-value atoms into the demoted set, so the
projection residual is exercised where it hurts. Recorded per
instruction: **the pool is neutral-to-negative on this corpus/length;
its value waits on long contexts, redundant corpora, or training
adaptation** (the uct v2 arms are exactly the third test).
Policy-level echo of engram's E3-SOTA "merge value = corpus
redundancy" finding.

Side fact worth keeping: at L4096 **evict-only beats full attention
at every budget** — m256e 2.4527 (frozen) / 2.4275 (plain) vs full
2.4591 / 2.4402: a 1/16-cache pure-eviction policy above the full
cache beyond the trained context.
3. **Exit portrait.** P=0 took zero traffic everywhere; sinks stay in
   the atom table — 5′-8's primary expectation ("honest pricing keeps
   sinks as atoms"). The slope-degeneracy branch (ε=1e-3) never fired;
   ε review noted, not blocking. The dominant v2 behavior is
   **demote-refusal**: at m512/L2048 the demotion share fell from 53%
   (v1) to 6.8% (v2) — the honest price says "evict, don't pool" for
   most candidates, and the small surviving pool is near-harmless.
4. **λ sensitivity (consistent direction, default validated).**
   L4096/m512 at λ=1/256: frozen 2.4924 vs 2.4742 (λ=1/1024); plain
   s43: 2.4663 vs 2.4464. Shorter horizon under-prices decoherence.

### Finding revision (v1 → v2)

The v1 conclusion "plain CT widens the management gap at tight
budgets" was mostly a pool-pathology artifact: under the v2 policy
the gaps are essentially identical frozen vs plain-ct (L2048 m256:
+0.066 vs +0.064; m512: +0.023 vs +0.021; L4096 m256: +0.101 vs
+0.100) — plain continued training neither widens nor closes the gap.
What survives of the v1 finding: uniform ~−0.025 improvement
everywhere, no differential adaptation. This makes the closed-loop
attribution for the uct v2 arms maximally clean: the static protocol
is seed-stable and CT-invariant at every budget, so **any gap-closing
by management-aware training is the model learning to write
management-survivable kv** — the direct evidence the training axis
wants (coordinator note, 2026-09-01).

### uct v2 relaunch (green-lit after this section's commit)

Per the restart plan above: fresh runs from the SAX2 origin under
commit ≥ d6f3a1f (recipes now carry `gnorm_gate: 1.1`), 24h walltime;
at ~0.02 Mtok/s a 2B arm needs one SIGUSR1 resume — the resume is the
same sbatch command:

    sbatch -J uct-s43 recipe/slurm/raven_managed.sbatch recipe/train/unified_ct2B_m512_s43.yaml
    sbatch -J uct-s44 recipe/slurm/raven_managed.sbatch recipe/train/unified_ct2B_m512_s44.yaml

Kill condition is automatic (trainer exits 3 with "GNORM GATE FAILED"
if step-100 gnorm > 1.1); health logs now carry the pool write
magnitudes (`unified_pool_w_max`, `unified_pool_t1_max`) as the
forward signature of the v1 amplifier — the strict pool-path/atom-path
gnorm split is deferred (checkpointed recompute makes retain_grad
unreliable); the gate plus these signatures cover recurrence
localization.

---

## Track A: v3 stepped per-band gate (2026-09-01, commit 0a9804f; job 29853967)

v3 = v2 with the pool's phase-carrying moments aged **per streaming
block** instead of damped once at write: t1/T1 ×= exp(−λ_b·block_len)
each block, λ_b = λ_q·ln(1/γ_b) with λ_q = 1/1024 and γ_b the
Lorentzian coherence — cumulative decay equals v2's static damping at
the mean query horizon 1/λ_q. t0/T0 and the P=0 tier never decay
(constants carry no phase; a content-drift gate is a separate knob,
off). Write-side projection mass factor e^{σ²/2} retained. The two
modes are a config-exclusive switch (`pool_gate: static|stepped`);
the static branch is the untouched v2 code path (the pre-existing
test suite is its regression guard). Tests: decay semigroup
(half∘half = full), horizon calibration (decay(1/λ) = γ_b),
empty-pool mode equality (the eonly sanity), switch liveness — 11/11.

**Honest annotation (recorded as instructed): this is a hypothesis
test, not a guaranteed win.** Under the (a′) position measure the
coherence modulus is age-independent — static damping is already the
L² answer for that measure; age-based decay instead tests (c″)(iv)'s
measured-decoherence reading (older pooled content should address
less). Criterion: the pool-vs-evict-only deficit, especially whether
the L2048→L4096 deficit slope (+0.010→+0.034 under v2 at m512)
flattens. Either outcome is informative.

A/B: job 29853967, frozen line, all 11 cells, `--pool_gate stepped`
→ runs/eval/frozen_sax2_v3.json; the m512e column must be
bit-identical to v2 (eviction never touches the pool).

Iron-rule note: uct v2 arms (29852392/93) were still queued when v3
merged on raven; they will load commit ≥ 0a9804f at start. Their
recipes leave `pool_gate` at the default `static`, whose code path v3
does not touch — training semantics are bit-identical to dae0f07;
only the run-dir commit hash differs.

---

## Parallel tracks (2026-09-01, coordinator-approved ladder)

Iron rule in force: uct v2 arms (29852392/93) and their monitoring
untouched; everything below runs beside them.

- **Track A (this repo)**: v3 stepped gate — section above; A/B job
  29853967.
- **Track B (engram)**: E2″(d) estimator dimension — extended
  `kvbm/e2pp_capacity.py` (the pre-existing untracked file; nothing
  else in the frozen kvbm layout touched): write-only projected-P=1
  Hebbian moment readout vs ridge-at-d per q-head (`delta_price`),
  key-collision mass portrait, per-tercile concentration check
  against §7′(c″)(ii)'s prediction. Decision rule recorded in the
  file header: delta implementation only after a substantial measured
  gap, and always with mass column + denominator. llama32-1b fit-only
  job 29854049 (captures reused, sbatch --wrap — no new file in the
  frozen slurm/ dir). engram's EXPERIMENTS.md carries uncommitted
  human edits, so the design note lives in the module docstring for
  now rather than the design doc.
- **Queued (fires after the uct arms land, per coordinator/user)**:
  training-time re-profiling — recompute D_h(m) on the uct
  checkpoints (training-free), waterfill the per-head profile, add a
  "uct+alloc" row vs frozen+alloc / plain+alloc, and compare the
  pre/post spectrum shape (unimodal kept vs moved — §7′(g)'s free
  verdict).

### Track A verdict: v3 ≈ v2 — age-based decay adds nothing at these lengths

Frozen A/B (job 29853967, runs/eval/frozen_sax2_v3.json; v3−v2 in nats):

| cell | v2 | v3 | Δ |
|---|---|---|---|
| L2048 m256 / m512 / m1024 | 2.4902 / 2.4472 / 2.4277 | 2.4939 / 2.4495 / 2.4275 | +0.004 / +0.002 / −0.000 |
| L4096 m256 / m512 / m1024 | 2.5596 / 2.4742 / 2.4371 | 2.5588 / 2.4728 / 2.4364 | −0.001 / −0.001 / −0.001 |
| m512e (both lengths) | 2.4370 / 2.4398 | bit-identical | 0 |
| L4096 m512_lam256 | 2.4924 | 2.4910 | −0.001 |

Criterion readout: the pool-vs-evict-only deficit slope
L2048→L4096 at m512 is +0.0102→+0.0344 (v2) vs +0.0125→+0.0330 (v3)
— a ~0.004-nat flattening, an order of magnitude below what would
matter, with the sign trade exactly along the age-hypothesis
direction (slightly worse short, slightly better long) but at
negligible magnitude. Guardrail rates unchanged. **Verdict: on this
corpus at ≤4k, stepped age-based decay ≈ static measure-level damping
— the (c″)(iv) age-decoherence hypothesis gains no usable support,
consistent with (a′)'s coherence modulus being age-independent under
the position measure. v2 `static` stays the default.** Scope note:
the L2048→L4096 lever is short; 16k+ contexts could still separate
the modes — out of tier scope, recorded. (The stepped path and its
tests remain in-tree behind `pool_gate`.)

## Naming registry (2026-09-01, user's PLANNING.md decision)

The dynamic management algorithm is **triage**; the full
three-operation version is **full-triage**; MRC survives as the merge
sub-policy's ancestor name. In this repo: `manage: triage` is the
canonical trainer value (`unified` accepted as a read alias — the
in-flight uct recipes keep working unmodified), the module path
`infra/components/unified.py` is unchanged so live imports never
break, and run/recipe names keep their historical spellings; new
recipes use triage names. Pure rename, no logic.

### Track B verdict: delta gap is real, but it concentrates where the theory said it wouldn't

E2″(d) on llama32-1b, both corpora (engram job 29854826; JSON+plots
mirrored to engram results/llama32-1b/full/). Direction-form
measurement (Hebbian slope's direction + per-sequence scalar/intercept,
2 dof, vs ridge's full whitened map, d+1 dof — the dof asymmetry IS
the estimator dimension; two ill-conditioned variants recorded in the
JSON as negative results: the whole-cache ratio readout collapses on
its signed denominator (guard rates up to 57%, R² ~ −150 — the
positivity-guardrail failure mode writ large) and frozen-Z̄ cannot
absorb the numerator's dynamic range).

| tercile | ridge(d) | hebbian-dir | delta_price | frac>0.05 |
|---|---|---|---|---|
| retrieval | 0.21/0.22 | 0.06/0.08 | 0.10/0.11 | 0.64/0.67 |
| mid | 0.39/0.41 | 0.16/0.13 | 0.19/0.24 | 0.81/0.91 |
| linear | 0.57/0.59 | 0.27/0.27 | **0.32/0.31** | 0.85/0.95 |

(prose/code; spearman(dp, R²_lin) = +0.47/+0.54;
spearman(dp, collision mass) ≈ 0.)

Findings: (1) the delta correction's price is substantial nearly
everywhere (dp > 0.05 on 64–95% of heads); (2) **the concentration
prediction of §7′(c″)(ii) is contradicted**: the gap grows toward the
LINEAR tercile — whitening pays most exactly where the linear class
works, i.e. on the heads the pool actually serves; even there the
Hebbian direction keeps <50% of ridge's explanatory power
("纯平滑头 Hebbian 已足够" does not hold in this measurement); (3) the
collision-mass mechanism gets no support in the direction form
(≈zero correlation). Caveat recorded: this is a whole-cache
measurement; the operating-point version (pool restricted to the
triage-demoted subset) is the sharper gate for implementation.
