# evict-only — management-aware continued training, eviction alone

Branch `evict-only` (from `unified-tier1`, 2026-09-02). The simplest
member of the triage family, run as its own training arm.

## Why (the tier-1 post-mortem)

Every uct kill in `unified-tier1.md` (v1 chronic divergence, v2.1 gate
fired at 807 / 851k, v2.2 "analytic-lever residue") traced to the
demote exit: the P=1 pool enters the readout as a **signed,
un-normalized denominator term** `Z = Z_a + e^{-M+b} Z_p` with
`Z_p = t0 + q·t1` linear in q; between the guardrail (`Z < 1e-4 Z_a`)
and `Z_a` the ratio readout amplifies up to 1e4 and its gradient goes
as 1/Z² — a rare-event amplifier consistent with the spiky,
seed-dependent gnorm portrait. Hard selection on the *eviction* exit
never misbehaved, but **no eviction-only training run existed**: all
evict-only numbers (m512e, the byte-fair D(R) leg) are frozen or
plain-CT evaluations. This arm supplies the missing cell.

Contrast with NSA/DSA (DeepSeek): both train with hard top-k selection
and no gradient through the choice; their scores only gate *access*
and never enter the readout arithmetic. The eviction-only block has
the same property — every atom in the readout passes through the
current query's softmax — so the gradient reaching any surviving
(k, v) is an ordinary attention probability.

## Design

- `infra/components/evict.py` — state (k, v, alive, pos, ring) over
  fixed-capacity slots; readout = masked SDPA (fused, bf16 under
  autocast like the stateless path); score = transported error-law
  Monte-Carlo over the ring (W=32), fp32, `no_grad`; hard top-k to
  budget; optional Gumbel perturbation (`gumbel_tau`, training only).
  Score forms: `lin` (E1 default, exact error norm with r = p/(1-p)),
  `sq`, `p2` (unified.py v1 form, regression anchor). The compiled
  cell takes RoPE tables + query positions as tensors: one graph per
  (do_manage, r) pair instead of one per block position.
- `infra/train/managed.py` — `manage: evict`; knobs `score`,
  `gumbel_tau`, `lookahead`, `use_checkpoint`, `compile_cell`.
- `infra/eval/evict_ppl.py` — cells `full`, `m<b>e_<score>`.
- `infra/eval/bench_evict.py` — Mtok/s + peak memory at the real
  geometry, with the stateless compiled forward as reference.
- Recipes `recipe/train/evict_ct2B_m128_s{43,44}.yaml`: SAX2 15B
  endpoint → 2B tokens, budget **128** (tier-1 used 512, where the
  frozen evict-only gap is only +0.013 nat; at 128 it is +0.043 at
  L2048, byte-fair grid), block 256, score `lin`, compiled cell, no
  activation checkpointing, 8×8 geometry, Muon 1.75e-4, WSD warmup
  0.04, `gnorm_gate 1.1` at step 100. Controls: the existing
  `plain-ct2B-s{43,44}` checkpoints, evaluated under the same cells.

## Gates

- Gate 0/1 + port anchor + gradient sanity: `tests/test_evict.py`
  (9 tests) — PASSED locally 2026-09-02 (torch 2.13 cpu, 17s):
  stream(manage=False) ≡ forward (<2e-4), loss ≡ stateless (<1e-4),
  `score=p2` ≡ `unified.stream_hidden(demote=False)` (<1e-3),
  checkpointed grads ≡ plain grads (<1e-4), managed/unmanaged gnorm
  ratio within [0.2, 5], compiled cell ≡ eager (<5e-4), Gumbel inert
  under `no_grad` and live in training, manage_every=2 counts.
- Gate 2 (scale): gnorm at step 100 ≤ 1.1 (10× plain's 0.11) — the
  prediction of this arm is that it passes by a wide margin (no pool,
  no signed denominator). Automatic kill otherwise.

## Predictions (registered before launch)

1. Gate 2 passes: step-100 gnorm within a small factor of plain (0.11),
   not 1e3–1e16. Falsifies "hard selection itself was the amplifier".
2. Throughput ≥ 5× the tier-1 sprint's best (0.012 Mtok/s/GPU): no
   measure_stats (41% of CUDA), no fp32 materialized readout, fused
   SDPA. Target ≥ 0.06 Mtok/s/GPU (2B in ≤ 2.5h on 4 GPUs).
3. Closed-loop signal: the managed−full gap at m128 (frozen +0.043 at
   L2048) shrinks in the trained arm and does **not** shrink in the
   plain-CT control (v2 finding: plain CT leaves the gap invariant).
   If it does not shrink, the model cannot (in 2B tokens) learn to
   write eviction-survivable kv — informative either way.

## Runs

| job | what | outcome |
|---|---|---|
| 29876412 | bench-evict: 6-cell throughput/memory grid (1×A100) | COMPLETED (grid below); its pytest step was skipped (no pytest in raven's venv then) |
| 29876584 / 29876672 | tests/test_evict.py on a GPU node | 8/9 then **9/9 PASSED** (the compiled-cell check moved to CUDA: raven nodes have no CPU-inductor toolchain; the login node kills the suite silently) |
| 29876414 | eval-evict-fp: frozen SAX2 + plain-ct s43/s44, budgets {32,64,128,256,512} × scores {lin, p2}, L∈{2048,4096} → runs/eval/evict_frozen_plain.json | running |
| 29876528 | evict-ct2B-m128-s43 (4×A100, 24h wall, resumes on resubmit) | RUNNING from 20:26 on ravg1158 |
| 29876531 | evict-ct2B-m128-s44 | RUNNING from 20:26 on ravg1184 |

### Bench (job 29876412, 340M, one A100-40GB, per-GPU Mtok/s, 5 steps)

| config | Mtok/s/GPU | peak GiB |
|---|---|---|
| plain stateless, compiled, micro 8 (reference) | 0.0603 | 17.0 |
| **evict m128 block256, compile, no ckpt, micro 8 (recipe)** | **0.0346** | **26.3** |
| evict m128 block512, compile, no ckpt, micro 8 | 0.0384 | 26.6 |
| evict m512 block256, compile, no ckpt, micro 8 | 0.0301 | 33.0 |
| evict m128 block256, compile + activation ckpt, micro 8 | 0.0208 | 6.7 |
| evict m128 block256, eager, no ckpt, micro 8 | 0.0165 | 32.6 |
| evict m128 block256, compile, no ckpt, micro 16 | OOM | > 39.5 |

Reading: 2.9× the tier-1 sprint's best (0.0119) and 6× its eager
baseline; 57% of the plain forward. Block 512 buys only 11%, so the
launch storm is gone and the remaining gap is real compute (masked
SDPA over cap slots + scoring). compile × non-reentrant checkpoint
works in this cell (the tier-1 metadata clash does not recur) but
costs 40% throughput; not used. Prediction 2's 0.06 target was not
met; 2B tokens ≈ 4.0 h on 4 GPUs.

### Evict-only floors (job 29876414, complete → runs/eval/evict_frozen_plain.json; nats, gap vs own full in parentheses)

| ckpt | L | score | full | m32 | m64 | m128 | m256 | m512 |
|---|---|---|---|---|---|---|---|---|
| frozen | 2048 | lin | 2.4244 | 2.5027 (+.078) | 2.4858 (+.061) | **2.4694 (+.045)** | 2.4525 (+.028) | 2.4378 (+.013) |
| frozen | 2048 | p2 | 2.4244 | 2.5010 (+.077) | 2.4835 (+.059) | 2.4669 (+.043) | 2.4508 (+.026) | 2.4370 (+.013) |
| frozen | 4096 | lin | 2.4591 | 2.5060 (+.047) | 2.4883 (+.029) | 2.4721 (+.013) | 2.4546 (−.005) | 2.4408 (−.018) |
| frozen | 4096 | p2 | 2.4591 | 2.5050 (+.046) | 2.4861 (+.027) | 2.4692 (+.010) | 2.4527 (−.006) | 2.4398 (−.019) |
| plain-s43 | 2048 | lin | 2.4000 | 2.4781 (+.078) | 2.4609 (+.061) | **2.4445 (+.045)** | 2.4275 (+.028) | 2.4131 (+.013) |
| plain-s43 | 2048 | p2 | 2.4000 | 2.4762 (+.076) | 2.4582 (+.058) | 2.4421 (+.042) | 2.4258 (+.026) | 2.4122 (+.012) |
| plain-s43 | 4096 | lin | 2.4403 | 2.4814 (+.041) | 2.4633 (+.023) | 2.4470 (+.007) | 2.4295 (−.011) | 2.4158 (−.025) |
| plain-s43 | 4096 | p2 | 2.4403 | 2.4802 (+.040) | 2.4608 (+.021) | 2.4440 (+.004) | 2.4276 (−.013) | 2.4147 (−.026) |
| plain-s44 | 2048 | lin | 2.3997 | 2.4779 (+.078) | 2.4609 (+.061) | **2.4443 (+.045)** | 2.4272 (+.028) | 2.4127 (+.013) |
| plain-s44 | 2048 | p2 | 2.3997 | 2.4760 (+.076) | 2.4581 (+.058) | 2.4418 (+.042) | 2.4256 (+.026) | 2.4118 (+.012) |
| plain-s44 | 4096 | lin | 2.4401 | 2.4815 (+.041) | 2.4632 (+.023) | 2.4467 (+.007) | 2.4292 (−.011) | 2.4155 (−.025) |
| plain-s44 | 4096 | p2 | 2.4401 | 2.4803 (+.040) | 2.4605 (+.020) | 2.4437 (+.004) | 2.4274 (−.013) | 2.4144 (−.026) |

Readings:

1. **Port anchor at scale**: every frozen `p2` cell equals the tier-1
   byte-fair / m<b>e number to the last digit; `full` = 2.4244 /
   2.4591 (gate 0 at scale).
2. **Plain CT is gap-invariant at every budget and length** (frozen vs
   plain gaps agree to ≤ 0.002 at L2048; at L4096 plain CT even widens
   the *negative* gaps slightly). Both seeds agree to ≤ 0.0003. The
   closed-loop test is therefore sharp: the evict arm's own
   m128e_lin − full at L2048 must fall below **+0.044** to count.
3. **Score form**: `p2` (squared mass × squared displacement) beats
   `lin` by 0.001–0.003 nat at every cell on SAX2-340M — the opposite
   sign from E1 on Llama/Qwen, at a magnitude below what matters
   here. The training arm keeps the pre-registered `lin`.
4. **Beyond the trained context evict-only beats full** from m256 up
   (L4096: −0.005 frozen, −0.011 plain-CT at m256; −0.018 / −0.025 at
   m512) — the tier-1 side fact, now on the compacted SDPA path.
