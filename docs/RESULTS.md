# Measured results (A100 80GB PCIe, SGLang 0.5.23, DFlash b16 drafter)

Every number here comes from `results/*.json` on the box, produced by
`scripts/run_grid.py` against `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16`.
Nothing in this file is projected. The register that decides whether a claim
counts is `docs/EXPERIMENTS.md`; the verdict is `scripts/decide.py`.

## Configuration

| field | value |
|---|---|
| GPU | 1x A100 80GB PCIe, driver 580.178.04, CUDA 13.0 |
| target / drafter | `Qwen/Qwen3-8B` / `z-lab/Qwen3-8B-DFlash-b16` |
| backend | SGLang 0.5.23, in-process `Engine`, `speculative_algorithm=DFLASH` |
| decoding | greedy (temperature 0), `max_new_tokens=256` |
| benchmark | frozen MATH-500 prompts, first 32 (replication at 64 for c=32) |
| arms | A0 = 17 verify columns, A1 = 9 columns, A3 = block 8 / 9 columns, A5 = plain AR |
| concurrency | batch chunk size c in {1, 4, 8, 16, 32} (sglang clamps `max_running_requests` to 48 for spec decoding; our cells fit inside it) |

## The main grid: math500, 32 prompts, c = 1 to 32

| c | AR tok/s | A0 tok/s | A1 tok/s | A0/AR | A1/AR | A1/A0 | ms/round A0 | ms/round A1 | A1/A0 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 88.9 | 301.4 | 259.1 | **3.39x** | 2.91x | 0.86 | 15.41 | 15.05 | 0.98 |
| 4 | 348.5 | 787.9 | 772.7 | 2.26x | 2.22x | 0.98 | 16.71 | 16.14 | 0.97 |
| 8 | 684.3 | 1390.9 | 1327.8 | 2.03x | 1.94x | 0.95 | 19.41 | 16.94 | 0.87 |
| 16 | 1324.5 | 1779.5 | 2363.9 | 1.34x | 1.79x | **1.33** | 29.36 | 19.58 | 0.67 |
| 32 | 2451.1 | 2245.1 | 3363.7 | **0.92x** | 1.37x | **1.50** | 50.74 | 30.31 | 0.60 |

Replication at c=32 with 64 prompts (`results_p64/`):

| arm | tau | tok/s | vs AR | tokens/s vs A0 |
|---|---|---|---|---|
| A5 (AR) | 1.00 | 2418.9 | 1.00x | - |
| A0 (17 cols) | 5.63 | 2334.5 | **0.97x** | 1.00x |
| A1 (9 cols) | 4.86 | 3183.2 | 1.32x | **1.36x** |
| A3 (block 8) | 4.85 | 3065.9 | 1.27x | 1.31x |

## Findings

**F1. The speedup collapses with concurrency, and at c=32 it is gone.** A0 goes from
3.39x the autoregressive baseline at c=1 to 0.92x at c=32 (32 prompts) and 0.97x at
64 prompts. At serving batch, DFlash on this box is at par with plain AR decoding,
not 2.8x faster as the paper reports for B200. The loss is 73% of the c=1 advantage.

**F2. The mechanism is the verify pass, not the drafter.** A0's round time grows from
15.41 ms (c=1) to 50.74 ms (c=32), a factor of 3.3. A1's grows from 15.05 to 30.31 ms,
a factor of 2.0. The 8 extra verify columns are what inflate the round once the batch
is full: the drafter's wide pass is cheap, verifying its output is not.

**F3. A 9-column verify block overtakes 17 columns above c=8.** A1/A0 is 0.86 at c=1,
0.95 at c=8, 1.33 at c=16 and 1.50 at c=32; the 64-prompt replication gives 1.36x at
c=32. The crossover sits between c=8 and c=16 in both runs.

**F4. The cost is real and it is paid at low concurrency.** At c=1 A1 is 14% slower
than A0 (259.1 vs 301.4 tok/s) and its tau drops from 5.64 to 4.91 (-13%). Shortening
the verify block trades acceptance for round time, and it is a bad trade below c=8.

**F5. On this backend the truncation arm and the block-width arm are the same
configuration.** At 32 prompts, A1 (block 9, verify 9) is tau 4.90 / 3363.7 tok/s and A3
(block 8, verify 8) is tau 4.86 / 3370.4: **0.2% apart in throughput**. At 64 prompts
the same pair reads 4.86 / 3183.2 against 4.85 / 3065.9 (3.8% apart). `decide.py`
scores H4 (A1/A3 >= 1.10) at **0.998**, i.e. no difference. SGLang's **Source confirmation (installed 0.5.23,
`srt/arg_groups/speculative_hook.py::_handle_dflash`):** `speculative_dflash_block_size` is
documented as "Alias of --speculative-num-draft-tokens for DFLASH", the hook writes one into
the other with `speculative_num_draft_tokens=int(cfg.speculative_dflash_block_size)`, and
setting both to different values is a hard `ValueError`. The draft worker reads that single
value as its block size, so "draft wide, verify narrow" is not expressible through the
public API. The two-knob separation our
proposal is built on is **not testable on sglang 0.5.23** without patching the draft
worker to keep its 16-token block and hand only the first 9 columns to verification.

**F6. Exact match needs batch-invariant kernels, and with them DFlash is lossless.**
Two identical autoregressive runs with the default kernels agree on only 27/64 = 0.422
of generations, so the default-kernel agreement numbers (0.34 to 0.42) measure backend
nondeterminism, not quality. With `enable_deterministic_inference`: AR-vs-AR is 990.2 vs
989.9 tok/s and every arm matches AR on **all 64 prompts (1.000)**. DFlash's losslessness
holds; it was never in question, only invisible without the flag.

| c=32, 64 prompts, deterministic kernels | tau | tok/s | vs AR | match |
|---|---|---|---|---|
| A5 (AR) run A | 1.00 | 990.2 | 1.00x | 1.000 |
| A5 (AR) run B | 1.00 | 989.9 | 1.00x | 1.000 |
| A0 (17 columns) | 5.66 | 1495.8 | 1.51x | **1.000** |
| A1 (9 columns) | 4.88 | 1842.2 | 1.86x | **1.000** |

**F7. Our tau is not the paper's tau.** We measure 5.5 to 5.7 at 17 columns; the paper
reports 8.01 on MATH-500 with block 16 on a B200. Prompt subset, output length
(256 vs the paper's setting) and GPU all differ, so this is a configuration gap, not a
contradiction, and no cross-paper comparison of absolute tau should be made from it.

## Deterministic-kernel sweep (quality-valid, batch-invariant kernels)

| c | AR tok/s | A0 tok/s | A1 tok/s | A0/AR | A1/AR | A1/A0 |
|---|---|---|---|---|---|---|
| 1 | 29.3 | 119.7 | 95.5 | 4.09x | 3.26x | 0.80 |
| 4 | 115.5 | 426.8 | 334.1 | 3.70x | 2.89x | 0.78 |
| 8 | 228.7 | 899.2 | 715.5 | 3.93x | 3.13x | 0.80 |
| 16 | 477.0 | 1438.3 | 1534.6 | 3.02x | 3.22x | 1.07 |
| 32 | 993.6 | 1674.3 | 2206.7 | 1.68x | 2.22x | **1.32** |

Every cell matches AR on 32/32 prompts. tau is constant across concurrency in this mode
(A0 5.86, A1 5.00), which is the batch-invariance showing up as exactly reproducible
acceptance. A3 at c=32 in this mode: tau 5.00 / 2228.3 tok/s, i.e. 0.99% from A1.

## F10. Our engine reproduces the paper's own width ablation within 7%

DFlash's Table 8 (p.8) runs train-block x test-block at single stream on an H200, greedy. Two of
its four cells are directly comparable to arms we ran, because our backend could only execute
block == submitted:

| cell | paper tau (H200, batch 1) | our tau (A100, deterministic, c=1) | delta |
|---|---|---|---|
| b16 drafter, tested at b16 (= A0) | 6.33 | 5.86 | -7.4% |
| b16 drafter, tested at b8 (<= A3) | 5.09 | 5.00 | -1.8% |

Different GPU class, different concurrency, batch-invariant kernels on our side, so the agreement
is not a bit-level reproduction and must not be quoted as one. What it does establish is that the
engine-side setup sits on the paper's own operating point rather than on a misconfigured one: the
7% and 2% gaps are the size of a hardware/kernel-mode difference, not of a wrong arm. It also means
the block-width effect the paper reports (6.33 -> 5.09, a 20% drop) is visible in our setup
(5.86 -> 5.00, a 15% drop), which is the effect the verify-window cell has to beat.

Caveat that survives: neither cell is a verify-window test. Both ran block == submitted.

## F11. Kernel mode moves absolute throughput 2.5x; the paired ratio barely moves

Same box, same prompts, math500, 32 prompts, c=32, two kernel modes:

| arm | det tok/s | det x AR | default tok/s (seed 1) | default x AR | ratio stability |
|---|---|---|---|---|---|
| A5 (AR) | 993.6 | 1.00 | 2460.8 | 1.00 | abs 2.48x, by design |
| A0 (draft 16 / 17 col) | 1674.3 | 1.685 | 2234.5 | 0.908 | sign flips |
| A1 (draft 8 / 9 col) | 2206.7 | 2.221 | 3149.0 | 1.284 | 1.318 det vs 1.409 default |

Two things this pins down, and one it forbids:

1. **Repeatability.** A5 at c=32 measured in two adjacent passes is 2460.8 vs 2461.1 tok/s, a
   **0.013%** gap. Adjacent-run noise on an identical configuration is negligible next to the
   1.10x decision bar, so a 1.3x paired ratio is not a noise artefact.
2. **Ratios are the reportable quantity.** Absolute tok/s depends on kernel mode by 2.5x, and
   A0's position against AR even changes side of parity (1.685x det, 0.908x default). Quoting an
   absolute number without the mode is meaningless; the paired ratio moves far less (1.318 vs
   1.409, 7% apart) because both arms pay the same kernel tax.
3. **Forbidden:** comparing a deterministic-mode arm against a default-mode arm. Every arm in one
   table must come from one mode, and the mode must be in the caption. Two numerics regimes also
   give different tau for the same config (A0 5.861 det vs 5.510 default; A1 5.001 vs 4.850),
   because batch-invariant kernels change argmax ties - so tau is not portable across modes either.

Consequence for the VLC cell: report `tau_9(16)/tau_9(8)` and the predicted tok/s ratio from the
in-process cost model as the primary quantities, and use engine tok/s only as corroboration of the
model on the expressible arm.

## F12. The paired ratio replicates to four significant figures while absolutes move 10%

math500, 32 prompts, c=32, default kernels, two seeds, plus the deterministic pass for reference:

| pass | A5 (AR) tok/s | A0 tok/s | A0 xAR | A1 tok/s | A1 xAR | **A1/A0** | A0 tau | A1 tau |
|---|---|---|---|---|---|---|---|---|
| det, s0 | 993.6 | 1674.3 | 1.685 | 2206.7 | 2.221 | **1.318** | 5.861 | 5.001 |
| default, s1 | 2460.8 | 2234.5 | 0.908 | 3149.0 | 1.284 | **1.409** | 5.510 | 4.850 |
| default, s2 | 2461.1 | 2461.8 | 1.002 | 3467.9 | 1.409 | **1.409** | 5.730 | 4.872 |

- **The ratio is the measurement, and its spread is wider than the pair that reads best.**
  A1/A0 at c=32 math500, default kernels, is 1.498 (s0), 1.409 (s1), 1.409 (s2), and 1.364 at 64
  prompts (results_p64, s7); deterministic kernels give 1.318. So the estimable quantity is a
  **range of 1.32 to 1.50**, not 1.409 twice: quoting the two seeds that agree to four figures and
  not s0 would be exactly the selective reading this critique exists to catch. The sign and the
  approximate size survive every seed; the fourth digit does not.

Consequence: the VLC cell's headline is a paired ratio with a CI, and its corroboration on the
expressible arm is the sequence 1.318 / 1.409 / 1.409, not a single absolute number.

## F13. The width optimum moves with concurrency, and at c=32 the published default is below AR

math500, 32 prompts, 256 new tokens, default kernels, one box, one pass per cell:

| c | A5 (AR) tok/s | A0 (b16) tok/s | A0 xAR | A1=A3 (b8) tok/s | b8 xAR | **A0 / b8** |
|---|---|---|---|---|---|---|
| 1 | 88.9 | 301.4 | 3.39 | 259.1 | 2.91 | 1.163 |
| 4 | 348.5 | 787.9 | 2.26 | 772.7 | 2.22 | 1.020 |
| 8 | 684.3 | 1390.9 | 2.03 | 1327.8 | 1.94 | 1.048 |
| 16 | 1324.5 | 1779.5 | 1.34 | 2363.9 | 1.78 | 0.753 |
| 32 | 2451.1 | 2245.1 | 0.92 | 3363.7 | 1.37 | 0.667 |

- **The two width arms cross over between c=8 and c=16.** Wide drafting is worth 1.163x over
  narrow at c=1 and 0.667x at c=32. This is the paper's §5.5.4 statement measured as a curve with
  the crossover located, rather than asserted as a serving caveat.
- **At c=32 the published block-16 configuration is 0.92x autoregressive decoding** on this box,
  while the narrow block reaches 1.37x. A reader told only the headline (2.8x at c=32) would
  conclude the opposite of what this box does.
- **The c=32 gap decomposes exactly, with no cost model:** tau ratio
  5.516/4.901 = 1.125, tok/s ratio 2245.1/3363.7 = 0.667. Round time is tau / tok_per_s, so the
  implied round-cost ratio cost(b16)/cost(b8) = tau_ratio / tok_ratio = 1.125/0.667 = **1.686**.
  Verifying 17 columns instead of 9 costs 1.69x per round on an A100 at c=32, which is the entire
  reason the wide block loses there.
- **That 1.69 is the quantity VLC attacks.** Keeping the wide block's proposals (higher tau) while
  verifying only 9 columns removes the cost the wide block pays but not the acceptance it earns.
  The bar is stated in advance: tau_9(16) >= 1.05 x tau_9(8) and tok/s >= 1.10 x the narrow arm.

## F14 (provisional, n=8). The first validated acceptance measurement lands inside the collapse band

math500, 8 prompts, raw prompt convention, 256 new tokens, greedy, one pass per block through the
loop that reproduces the checkpoint's own `spec_generate` token for token (reference tau 3.73 on
fresh prompts, gate passed at 0.6 x the engine's 5.86):

| quantity | value |
|---|---|
| tau_16(16) | 4.452 |
| tau_9(16) - the treatment cell | **3.948** |
| tau_9(9) - the truncation control | **3.901** |
| ratio tau_9(16) / tau_9(9) | **1.012** |

The pre-committed collapse rule (`docs/NOVELTY.md` §3) fires on ratio in [0.95, 1.05] and reads:
the run counts as a replication plus a negative result, and the contribution loses the "better
knob" title. 1.012 is inside that band.

Read as a decomposition, not as a verdict: capping the wide block at 9 columns costs
4.452 -> 3.948, **11% of its committed tokens per round**, and drafting narrow instead of wide
costs a further 3.948 -> 3.901, **1.2%**. So the wide block's advantage is not in its first nine
columns; it is in the columns the window throws away. That is the mechanism the proposal was
betting on, and at n=8 it is not there.

Caveats that keep this provisional: n=8 prompts, ~520 rounds, no CI yet, and the harness sits 24%
below the engine's absolute tau (4.45 vs 5.86 at block 16) because it decodes from its own greedy
contexts rather than the engine's scheduler state. The ratio is unaffected by that offset (both
arms ride the same contexts), but the absolute level is not comparable and must not be quoted as a
reproduction of the engine. The 32-prompt run on both benchmarks is the version that decides.

## F15. The decisive cell: truncating the wide block at 9 columns buys nothing (NOT SUPPORTED)

math500, 32 frozen prompts, raw prompt convention, 256 new tokens, greedy, one validated loop, one
pass per drafter block, **7079 rounds**. Acceptance cells are `tau_k(b) = mean over rounds of
min(committed, k)`, so every cell in a column is the same quantity at the same verify length
(`experiments/standalone/accept_math500.json`, key `tau`):

| drafter block b \ verify columns k | 5 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|
| 8  | 3.278 | 4.192 | n/a | n/a | n/a |
| 9  | 3.277 | 4.165 | **4.381** | n/a | n/a |
| 12 | 3.259 | 4.123 | 4.328 | 4.810 | n/a |
| 16 | 3.326 | 4.198 | **4.409** | 4.898 | 5.278 |

**The mechanism, stated as the table states it.** Up to 8 columns the drafter's block width changes
acceptance by less than 2% (3.26-3.33 at k=5, 4.12-4.20 at k=8): the first eight proposals from a
16-slot block are as good as those from an 8-slot block, and no better. The width effect appears
only above 8 columns, where the narrow block cannot follow: 5.278 against 4.192 (**+25.9%**) at
k=16. So the wide block's entire acceptance advantage lives in the columns that a 9-column window
throws away, and capping at 9 gives up 5.278 -> 4.409, **16.5%**.

**The contrast, pre-registered before the run.**

| quantity | value | source |
|---|---|---|
| tau_9(16) | 4.409 | `tau["b16_k9"]` |
| tau_9(9) | 4.381 | `tau["b9_k9"]` |
| ratio of means | 1.006 | the table above |
| **paired ratio on (prompt, ctx_len)** | **1.018, 95% CI [0.989, 1.049]** | `vlc.py` [2], 10000 resamples, seed 0 |
| per-prompt wide>narrow / tie / wide<narrow | 20 / 0 / 12 | `vlc.py` [2] |
| sign test on the 32 discordant prompts | p = 0.215 | binomial, two-sided |
| cost factor round(9,9)/round(16,9) | 0.995 | medians 45.07 / 45.29 ms, control over treatment |
| predicted tok/s ratio | 1.013 | tau x cost = 1.018 x 0.995 |
| bars | tau >= 1.05, tok/s >= 1.10 | `docs/EXPERIMENTS.md` H4-VLC |

**Which arm is the control, stated once.** A round commits up to its verify-column count, so a
harness `columns` value equals the engine's `verify_columns` and equals draft width + 1. The
registered control D8 is the engine arm A3: `draft_width: 8, verify_columns: 9`
(`results/A3_math500_c32_s0.json`), which is `--narrow 9` here. A 7-draft / 8-column arm exists in
the sweep and is *not* the registered control; measured against it the tau ratio is 1.073
[1.040, 1.104], which clears the tau bar, and the predicted tok/s ratio is 1.073 x round(8)/round(9)
= 1.076, which does not. Reported so the choice is visible: the verdict is stable under both
readings, and it rests on the tok/s bar under the 8-column one.

Both bars fail. The 95% CI excludes a true ratio of 1.05, and the direction is not significant on
its own (p = 0.215). Under the pre-committed collapse rule this is the negative result the rule was
written for:

> On a bidirectional-block drafter, the block-size remedy DFlash nominates in 5.5.4 and the
> verify-window truncation proposed here buy the same thing. Not a separate knob.

**Why the cost side is not the consolation prize.** At batch 1 the verify pass is memory-bound:
the round costs 44.94, 45.08, 45.18 and 45.29 ms for blocks 8, 9, 12 and 16, a 0.8% spread across
an 8-to-16 column range, and the draft pass is 6.3 ms at both 8 and 16 slots. Columns are free
here, so this instrument cannot price them. Pricing them from the engine instead (c=32, 9 vs 17
columns: 30.31 vs 50.74 ms, results/A1_math500_c32_s0.json, results/A0_math500_c32_s0.json, ratio
1.674) would hand the proposal a 1.67x round advantage and it would still need tau >= 0.66 to clear
the tok/s bar at 1.10, which the measured 1.018 does not reach on its own terms. The verdict does
not depend on which cost model is used: at a cost factor of 1.674 the predicted tok/s ratio is
1.018 x 1.674 = 1.70, but that is a claim about column pricing, not about the knob, and the tau bar
fails independently.

**Threat specific to this table.** The loop reproduces the checkpoint's own `spec_generate` token
for token, and it still under-accepts against the engine: tau_16(16) = 5.278 here against 5.861
from the engine on the same benchmark (math500 3.73 vs 5.86 and mtbench 2.29 vs 4.08 at the gate).
The engine's integration differs from the checkpoint's reference loop in how target hidden states
are carried across rounds, and this harness inherits the reference loop's convention. Both arms sit
inside one instrument on byte-identical contexts, so the ratio travels; the absolute level does not,
and no sentence here compares this tau to a paper tau.

## F16. What the knob is worth against the arm everyone runs: 1.34x over block 16, parity with block 9 (COMPOSED)

COMPOSED, NOT END-TO-END: acceptance comes from the batch-1 harness and round cost from the
batch-32 engine records. One end-to-end run of draft 16 with a 9-column verify window would settle
it, and this backend cannot launch it, because one flag sets both the drafter block and the verify
width (`Correction 2026-11-19`). `scripts/vlc_tokps.py` prints every factor with its file:key:

| target | acceptance factor | round-cost factor | composed | 95% CI |
|---|---|---|---|---|
| A0 (block 16 / 17 columns, as shipped) | 0.804 paired, 0.835 pooled | 1.674 | **1.345 paired, 1.398 pooled** | [1.291, 1.406] |
| A1 (= D8: 8 drafts / 9 columns) | 1.018 | 0.995 | **1.013** | [0.989, 1.049] |

Sources: `accept_math500.json` `tau` for the acceptance cells and `rows` for the paired reading;
`results/A0_math500_c32_s0.json:ms_per_round_p50` = 50.736 ms and
`results/A1_math500_c32_s0.json:ms_per_round_p50` = 30.309 ms for the cost factor. Paired means
per-prompt tau ratios, CI from 10000 bootstrap resamples of prompts.

**The claim this licenses.** Truncating the shipped block-16 drafter at nine verify columns keeps
80% of its committed tokens per round and cuts the round to 60% of its cost, which composes to
**1.34x** over the block-16 arm at c=32 (paired; 1.40x on the pooled cells) and is bit-identically
decodable, since only the verify window changes. **The claim it does not license:** that this beats
changing the block size. It does not: block 8 with 9 columns measures 1.318x (deterministic), 1.498x
(s0), 1.409x (s1), 1.409x (s2) and 1.364x (64 prompts) over the same arm, and VLC sits at parity
with it (1.013, CI [0.989, 1.049]).

**How much the composition can be trusted, measured rather than assumed.** The same model run on
the engine's own two arms reproduces the measured A1/A0 throughput to 0.7% with default kernels
(0.889 tau x 1.674 cost = 1.487 against 1.498 measured) and over-predicts by 14% with deterministic
kernels (0.853 x 1.765 = 1.506 against 1.318 measured): with CUDA graphs the round split stops
being a valid cost model. So the 1.345 point estimate carries an unknown systematic of that size,
and the defensible interval for VLC over A0 is roughly **1.18x to 1.41x**, not 1.345x to four
figures. Both ends clear the 1.10 bar; neither approaches the 1.67x that the cost ratio alone would
suggest, because the acceptance loss is real and lands on the same side.

**Third instrument, same answer.** The engine, with both terms measured in one run and no
composition at all, gives A1/A0 = tau 0.889 x round 1.674 = 1.487 predicted against 1.498 measured
(default kernels). The knob the paper nominates and the knob proposed here are the same knob, and
the engine says so without any harness in the loop.

## Verdict from the pre-registered rules

`uv run python scripts/decide.py --results results_det --benchmark math500 --adaptive-na`
(exit 0):

```
verdict: MECHANISM ONLY

  H1  PASS    ms/round(A1)/ms/round(A0) = 0.566
  H2  FAIL    tau(A1) = 5.00 (min 7.2)
  H3  PASS    tok/s ratio = 1.318
  H4  FAIL    tok/s(A1)/tok/s(A3) = 0.990
  H5  FAIL    tok/s ratio at c=1 = 0.798
  H6  PASS    min greedy agreement = 1.0000
  H7  NA      A4 not implementable on this backend (no per-request verify length)

model vs measurement:
  rho:      predicted 0.50   measured 0.08  (column-scalable share of the round 43.4%)
  w_tail:   predicted 0.10   measured 0.38  (tau at 9 columns 5.00)
```

**F9. The verdict is MECHANISM ONLY, and that is the result to report.** The mechanism
is confirmed: the round time does scale with verify columns (H1, 0.566), throughput does
improve at serving batch (H3, 1.318), and the improvement is lossless (H6, 1.000). What
fails is the framing: tau at 9 columns is 5.00, well under the 7.2 the register demanded
(H2), so the cost is larger than projected; the 9-column arm is 20% slower at c=1 (H5);
and the block-width control is indistinguishable from the truncation arm (H4, 0.990),
because the backend makes them the same configuration (F5). The thresholds were not
moved after the data arrived.

What survives for the report: "at serving batch, DFlash's verify block is the binding
cost, and halving it recovers 1.32x over DFlash and 2.22x over autoregressive decoding
at a measured tau of 5.00 with bit-identical outputs". What does not survive:
"truncation dominates block-width control", which needs a patched draft worker to test.
**Label correction (2026-11-19).** Earlier revisions of this section called A1 "draft 16, verify 9".
It was not: `speculative_num_draft_tokens=9` sets the drafter's block to 9 as well
(`dflash_worker_v2.py:511-523` calls `set_block_size(9)`), and the same applies to A3 at 8. Every
A1/A3 number in this file is block-k/verify-k, k equal. The "draft wide, verify narrow" cell could
not be run on the engine at all; it was built as the standalone instrument and has now been measured
(F15).

**H4-VLC, run 2026-11-18 on the standalone instrument: FAIL.**
`tau_9(16)/tau_9(9) = 1.018, 95% CI [0.989, 1.049]` against a 1.05 bar, predicted tok/s ratio 1.023
against 1.10, per-prompt 20 wins to 12 losses (sign test p = 0.215), 32 prompts, 7079 rounds,
math500. The verdict is unchanged from MECHANISM ONLY, and it now has the one cell that would have
promoted it, measured, and the cell does not promote it. The contribution is the critique plus a
negative result, not a better knob (F15).


## Correction (2026-11-19): the arms were not the configurations their labels claim

Found while wiring the standalone instrument, then confirmed in the source.

`SGLangBackend` built the engine with exactly one speculative flag,
`speculative_num_draft_tokens = verify_columns - 1` (`src/serve.py`). `Arm.draft_width` never
reached the backend. Because that single flag sets the drafter's block size *and* the number of
verified draft tokens, **every arm ran with a block equal to its submitted length**:

| arm | declared | what actually launched | tau (c=32, det) |
|---|---|---|---|
| A0 | draft 16 / 17 columns | draft 16 / 17 columns | 5.86 |
| A1 | draft 16 / 9 columns | **draft 8 / 9 columns** | 5.001 |
| A3 | draft 8 / 9 columns | draft 8 / 9 columns | 5.001 |

A1 and A3 are the same configuration. Their 0.990 (default kernels) and 0.99% (deterministic)
agreement is **not** a durability result; it is one configuration measured twice.

### What this retracts, and what it buys

- **Retracted:** reading A1-vs-A3 as "truncation reproduces the wide-draft gain". No arm ever
  drafted wider than it verified. The report may not attribute anything to a verify window.
- **Retracted:** reading A1-vs-A0 as a truncation effect. That comparison is the paper's own T8
  diagonal, draft 16 (17 columns) versus draft 8 (9 columns), measured at c=32 on an A100:
  **1.32x** deterministic. It reproduces the paper's §5.5.4 recommendation at serving batch, and it
  is a *width* result, not a truncation result.
- **Gained:** a measured noise floor. Two runs of one configuration sit **1.0%** apart, so the
  pre-registered 1.10x bar is roughly ten times run-to-run noise. Any VLC effect under 1% is
  indistinguishable from a repeat measurement, which is now a number rather than an assumption.

### Fixed at the root

`SGLangBackend.__init__` now refuses an arm whose `draft_width` differs from `submitted`, naming
the coupling and pointing at the standalone instrument, and every record carries
`executed_draft_width` / `executed_verify_columns` beside the declared ones. A mislabel of this
kind can no longer be produced silently.

### Consequence for the claim

The treatment cell, draft 16 and verify 9 columns, has never been run and **cannot** be run on this
backend. `experiments/standalone/run_accept.py` measures it, gated on reproducing the checkpoint's
own `spec_generate` token for token. Until that gate passes, no acceptance number from it counts.

## What this does and does not license

- It licences: "at batch 32 on an A100, DFlash's advantage disappears, and halving the
  verify block recovers 1.36x over DFlash and 1.32x over AR, at a 13% tau cost".
- It does not licence: any claim that truncation beats block-width control, because
  F5 shows the two are not separable in this implementation.
- It licenses the quality claim, under one condition: in deterministic-kernel mode all
  three arms agree with AR on 64/64 prompts, so the improvement is lossless there. The
  default-kernel agreement numbers are reported as backend noise, not as a quality delta.
- It does not licence any speed number without its kernel mode named (F8).

**F8. Kernel mode changes the size of the win, not its sign.** Deterministic inference
costs throughput on both sides (AR 2419 -> 990 tok/s, A0 2234 -> 1496) and it shrinks
A1's advantage over A0 from +36% to **+23%** (1842.2 vs 1495.8). Every speed number must
therefore name its kernel mode; quality numbers may only be quoted in deterministic mode.
