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

- **The ratio is the measurement.** A1/A0 is 1.409 at both default-kernel seeds, identical to
  four significant figures, while the absolute tokens/s for both arms move ~10% between those
  same seeds (A0 2234.5 -> 2461.8, A1 3149.0 -> 3467.9). Whatever moves the absolute number
  moves both arms together and cancels in the ratio. Absolute throughput is a property of the
  pass; the ratio is a property of the arms.
- **AR is not a stable denominator across seeds either** (2460.8 vs 2461.1, but A0 lands on both
  sides of parity with it: 0.908x at s1, 1.002x at s2). A claim phrased as "beats autoregressive
  decoding" is therefore a claim with a 10% denominator wobble; a claim phrased as a paired ratio
  between two speculative arms is not.
- **tau is not portable across kernel modes and not perfectly stable across seeds** (A0 5.861 det,
  5.510 s1, 5.730 s2). Report tau per pass, never averaged across modes.
- **Deterministic mode costs 2.5x absolute throughput and 7% of the ratio** (1.318 vs 1.409). Any
  bit-exactness claim and any throughput claim must come from the same pass.

Consequence: the VLC cell's headline is a paired ratio with a CI, and its corroboration on the
expressible arm is the sequence 1.318 / 1.409 / 1.409, not a single absolute number.

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
A1/A3 number in this file is block-k/verify-k, k equal. The "draft wide, verify narrow" cell has
never been run; it is registered as H4-VLC in `docs/EXPERIMENTS.md` and needs the standalone
instrument, because no public flag can express it.


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
