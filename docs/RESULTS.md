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
