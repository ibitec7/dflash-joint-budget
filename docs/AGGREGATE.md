# Aggregate results: every cell we ran

A100 80GB PCIe, SGLang 0.5.23, Qwen3-8B-DFlash-b16 drafter, frozen math500 and mtbench prompts,
greedy, 256 new tokens. Machine-readable copy: `results/aggregate.csv`. Raw records: `results/`
(default kernels, math500), `results_det/` (batch-invariant kernels), `results_mt/` (mtbench),
`results_bs/` (block-8 arm across concurrency), `results_s/` (seeds 1 and 2), `results_p64/`
(64 prompts), `experiments/standalone/accept_{math500,mtbench}_sweep.json` (acceptance surface).

Two instruments, and the table says which column comes from which:

- **engine** = SGLang, per-round median and end-to-end tok/s, batch 32. Everything in Tables 1, 4.
- **harness** = `experiments/standalone/run_accept.py`, one validated reference loop, batch 1, one
  pass per block. Acceptance only, and acceptance ratios only (its absolute tau runs ~10% below the
  engine's on the same input). Tables 2, 3.

## Table 1. Engine, end-to-end, both benchmarks (measured, no composition)

math500, default kernels, seed 0. `vs AR` is tok/s divided by the plain autoregressive arm at the
same concurrency; `vs DFlash` divides by block 16 / 17 columns at the same concurrency.

| c | arm | cols | tau | ms/round | tok/s | vs AR | vs DFlash |
|---|---|---|---|---|---|---|---|
| 1 | block 16 / 17 | 17 | 5.639 | 15.41 | 301.4 | 3.39x | 1.00x |
| 1 | 9 columns | 9 | 4.908 | 15.05 | 259.1 | 2.92x | 0.86x |
| 4 | block 16 / 17 | 17 | 5.486 | 16.71 | 787.9 | 2.26x | 1.00x |
| 4 | 9 columns | 9 | 4.873 | 16.14 | 772.7 | 2.22x | 0.98x |
| 8 | block 16 / 17 | 17 | 5.641 | 19.41 | 1390.9 | 2.03x | 1.00x |
| 8 | 9 columns | 9 | 4.749 | 16.94 | 1327.8 | 1.94x | 0.95x |
| 16 | block 16 / 17 | 17 | 5.508 | 29.36 | 1779.5 | 1.34x | 1.00x |
| 16 | 9 columns | 9 | 4.846 | 19.58 | 2363.9 | 1.78x | **1.33x** |
| 32 | block 16 / 17 | 17 | 5.516 | 50.74 | 2245.1 | 0.92x | 1.00x |
| 32 | 9 columns | 9 | 4.901 | 30.31 | 3363.7 | 1.37x | **1.50x** |
| 32 | AR (no draft) | - | 1.000 | 0.41 | 2451.1 | 1.00x | 1.09x |

mtbench, same protocol:

| c | arm | cols | tau | ms/round | tok/s | vs AR | vs DFlash |
|---|---|---|---|---|---|---|---|
| 1 | block 16 / 17 | 17 | 4.043 | 15.28 | 199.9 | 2.25x | 1.00x |
| 1 | 9 columns | 9 | 3.754 | 15.01 | 183.8 | 2.07x | 0.92x |
| 4 | block 16 / 17 | 17 | 4.066 | 16.59 | 537.9 | 1.53x | 1.00x |
| 4 | 9 columns | 9 | 3.725 | 16.01 | 514.6 | 1.47x | 0.96x |
| 8 | block 16 / 17 | 17 | 4.103 | 19.06 | 927.0 | 1.35x | 1.00x |
| 8 | 9 columns | 9 | 3.724 | 16.60 | 938.5 | 1.37x | 1.01x |
| 16 | block 16 / 17 | 17 | 4.021 | 29.26 | 1269.7 | 0.96x | 1.00x |
| 16 | 9 columns | 9 | 3.766 | 19.43 | 1554.0 | 1.17x | 1.22x |
| 32 | block 16 / 17 | 17 | 4.079 | 49.37 | 1539.6 | 0.63x | 1.00x |
| 32 | 9 columns | 9 | 3.703 | 29.37 | 2247.8 | 0.92x | **1.46x** |
| 32 | AR (no draft) | - | 1.000 | 0.41 | 2452.2 | 1.00x | 1.59x |

**The crossover is between c=8 and c=16 on both benchmarks**, and it is the same crossing in both:
below it the wide window wins, above it the narrow window wins, and the narrow arm's advantage grows
with load to 1.50x (math500) and 1.46x (mtbench) at c=32. A1 in these tables is the arm that ran with
nine verify columns; because `speculative_num_draft_tokens` set the drafter block too, it drafted 8
tokens, which is why A1 and A3 agree within 0.2% (see RESULTS, Correction 2026-11-19).

## Table 2. Acceptance across nine drafter widths, and what each width composes to

Harness, 32 prompts per benchmark, one pass per block, 15594 rounds (math500) and 21735 (mtbench).
`acc` is tau at the nine-column window divided by block 16 at its own 16-column window, so `acc - 1`
is the acceptance given up by truncating. `composed` multiplies that by the engine's measured round
cost ratio 1.674 (50.74 / 30.31 ms, c=32).

| drafts | tau_9 math500 | acc | composed | tau_9 mtbench | acc | composed |
|---|---|---|---|---|---|---|
| 8 | 4.192 (8 cols) | 0.794 | 1.33x | 3.170 (8 cols) | 0.847 | 1.42x |
| 9 | 4.381 | 0.830 | 1.39x | 3.229 | 0.863 | 1.45x |
| 10 | 4.355 | 0.825 | 1.38x | 3.208 | 0.858 | 1.44x |
| 11 | 4.320 | 0.819 | 1.37x | 3.183 | 0.851 | 1.42x |
| 12 | 4.328 | 0.820 | 1.37x | 3.162 | 0.845 | 1.42x |
| 13 | 4.293 | 0.813 | 1.36x | 3.160 | 0.845 | 1.41x |
| 14 | 4.284 | 0.812 | 1.36x | 3.149 | 0.842 | 1.41x |
| 15 | 4.298 | 0.814 | 1.36x | 3.178 | 0.850 | 1.42x |
| 16 | 4.409 | 0.835 | 1.40x | 3.310 | 0.885 | 1.48x |

Every width lands between 1.33x and 1.40x (math500) and 1.41x and 1.48x (mtbench). The spread is
7 percentage points on math500 and 7 on mtbench, against a run-to-run floor of 0.2% default and 1.0%
deterministic. Block width is not a quality knob at a fixed window; it is a ceiling on the window.

## Table 3. The verify-window surface at block 16 (acceptance only)

| window k | 5 | 8 | 9 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|
| tau, math500 | 3.326 | 4.198 | 4.409 | 4.597 | 4.898 | 5.278 |
| tau, mtbench | 2.739 | 3.201 | 3.310 | 3.404 | 3.553 | 3.741 |

The engine priced only two of these windows (9 columns at 30.31-30.74 ms, 17 columns at 49.37-50.74
ms, c=32), so no composed number is given for k in 5, 8, 10, 12: an interpolated cost would be
invented. What the two measured prices say is that widening the window from 9 to 17 columns costs
1.674x for +19.7% acceptance, which is a 0.71x tok/s trade at c=32 and the wrong direction at c=1.

## Table 4. Seed and prompt-count spread at c=32, math500 (engine, measured)

| run | block 16 / 17 | 9 columns | ratio |
|---|---|---|---|
| default, seed 0 | 2245.1 | 3363.7 | 1.498x |
| default, seed 1 | 2234.5 | 3149.0 | 1.409x |
| default, seed 2 | 2461.8 | 3467.9 | 1.409x |
| default, 64 prompts, seed 7 | 2334.5 | 3183.2 | 1.364x |
| deterministic, seed 0 | 1674.3 | 2206.7 | 1.318x |

Range 1.32x to 1.50x, deterministic at the bottom. Exact token match against the reference loop is
1.000 in the deterministic runs, which is the gate every number above passes.

## What the aggregate says

1. At high concurrency the shipped block-16 arm is the slowest correct option: 0.92x plain AR on
   math500, 0.63x on mtbench. The paper's 2.8x does not survive at batch 32 on an A100.
2. Cutting the verify window to nine columns is worth 1.33-1.50x over it, measured end to end on two
   benchmarks and five concurrencies, with the crossover between c=8 and c=16.
3. Widening the drafter behind a nine-column window is worth nothing: every width from 8 to 16 lands
   in a seven-point band. That is why the truncation proposal cannot add to the flag.
4. The window is the expensive lever: 1.674x round cost for 19.7% acceptance. The width is the cheap
   one: drafting 16 instead of 8 costs 1.0% (6.33 vs 6.27 ms), but buys under 5% acceptance at a
   fixed window.
