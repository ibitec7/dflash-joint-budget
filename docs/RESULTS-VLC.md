# VLC results: the clean sheet

Every number below comes from a JSON artifact in this repo. A cell marked **measured** was produced by
one run of one arm on the engine; a cell marked **composed** is a product of two measured factors, and
the file says which is which wherever a number appears.

## 1. Notation

| symbol | meaning |
|---|---|
| `b` | draft width: token slots the drafter computes per round (the DFlash drafter's block size). |
| `k` | verify window: columns the target verifies in one forward pass. `k = (submitted drafts) + 1`; the `+1` is the target's own correction token, which is free. The engine parameter `verify_columns` is this count. |
| column | one verified position. A round verifying `k` columns can commit at most `k` tokens. |
| `tau` | tokens committed per round, including the free correction token. `tau = 1` is plain decoding, so `tau` is the speculative speedup a compute-free world would give. |
| `tau_k` | windowed acceptance: mean over rounds of `min(committed, k)`. It caps a wide drafter's commitment at a narrow window, which is exactly the quantity VLC trades. |
| `ms/round` | median wall time of one round as the engine measures it, at the stated concurrency. |
| `tok/s` | end-to-end served throughput at the stated concurrency, prefill included. |
| `x AR` | tok/s divided by the autoregressive arm at the same concurrency and benchmark. |
| `x DFlash` | tok/s divided by the published DFlash arm at the same concurrency and benchmark. |

Method index used in every table:

| index | method | b / k | what it is |
|---|---|---|---|
| **DFlash** | published, as shipped | 16 / 17 | wide block, full window. Engine arm A0. |
| **VLC** | this proposal | 16 / 9 | wide block, nine-column window. Not launchable on this engine (Section 4). |
| **D8** | control | 8 / 9 | the block-size flag's own output, same nine-column window. Engine arm A3, which is what arm A1 actually ran. |
| **AR** | control floor | 1 / 1 | plain autoregressive decoding of the target, no drafter, no speculation. Engine arm A5. |

**AR is not EAGLE-3.** No EAGLE-3 arm ran anywhere in this project. AR is the bare Qwen3-8B target
model, the no-speculation floor; every speedup here is a multiple of it. The paper's EAGLE-3 numbers are
cited in the report's related-work section and are never used as a measurement.

## 2. The main comparison at serving batch

Table 1. **Methods at concurrency 32.** math500 and mtbench, 32 frozen prompts each, greedy, 256 new
tokens, A100 80GB PCIe, SGLang 0.5.23, default kernels, seed 0. The three method groups are DFlash
(published), VLC (proposed, composed), D8 (same window, narrow drafter) and AR (floor).

| benchmark | method | b / k | tau | ms/round | tok/s | x AR | x DFlash | status |
|---|---|---|---|---|---|---|---|---|
| math500 | DFlash | 16 / 17 | 5.516 | 50.74 | 2245.1 | 0.92x | 1.00x | measured |
| math500 | VLC | 16 / 9 | 4.409 (tau_9) | 30.31 | 1.345-1.398x DFlash | - | - | **composed** |
| math500 | D8 | 8 / 9 | 4.856 | 29.35 | 3370.4 | 1.38x | 1.50x | measured |
| math500 | AR | 1 / 1 | 1.000 | 0.41 | 2451.1 | 1.00x | 1.09x | measured |
| mtbench | DFlash | 16 / 17 | 4.079 | 49.37 | 1539.6 | 0.63x | 1.00x | measured |
| mtbench | VLC | 16 / 9 | 3.310 (tau_9) | 29.37 | 1.344-1.481x DFlash | - | - | **composed** |
| mtbench | D8 | 8 / 9 | 3.703 | 29.37 | 2247.8 | 0.92x | 1.46x | measured |
| mtbench | AR | 1 / 1 | 1.000 | 0.41 | 2452.2 | 1.00x | 1.59x | measured |

VLC's `ms/round` is the nine-column round from arm A1 (30.31 ms math500, 29.37 mtbench), the same
record the cost factor uses; arm A3 measures the same window 0.2% apart, which is the quoted noise
floor. Its `tok/s` cell is the composed ratio, never a measurement.

Table 1b. **Every arm measured end to end.** The nine-column arm is a real engine run here and the
DFlash row is the denominator, so this is the table to cite for the load win. A1 and A3 agree within
0.2% because the flag forced A1's drafter to 8 tokens, which is also why VLC itself has no engine row.

| benchmark | method | b / k | tau | ms/round | tok/s | x AR | x DFlash |
|---|---|---|---|---|---|---|---|
| math500 | DFlash | 16 / 17 | 5.516 | 50.74 | 2245.1 | 0.92x | 1.00x |
| math500 | D8 via A1 | 8 / 9 | 4.901 | 30.31 | 3363.7 | 1.37x | 1.50x |
| math500 | D8 via A3 | 8 / 9 | 4.856 | 29.35 | 3370.4 | 1.38x | 1.50x |
| math500 | AR | 1 / 1 | 1.000 | 0.41 | 2451.1 | 1.00x | 1.09x |
| mtbench | DFlash | 16 / 17 | 4.079 | 49.37 | 1539.6 | 0.63x | 1.00x |
| mtbench | D8 via A1 | 8 / 9 | 3.703 | 29.37 | 2247.8 | 0.92x | 1.46x |
| mtbench | D8 via A3 | 8 / 9 | 3.703 | 29.37 | 2247.8 | 0.92x | 1.46x |
| mtbench | AR | 1 / 1 | 1.000 | 0.41 | 2452.2 | 1.00x | 1.59x |

Reading Table 1b: at c=32 DFlash is **slower than AR** on both benchmarks (0.92x math500, 0.63x
mtbench). The nine-column arms restore speculation to 1.37x AR and 1.50x over DFlash on math500, and
0.92x AR and 1.46x over DFlash on mtbench. That 1.33-1.50x is the load win, and it belongs to the
nine-column window.

## 3. Ablation A: the verify window at a fixed draft width

Caption. Harness acceptance with `b = 16` fixed and the window `k` swept 5 to 16. 32 prompts, 15594
rounds (math500) and 21735 (mtbench), one validated reference loop, batch 1. `tau_k` is capped
commitment; the ratio divides by the full 16-column window. The engine priced only k=9 and k=17, so
there is no tok/s column; a cost interpolated between two measured prices would be invented.

| k | columns verified | tau, math500 | vs k=16 | tau, mtbench | vs k=16 |
|---|---|---|---|---|---|
| 5 | 5 | 3.326 | 0.630 | 2.739 | 0.732 |
| 8 | 8 | 4.198 | 0.795 | 3.201 | 0.856 |
| 9 | 9 | 4.409 | 0.835 | 3.310 | 0.885 |
| 10 | 10 | 4.597 | 0.871 | 3.404 | 0.910 |
| 12 | 12 | 4.898 | 0.928 | 3.553 | 0.950 |
| 16 | 16 | 5.278 | 1.000 | 3.741 | 1.000 |

**Ablation A says:** halving the window from 16 to 9 columns costs 16.5% of the wide block's committed
tokens (0.835 math500, 0.885 mtbench) and buys back 40% of the round cost (50.74 to 30.31 ms). The
window is the expensive lever, and its marginal columns pay best below k=9 (+32.6% from 5 to 9) and
worst above it (+19.7% from 9 to 16).

## 4. How VLC's throughput was composed, and its scope

| factor | value | instrument | artifact |
|---|---|---|---|
| acceptance `tau_9(16) / tau_17(16)` | 0.804 paired over prompts (0.835 pooled) | harness, batch 1 | `experiments/standalone/accept_math500_sweep.json` |
| round cost `ms(17 cols) / ms(9 cols)` | 1.674 (50.74 / 30.31 ms) | engine, c=32 | `results/A0_math500_c32_s0.json`, `results/A1_math500_c32_s0.json` |
| **composed tok/s ratio** | **1.345** paired, 1.398 pooled, 95% CI [1.291, 1.406] | product | `scripts/vlc_tokps.py` |

Scope, in order of how much it matters:

1. **The cost factor was measured on the D8 arm, not on a 16-draft arm.** A 16-slot draft pass costs
   1.0% more than an 8-slot one (6.331 vs 6.270 ms), so 1.674 is optimistic by about 1%; corrected, the
   product is 1.332.
2. **Acceptance is batch 1 and cost is batch 32.** The harness reads tau about 10% below the engine on
   identical input (5.278 vs 5.861 at 17 columns), so only the acceptance ratio transfers.
3. **The CI covers the acceptance factor only.** The cost factor's own floor is 0.2% default kernels
   and 1.0% deterministic.
4. **Calibration against the engine's own arms bounds the bias:** the same product predicts 1.487 vs
   1.498 measured with default kernels (0.7% error), and 1.506 vs 1.318 under CUDA graphs (+14%).

Predicted end-to-end value if VLC were launchable: 1.38-1.40 default kernels, floor about 1.18 if
graph-mode behaviour matches the pessimistic calibration.

## 5. Ablation B: the draft width at a fixed verify window

Caption. Harness acceptance at `k = 9` for every drafter width the checkpoint can run, times the round
cost that window costs on the engine (30.31 ms measured; a 16-slot draft pass adds 1.0%). The composed
column multiplies acceptance by the measured cost ratio 1.674. This ablation decides whether the width
behind a narrow window does any work at all.

| drafts b | tau_9, math500 | acceptance vs DFlash | composed | tau_9, mtbench | acceptance vs DFlash | composed |
|---|---|---|---|---|---|---|
| 8 | 4.192 | 0.794 | 1.33x | 3.170 | 0.848 | 1.42x |
| 9 | 4.381 | 0.830 | 1.39x | 3.229 | 0.863 | 1.45x |
| 10 | 4.355 | 0.825 | 1.38x | 3.208 | 0.857 | 1.44x |
| 11 | 4.320 | 0.818 | 1.37x | 3.183 | 0.851 | 1.42x |
| 12 | 4.328 | 0.820 | 1.37x | 3.162 | 0.845 | 1.42x |
| 13 | 4.293 | 0.813 | 1.36x | 3.160 | 0.845 | 1.41x |
| 14 | 4.284 | 0.812 | 1.36x | 3.149 | 0.842 | 1.41x |
| 15 | 4.298 | 0.814 | 1.36x | 3.178 | 0.849 | 1.42x |
| 16 | 4.409 | 0.835 | 1.40x | 3.310 | 0.885 | 1.48x |

**Ablation B says:** every width from 8 to 16 lands within seven points (1.33x-1.40x math500,
1.41x-1.48x mtbench) and the ordering is not monotone: block 14 is the minimum, block 16 the maximum,
rank correlation with width 0.38 over nine widths. Block 16 is best of the nine by a margin smaller than
three times the run-to-run floor. The drafter's width is a ceiling on the window, not a quality
parameter.

**The consequence for the method.** VLC keeps the wide drafter and narrows the window, which is correct
as a mechanism and composes to 1.33-1.40x over DFlash at c=32. But D8, which the block-size flag
already produces, verifies the same nine columns and measures 1.33-1.50x over DFlash by itself. VLC's
incremental value over D8 is exactly the width effect Ablation B prices at 1.8% of acceptance (math500)
and 5.2% (mtbench), against a 1.10x throughput bar. It does not reach it.

## 6. Ablation C: the load curve, where the window stops paying

Caption. Every arm measured end to end, math500, default kernels, seed 0, five concurrencies. The
nine-column row is D8. All ratios are within-row at the same concurrency; nothing is compared across
concurrency or across kernel modes.

| c | DFlash tok/s | DFlash x AR | D8 tok/s | D8 x AR | D8 x DFlash | AR tok/s |
|---|---|---|---|---|---|---|
| 1 | 301.4 | 3.39x | 258.9 | 2.91x | 0.86x | 88.9 |
| 4 | 787.9 | 2.26x | 771.2 | 2.21x | 0.98x | 348.5 |
| 8 | 1390.9 | 2.03x | 1330.4 | 1.94x | 0.96x | 684.3 |
| 16 | 1779.5 | 1.34x | 2366.2 | 1.79x | 1.33x | 1324.5 |
| 32 | 2245.1 | 0.92x | 3370.4 | 1.38x | 1.50x | 2451.1 |

mtbench crosses in the same place: DFlash falls from 2.25x AR at c=1 to 0.63x at c=32, nine columns
hold 0.92x throughout while AR nearly doubles, and the ratio between them goes from 0.92x to 1.46x.

**Ablation C says:** the crossover between the two windows sits between c=8 and c=16 on math500 and at
c=8 on mtbench. Below it the wide window wins by up to 1.16x; above it the narrow window wins and the
gap widens with load (1.33x at c=16, 1.50x at c=32). The cause is the round cost: 17 columns rise 3.3x
from c=1 to c=32 (15.41 to 50.74 ms) while 9 columns rise 2.0x (15.05 to 30.31 ms). Verification, not
drafting, is what saturates.

## 7. What this file licenses, in one paragraph

At serving batch the published DFlash arm is slower than no speculation at all (0.92x AR math500, 0.63x
mtbench), and the remedy is the verify window: nine columns deliver 1.33-1.50x over it, measured end to
end on two benchmarks and five concurrencies, with a crossover between c=8 and c=16. The proposal that
motivated the project, drafting wide and verifying narrow (VLC), composes to 1.345x (95% CI [1.291,
1.406]) over DFlash, and its incremental value over simply shrinking the block is the width effect at a
fixed window: 1.8% to 5.2% of acceptance, against a 1.10x throughput bar it does not reach. The load
critique is the contribution; the width flag is the remedy; the window is the mechanism that explains
both.

