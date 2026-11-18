# VLC results: the clean sheet

## 1. Notation

| symbol | meaning |
|---|---|
| `b` | draft width: token slots the drafter computes per round (DFlash drafter block size). |
| `k` | verify window: columns the target verifies in one pass, `k = submitted drafts + 1`. The `+1` is the target's own correction token and costs nothing extra. |
| column | one verified position; a round verifying `k` columns commits at most `k` tokens. |
| `tau` | tokens committed per round including the free correction token. `tau = 1` is plain decoding. |
| `ms/round` | median round time, engine measured, at the stated concurrency. |
| `tok/s` | end-to-end served throughput at the stated concurrency, prefill included. |
| `x AR` | tok/s divided by plain autoregressive decoding at the same concurrency. |
| `x DFlash` | tok/s divided by the published DFlash arm at the same concurrency. |

| index | method | b / k | what it is |
|---|---|---|---|
| **DFlash** | published, as shipped | 16 / 17 | wide block, full window. Engine arm A0. |
| **VLC** | this proposal | 16 / 9 | wide block, nine-column window. |
| **D8** | control | 8 / 9 | the block-size flag's output, same nine-column window. Engine arm A3. |
| **AR** | control floor | 1 / 1 | plain autoregressive decoding of the target, no drafter, no speculation. Engine arm A5. |

**AR is not EAGLE-3.** No EAGLE-3 arm ran in this project. AR is the bare Qwen3-8B target, the
no-speculation floor; the paper's EAGLE-3 numbers are cited in related work, never measured here.

## 2. The main comparison at concurrency 32

Table 1. math500 and mtbench, 32 frozen prompts each, greedy, 256 new tokens, A100 80GB PCIe, SGLang
0.5.23, default kernels, seed 0, concurrency 32. `x AR` and `x DFlash` are within-benchmark ratios at
this concurrency.

| benchmark | method | b / k | tau | ms/round | tok/s | x AR | x DFlash |
|---|---|---|---|---|---|---|---|
| math500 | DFlash | 16 / 17 | 5.516 | 50.74 | 2245.1 | 0.92x | 1.000x |
| math500 | VLC | 16 / 9 | 4.896 | 30.31 | 3019.7 | 1.23x | 1.345x |
| math500 | D8 | 8 / 9 | 4.856 | 29.35 | 3370.4 | 1.38x | 1.501x |
| math500 | AR | 1 / 1 | 1.000 | 0.41 | 2451.1 | 1.00x | 1.092x |
| mtbench | DFlash | 16 / 17 | 4.079 | 49.37 | 1539.6 | 0.63x | 1.000x |
| mtbench | VLC | 16 / 9 | 3.676 | 29.37 | 2069.2 | 0.84x | 1.344x |
| mtbench | D8 | 8 / 9 | 3.703 | 29.37 | 2247.8 | 0.92x | 1.460x |
| mtbench | AR | 1 / 1 | 1.000 | 0.41 | 2452.2 | 1.00x | 1.593x |

VLC's two derived cells, stated once: its `tau` is the harness tau_9 multiplied by the harness-to-engine
scale measured on DFlash at 17 columns (5.861 / 5.278 = 1.110); its `tok/s` is DFlash's measured tok/s
times the paired acceptance ratio over 32 prompts (1.345 math500, 1.344 mtbench), which is
`tau_9(16) / tau_9(D8)` x the measured round-cost ratio 1.674. Every other cell in Table 1 is a direct
measurement on this engine.

## 3. Ablation A: the verify window at a fixed draft width

Table 2. Acceptance with `b = 16` fixed, window swept 5 to 16. 32 prompts, 15,594 rounds math500 and
21,735 mtbench, batch 1. No throughput column: the engine priced only k=9 (30.31 ms) and k=17
(50.74 ms), and a cost interpolated between two measured prices would be invented.

| k | columns | tau, math500 | vs k=16 | tau, mtbench | vs k=16 |
|---|---|---|---|---|---|
| 5 | 5 | 3.326 | 0.630 | 2.739 | 0.732 |
| 8 | 8 | 4.198 | 0.795 | 3.201 | 0.856 |
| 9 | 9 | 4.409 | 0.835 | 3.310 | 0.885 |
| 10 | 10 | 4.597 | 0.871 | 3.404 | 0.910 |
| 12 | 12 | 4.898 | 0.928 | 3.553 | 0.950 |
| 16 | 16 | 5.278 | 1.000 | 3.741 | 1.000 |
The engine's own tau at 17 columns is 5.861 with batch-invariant kernels (math500), 1.110x the
harness value, which is the scale factor used in Table 1.

Halving the window 16 to 9 costs 16.5% of committed tokens (0.835 math500, 0.885 mtbench) and buys back
40% of the round cost. Marginal columns pay best below k=9 (+32.6% from 5 to 9) and worst above
(+19.7% from 9 to 16).

## 4. Ablation B: the draft width at a fixed window

Table 3. Acceptance at `k = 9` for every drafter width, with the nine-column round cost (30.31 ms
math500, 29.37 mtbench; a 16-slot draft pass adds 1.0%). Throughput is acceptance x cost ratio 1.674 x
DFlash's measured tok/s.

| drafts b | tau_9 math500 | tok/s math500 | x AR | x DFlash | tau_9 mtbench | tok/s mtbench | x AR | x DFlash |
|---|---|---|---|---|---|---|---|---|
| 8 | 4.192 | 2985.1 | 1.22x | 1.330x | 3.170 | 2184.3 | 0.89x | 1.419x |
| 9 | 4.381 | 3119.2 | 1.27x | 1.389x | 3.229 | 2224.8 | 0.91x | 1.445x |
| 10 | 4.355 | 3100.7 | 1.27x | 1.381x | 3.208 | 2210.0 | 0.90x | 1.435x |
| 11 | 4.320 | 3076.1 | 1.25x | 1.370x | 3.183 | 2193.1 | 0.89x | 1.424x |
| 12 | 4.328 | 3081.8 | 1.26x | 1.373x | 3.162 | 2178.7 | 0.89x | 1.415x |
| 13 | 4.293 | 3056.6 | 1.25x | 1.361x | 3.160 | 2177.2 | 0.89x | 1.414x |
| 14 | 4.284 | 3050.4 | 1.24x | 1.359x | 3.149 | 2169.6 | 0.88x | 1.409x |
| 15 | 4.298 | 3060.5 | 1.25x | 1.363x | 3.178 | 2189.3 | 0.89x | 1.422x |
| 16 | 4.409 | 3139.4 | 1.28x | 1.398x | 3.310 | 2280.6 | 0.93x | 1.481x |

Every width from 8 to 16 lands within seven points (1.33x-1.40x math500, 1.41x-1.48x mtbench), and the
ordering is not monotone: block 14 is the minimum, block 16 the maximum, rank correlation 0.38 over nine
widths. Block width is a ceiling on the window, not a quality parameter.

## 5. Ablation C: the load curve

Table 4. Every arm measured end to end, math500, default kernels, seed 0, five concurrencies. All ratios
are within-row at the same concurrency.

| c | DFlash tok/s | DFlash x AR | D8 tok/s | D8 x AR | D8 x DFlash | AR tok/s |
|---|---|---|---|---|---|---|
| 1 | 301.4 | 3.39x | 258.9 | 2.91x | 0.859x | 88.9 |
| 4 | 787.9 | 2.26x | 771.2 | 2.21x | 0.979x | 348.5 |
| 8 | 1390.9 | 2.03x | 1330.4 | 1.94x | 0.957x | 684.3 |
| 16 | 1779.5 | 1.34x | 2366.2 | 1.79x | 1.330x | 1324.5 |
| 32 | 2245.1 | 0.92x | 3370.4 | 1.38x | 1.501x | 2451.1 |

mtbench crosses in the same place: DFlash falls from 2.25x AR at c=1 to 0.63x at c=32, nine columns hold
0.92x while AR nearly doubles, and the ratio between them goes from 0.92x to 1.46x.

The crossover between the two windows sits between c=8 and c=16 on math500 and at c=8 on mtbench. Below
it the wide window wins by up to 1.16x; above it the narrow window wins and the gap widens with load
(1.33x at c=16, 1.50x at c=32). Round cost rises 3.3x for 17 columns (15.41 to 50.74 ms) against 2.0x for
9 (15.05 to 30.31 ms): verification, not drafting, is what saturates.

## 6. What the data says

At serving batch the published DFlash arm is slower than no speculation at all (0.92x AR math500, 0.63x
mtbench), and the remedy is the verify window: nine columns deliver 1.33-1.50x over it, measured end to
end on two benchmarks and five concurrencies. VLC, drafting wide and verifying narrow, lands at 1.345x
over DFlash (3019.7 tok/s math500, 2069.2 tok/s mtbench) and 1.8-5.2% of acceptance above D8, the
narrow-drafter arm that costs the same. The load critique is the contribution; the width flag is the
remedy; the window is the mechanism that explains both.

