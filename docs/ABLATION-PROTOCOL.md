# Ablation protocol: is verify-length truncation a separate knob from block width?

Pre-registration for the claim *"truncating the verify window is a separate and better knob than
shrinking the draft block."* Every DFlash number is from **arXiv 2602.06036v2**, full text retrieved
2026-11-19 (`pdftotext -layout`), **PDF pages** cited. `NOT STATED` = absent from the whole document,
not merely from the pages read. This file specifies what to run; it reports no runs.

## 1. What the paper ablates

| Table (p.) | Axis varied | Cells / values | τ | Speedup | Concurrency | Hardware | Seeds |
|---|---|---|---|---|---|---|---|
| T1 (p.6) | method only; block fixed 16 (EAGLE-3 tree 16/60) | Q3-8B, T=0 | avg 6.49 vs 2.96 / 3.40 | 4.86× vs 1.76× / 2.02× | **single stream** | H200 | NOT STATED |
| T2 (p.7) | thinking mode on | Q3-8B T=0 MATH-500 | 5.82 | 4.64× | single stream | H200 | NOT STATED |
| T3 (p.7) | **concurrency** 1/4/8/16/32, block 16 | Q3-8B MATH-500 | 8.01 (one value, no per-c index) | 5.1× / 4.5× / 4.5× / 3.9× / **2.8×** | 1–32 | 1× B200, FA4, Spec-v2 | NOT STATED |
| T5 (p.8) | concurrency, block 10 | LLaMA-3.1-8B GSM8K | 4.32 | 2.4× → **1.6×** (c=32) | 1–32 | 1× B200, FlashInfer, Spec-v1 | NOT STATED |
| T6 (p.8) | draft depth 3-L/5-L/8-L, block 16 | MATH-500 | 5.64 / 5.99 / 6.33 | 4.69× / 4.71× / 4.64× | single stream | H200, greedy | NOT STATED |
| T7 (p.8) | #target hidden feats 3-H/5-H, block 16, 3-L | MATH-500 | 5.38 / 5.64 | 4.49× / 4.69× | single stream | H200 | NOT STATED |
| **T8 (p.8)** | **train BS × test BS: b16→b16 / b16→b8 / b8→b16 / b8→b8**, 8-L | MATH-500 | **6.33 / 5.09 / 5.02 / 5.21** | **4.64× / 3.87× / 3.78× / 3.97×** | **single stream** | H200, greedy | NOT STATED |
| T9 (p.9) | KV injection vs input fusion, block 8, 5-L, Q3-4B | GSM8K | 4.2 vs 3.5 | 3.3× vs 2.9× | single stream | H200 | NOT STATED |
| T10 (p.12) | drafter with no target features, 5-L | MATH-500 T=0 | 4.61 | 3.73× | single stream | H200 | NOT STATED |
| T13 (p.13) | anchor sampling standard vs random, 3-L, block 16 | MATH-500 | 4.94 / 5.64 | 4.13× / 4.69× | single stream | H200 | NOT STATED |
| Fig. 3 (p.3) | **draft latency only** vs #draft tokens {4,8,16}, 1/3/5-L + EAGLE-3 1-L | — | not measured | not measured | single stream | NOT STATED | — |
| T11 (p.12) | model scale, SGLang | c=8 only | 3.7–9.1 | 1.2×–3.9× | 8 | 1× B200 | NOT STATED |
| T12 (p.12) | vLLM, Qwen3.5-9B | 849→9836 tok/s | not reported | 4.0× → 1.9× (c=32) | 1/8/16/32 | NOT STATED | NOT STATED |
| p.7 §5.5 preamble | scope of every ablation | "single H200 GPU with greedy decoding, except those evaluated on SGLang" | — | — | **none above c=1** | H200 | — |
| p.9 §5.5.4 | position, not an experiment | "large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes); reducing the block size … can therefore yield better overall speedup. We leave adaptive block-size scheduling to future work." | — | — | — | — | — |

The last two rows are the whole opening: they name our mechanism, prescribe the *wider* knob for it,
and measure neither at batch. T8 is their only block-width evidence, and it is batch-1.

## 1b. Novelty anchor (verbatim, arXiv 2602.06036v2 §5.5.4 "Training-Inference Time Block Size")

> "DFlash models trained with larger block sizes generalize well to smaller inference-time block
> sizes. This property enables dynamic block-size scheduling during inference to improve end-to-end
> efficiency. In practical serving scenarios, large blocks can increase verification cost under
> compute-bound settings (*e.g*., large batch sizes); reducing the block size in such cases can
> therefore yield better overall speedup. We leave adaptive block-size scheduling to future work."

Read precisely, this does three things for us:

1. It **names our problem** (verification cost under serving batch), so the phenomenon is not ours to claim.
2. It **nominates block-size reduction as the remedy**, which is exactly the A1/A3 regime we already measured. Any VLC claim must beat that nominated fix, not just beat DFlash-at-block-16.
3. It **does not consider verify-window truncation at all** (full-text: `truncat` 0 hits, `prefix` 0, `window` 0, `partial` 0), and leaves the scheduling question open. That is the gap our cell fills; it is a gap in the remedy space, not in the problem statement.

Consequence for the report: the honest title of our contribution is "a second remedy for a cost the
authors identified", and the ablation below must show it dominates the remedy they nominated.

## 2. Does the paper separate draft width from verify length? No — one symbol

| Question | Answer | Anchor |
|---|---|---|
| Is there a verify-window parameter distinct from block size? | **No.** One symbol γ: "the draft model proposes γ tokens, which are verified in parallel"; τ ∈ [1, γ+1]. Verify columns = γ+1 by construction. | p.3 §3.1, Eq. 1 |
| Any cell holding γ_draft fixed while varying columns? | **NOT STATED.** Full-text counts: `truncat` 0, `prefix` 0, `partial` 0, `window` 0, `number of verified` 0. | whole document |
| What does T8's b16→b8 row change? | Both knobs at once: 17 verify columns → 9, *and* a different proposal distribution. τ 6.33→5.09 is the joint effect; it attributes to neither. | p.8 T8 |
| Why isn't a 16-block truncated to 9 columns the same as an 8-block? | Attention is bidirectional within a block (Fig. 4), so position *i* of a 16-block is computed with 16 positions in view. The two arms put **different tokens in the same 9 slots**. | p.5 §4.2 |
| Does the b8 arm pay an acceptance ceiling? | Yes: 35.7% of b8 blocks fully accepted on MATH-500, i.e. the arm saturates and τ is capped at 9 however easy the context. | p.9 |
| Is drafting cost claimed flat in γ? | Yes: "For moderate block sizes, Tdraft is therefore largely insensitive to γ", with Fig. 3 (4/8/16 tokens). That is the licence for drafting wide and verifying narrow — asserted at batch 1, so re-measure it. | p.3 §3.2 |

**Coverage:** the VLC cell does not exist in DFlash. What exists is (16,17) and (8,9), the diagonal of
the grid. The identifying cell is off it: **(16 draft, 9 columns)** vs **(8 draft, 9 columns)**.

## 3. The minimum design a reviewer would demand

### 3.1 The grid is a triangle, not a rectangle — declare that first

Columns ≤ drafts + 1 (p.3 Eq. 1), so (8 draft, 17 columns) is structurally absent; producing it needs
a second draft pass, which is a *different algorithm* and must be labelled as one.

| Cell = (draft γ, verify columns) | γ | columns | in paper? | role |
|---|---|---|---|---|
| A0 | 16 | 17 | yes (T1, T3, T8 b16→b16) | published baseline |
| **VLC** | 16 | **9** | **no** | **treatment: narrow window, wide draft** |
| **D8** | 8 | **9** | yes (T8 b16→b8, batch 1 only) | **control: narrow block** |
| D8-small | 8 | 5 | no | shows the control isn't a column artefact |
| AR | — | — | yes | anchor for the lossless check |

### 3.2 Axes

| Axis | Values | Why exactly these |
|---|---|---|
| verify columns k | {5, 9, 13, 17} at b=16; {5, 9} at b=8 | 9 is the matched cell, 17 the published one, 5 and 13 give the slope of τ(k) and ms_round(k) — the mechanism claim |
| draft width b | {8, 16}, **same b16 checkpoint run at BS 8** | T8 (p.8) shows b16→b8 generalizes and b8→b16 does not; a separately trained b8 checkpoint would confound width with capacity |
| concurrency c | {1, 8, 16, 32} | matches T3/T5/T12 so the batch regime is comparable; our crossover sits between 8 and 16 (RESULTS F3) |
| prompts / seeds | frozen MATH-500, 64 at c=32, 32 elsewhere; 3 seeds on A0, VLC, D8, AR at c=32 | EXPERIMENTS §4 rule 4, not to be silently dropped |

Six spec cells — (16,5) (16,9) (16,13) (16,17) (8,5) (8,9) — × 4 concurrencies = 24 runs, + AR + 8 repeats at c=32.

### 3.3 Error bars at this scale

| Item | Reviewer expects | Paper's practice |
|---|---|---|
| Throughput | 3 seeds, mean and min–max (not σ at n=3) | NOT STATED — no seed, repeat count, or variance anywhere |
| τ | exact repeat under greedy, else per-seed spread | NOT STATED |
| Quality | exact-match vs AR, per arm | NOT STATED — "lossless" asserted (p.1, p.3), never tabulated |
| Kernel mode | named for every number | FA4/Spec-v2 (p.7), FlashInfer/Spec-v1 (p.8); no determinism flag mentioned |

Determinism is the cheap substitute, and it must be shown: AR-vs-AR agreement is 0.422 with default
kernels and 1.000 with batch-invariant kernels (RESULTS F6), so default-kernel spread is backend
noise, not statistics. Rule: quality numbers only in deterministic mode, speed numbers labelled with mode.

### 3.4 How "same cost" must be constructed

| # | Construction | Definition | What it licenses |
|---|---|---|---|
| i | **Equal verify columns** (mandatory) | C_verify(VLC) = C_verify(D8) = 9; same query positions, same τ ceiling; only drafter latency differs, and that is measured | "at equal verification work the wide draft accepts more" — kills relabelling outright |
| ii | Equal round time | measure ms_round both arms; if the wider draft costs more, compare at the columns where ms_round matches and report the mismatch | "at equal wall-clock per round" — the strict form |
| iii | Equal cost per accepted token (mandatory) | frontier of τ vs tok/s over every cell in §3.2, both families | "better", not merely "different": VLC's frontier must lie above D8's at c=32 |

Report the identity per cell (p.3 Eq. 1): η = τ / (T_draft + T_verify), with `T_draft(b)` and
`T_verify(k)` measured separately — one nsys trace per arm at c=32 (EXPERIMENTS §4 rule 6). The claim
is that the two terms answer to *different* knobs; end-to-end tok/s alone is correlational and a
reviewer will say so. Decision rule (EXPERIMENTS §2b, in (γ, columns)): pass if `tok/s(16,9) ≥ 1.10 ×
tok/s(8,9)` at c=32 with `τ(16,9) ≥ 1.05 × τ(8,9)`; fail → report the break-even width.

## 4. What would make the claim fatally confounded

| # | Trap | Why fatal | Guard |
|---|---|---|---|
| T1 | **Backend coupling**: the "VLC" arm set via `speculative_num_draft_tokens=9`, which also sets the drafter block to 9 | the cell becomes (9,9) vs (8,9): one column apart, no wide draft. Exactly what produced 0.990 (RESULTS F5) | assert at runtime that drafter attention spans 16 positions and target logits span 9; log both counters |
| T2 | **Unequal target passes per accepted token** | a narrow arm needing more rounds pays more target passes; tok/s hides it | report counted target forwards per generated token, per arm |
| T3 | **Cross-machine comparison**: our c=32 VLC against T8's batch-1 H200 row (3.87×/5.09) | different GPU, batch, prompts, kernel mode — and load regime, the variable under test | all ratios internal to our grid; T8 an anchor, never a denominator |
| T4 | **Mismatched τ ceilings**: VLC caps at 9 | won trivially by whichever arm has the higher cap | ceilings equal by construction, printed per arm |
| T5 | **Different effective batch**: a 9-column arm admits more in-flight requests, so the win is admission | gain vanishes at equal occupancy | hold `max_running_requests` and in-flight batch fixed (SGLang clamps to 48, RESULTS §Configuration) |
| T6 | **Mixed kernel modes** across arms | the win moves +36% → +23% between modes (RESULTS F8); mixing can flip a 10% threshold | one mode per table, named in the caption |
| T7 | **Different accepted-prefix inputs**: each arm continuing its own prompt | τ differences become input differences | AR-generated prefixes frozen and shared byte-for-byte (EXPERIMENTS §2b) |
| T8 | **Wrong b8 arm**: a separately trained b8 checkpoint, or b8→b16 | T8's four cells read 4.64/3.87/3.78/3.97×; picking the wrong one moves the control by 0.2× | arm is "b16 checkpoint, inference BS 8", stated in the caption |
| T9 | **Arm bookkeeping drift**: RESULTS F5 calls A1 "draft 16, verify 9"; EXPERIMENTS §2b's H4 note establishes A1 ran draft 9 | a table with ambiguous arm labels cannot carry a knob claim; the reviewer's first act is to re-derive the config | every cell printed as the explicit pair (γ, C_verify), never an arm name |
| T10 | **tok/s without τ, ms_round and submitted length** | unfalsifiable, and hides T2 | EXPERIMENTS §4 rule 3, enforced by `decide.py` |

## 5. What this protocol cannot deliver

Novelty. AdaFlash already truncates the submitted prefix at C∈{1,32,64,128} with a trained head, and
LibraSpec has the training-free selection rule (RELATED.md). What remains is one controlled cell no
paper reports: **VLC (16,9) vs D8 (8,9) at c=32 — same columns, same ceiling, same batch**. If it does not
clear 1.10×, the honest result is that on a bidirectional-block drafter the two knobs are near-
substitutes, and the report says so, as EXPERIMENTS §2b already commits it to.
