# Related work: what is ours, and what is not

Positioning for the verify-length-control (VLC) report. Every row is anchored to a page of the
version opened on 2026-11-18 (provenance at the end). Quotes are verbatim, under 30 words. "No
concurrency axis on the pages read" means exactly that — none appears in the sections listed, not
that the authors deny one. Pages are PDF pages.

## Positioning table

| Paper | What it does | What it measures (page, quote) | What our report claims that it does not |
|---|---|---|---|
| **LibraSpec** 2608.08721v1 | Training-free controller that decides how many drafted tokens to submit, from drafter confidence, by a marginal-gain criterion. Plug-and-play on existing diffusion drafters. | p.1 abstract: "derive a marginal criterion that extends the speculative sequence only when its acceptance gain outweighs the additional verification cost". p.2 §1: "requires neither an auxiliary predictor nor additional training". p.3 §2.1: integrated with "FastdLLM, DFlash and DDTree". p.6 Table 2, Q3-8B avg: 4.43× → 5.12× = +15.6%. | **The rule is theirs.** Marginal-gain-per-token selection on a diffusion block, training-free, already published and already run on DFlash. Nothing in our method section is new relative to this paper. What it does not do: no concurrency axis on the pages read (§4.1–4.6, pp.5–7 are per-benchmark speedup at one operating point), and no block-width arm — the drafter is held fixed. |
| **XPress** 2608.02438v2 | Lightweight causal refiner that adds a per-position logit bias through a strictly lower-triangular mixer, restoring intra-block causality; resolved by K parallel Jacobi passes instead of a serial sweep. | p.3 §1: "all positions are updated in parallel from the previous iterate, and K ≪ B such passes". p.9 §4.2: "We find K ≈ 6–7 iterations suffice to match … a 15-step sequential decode". p.14: "The iteration count K is currently fixed for a whole run". p.11 §5.2: "reaches a 6.2× average speedup at 8.02 accepted tokens"; p.11 Table 2, GSM8K: "XPress moves from 6.0× at bs=4 to 2.8× at bs=32". | It is not verify-length control at all — it repairs the drafter's distribution and leaves verification untouched. It is **trained** (§4.3, co-trained with a frozen target), so it is not an alternative to a training-free method. And it already publishes a batch-size sweep: the 6.2×→2.8× collapse under load is their measurement, not our discovery. K fixed per run is their own stated future work, not our critique. |
| **DDTree** 2604.12989v1 | Builds a draft tree directly from one-pass marginals, selecting continuations with a best-first heap under a node budget; verifies the tree in one target pass with ancestor-only attention. | p.1 abstract: "Under a fixed node budget, DDTree uses a simple best-first heap algorithm". p.4 §4.1: "DDTree builds a draft tree under a node budget B". p.10 Table 1, Q3-8B MATH-500, T=0: τ 7.79 (DFlash) → 10.73; T=1: 6.46 → 9.54. p.11 §5.3: speedup peaks at budget "256 to 512" then falls. | **No concurrency anywhere.** p.10 §5.1 runs on "8 H200 GPUs"; p.17 App. B clarifies they "shard the evaluation set across workers" — data-parallel evaluation, not batching. So it neither measures nor claims behaviour above concurrency 1. Also: its budget is tree *width*, which widens what is verified; it is not the length of a single linear block, which is what we shorten. Its budget-vs-speedup curve is the closest published analogue of our frontier, and we should say so rather than claim the shape. |
| **AdaFlash** 2607.19223v3 | On-policy distilled diffusion drafter (reverse-KL, clipped, online) **plus** a trained adaptive length head that predicts an acceptance ratio and truncates the draft to it; serving scheduler packs variable lengths. | p.5 §3.2: "only the first k̂_t tokens of the draft sequence are sent to the target model for verification", with k̂_t = clamp(⌊â_t·k⌋, 1, k). p.8 Table 1, Q3-8B vs DFlash: C=1 3.53×→4.06×, C=32 1.54×→1.74×, C=128 0.76×→1.15×. | **Verify-length control measured at concurrency is not our gap** — this paper does it at C ∈ {1,32,64,128}. Ours differs only in being training-free (vs a head trained by MSE and updated online, p.6 §3.2) and in selecting per-token by marginal gain rather than predicting a scalar ratio. It has **no block-width arm**: its DFlash baseline is fixed at k=15. That missing arm is the only thing left of our "which knob" claim. |
| **EAGLE-3** 2503.01840v3 | One Transformer decoder layer drafting autoregressively over fused low/mid/high target features (training-time test), with a context-aware dynamic draft tree and tree attention. | p.5 §3.1: "subsequently inputted into a single layer decoder". p.4 §2.2: "EAGLE-3 also adopts the context-aware dynamic draft tree". p.12 App. A: "increase the draft tree depth from 6 to 8". p.8 Table 3: SGLang bs 2→64, "1.38x" at bs=64 (LLaMA-3.1-8B, chain length 3). | **Our "7 draft steps" is a misattribution and must be fixed.** EAGLE-3's own depth is 6→8 (p.12); 7 is the value *DFlash* chose for its EAGLE-3 baseline — DFlash p.6: "the draft steps and top-k are set to 7 and 10". EAGLE-3 also already reports batch sizes up to 64, so it is not a batch-1-only baseline. What stands from C2 is narrower: our 4.40 τ / 2.34× is not an EAGLE-3 measurement at all — it is AdaFlash's Table 1 average (p.8), while DFlash's Table 1 (p.6) puts EAGLE-3 (60) at 2.02× / τ 3.40 on the same model. |

## Five corrections this exercise forced on our own text

1. **"No one measures above concurrency 1" is false.** XPress (p.11) sweeps bs 4–32 and AdaFlash (p.8) sweeps C ∈ {1,32,64,128}. Only LibraSpec and DDTree are single-stream.
2. **"7 draft steps" for EAGLE-3** is DFlash's baseline setting (DFlash p.6), not EAGLE-3's configuration (6→8, EAGLE-3 p.12). Cite the source of the number.
3. **Our EAGLE-3 baseline numbers are borrowed from a third paper.** 4.40 τ / 2.34× matches AdaFlash Table 1 (p.8) exactly; DFlash's own Table 1 (p.6) reports EAGLE-3 (60) at 2.02× / τ 3.40 for Qwen3-8B. Label the source, or re-run EAGLE-3.
4. **The block-width anchor spans two DFlash tables.** τ 8.01 comes from Table 3 (p.7, SGLang, avg τ, Qwen3-8B); τ 5.09 comes from Table 8 (p.9, b16→b8, the *8-layer* ablation on Math500). They are not one measurement of one model. Report the pair as two anchors or re-measure.
5. **The "compute-matched row" critique (C2) survives but shrinks.** DFlash Table 6 (p.8) does contain 3-L / 5-L / 8-L at 4.69×/4.71×/4.64× on Math500, so capacity is separable from parallelism in their own ablation — but they never cross it with a 1-layer head, and EAGLE-3-5L appears only in Table 9 (p.9), a different experiment.

## Residual contribution

In the pages we read, no paper reports verify-length control and block-width reduction head-to-head
at the same verification-column count **and** above concurrency 1: LibraSpec has the rule but is
single-stream and holds the drafter fixed, AdaFlash has concurrency but a trained head and no width
arm, and DDTree's budget is tree nodes rather than submitted prefix length. That leaves our
contribution as a controlled two-arm ablation on one model, not a new controller — the claim that
the two knobs are non-substitutes, one keeping acceptance while cutting cost and the other cutting
both, is untested by any of them. If H1 (column sensitivity) or H4 (VLC ≥ 1.10× the block-8 arm)
fails, nothing is left that these five papers do not already cover, and the report should say so
rather than dress up the critique as a method.

## Provenance

Opened and read on 2026-11-18 via alphaXiv page-filtered PDF retrieval:

| Paper | Version | Pages actually read |
|---|---|---|
| DFlash 2602.06036 | v2 | 1, 2, 3, 6, 7, 8, 9, 12, 13 |
| LibraSpec 2608.08721 | v1 | 1, 2, 3, 5, 6, 7, 9, 11 |
| XPress 2608.02438 | v2 | 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14 |
| DDTree 2604.12989 | v1 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17 |
| AdaFlash 2607.19223 | v3 | 1, 4, 5, 6, 8, 10, 17, 20, 21 |
| EAGLE-3 2503.01840 | v3 | 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 |

No row is UNREAD. Two limits on the above: pages not listed were not read, so "no concurrency axis"
is a statement about the listed sections only, and AdaFlash's appendices were read only where they
touch DSpark (pp.20–21) — a block-width arm there would not have been seen.
