# Proposal — Critique of DFlash and a verify-length improvement

Course: undergraduate ML group project. Guideline: select a paper, critique it, implement a trick of improvement (empirical or theoretical), present in class, submit a signed report with per-member contributions.

**Paper:** *DFlash: Block Diffusion for Flash Speculative Decoding*, Jian Chen, Yesheng Liang, Zhijian Liu. ICML 2026. arXiv:2602.06036.

**Project claim in one sentence:** DFlash spends 45% of its speedup when the server gets busy, and the fix the authors propose (smaller draft blocks) is the wrong knob; we measure the right one and show when it pays.

---

## 1. Background: speculative decoding in five lines

A large model generates one token per forward pass, and each pass costs a full read of its weights. Speculative decoding exploits this: a small cheap model (the **drafter**) guesses several tokens ahead, then the large model (the **target**) checks all the guesses in a single pass, because checking k tokens costs barely more than generating one.

The checking is exact, not approximate. Token *i* is accepted only if the target would have produced it anyway; the first disagreement stops the run and the target's own token is kept as a bonus. Output is therefore identical to the target model's output. When the drafter is right often, the target does the work of one step and emits many tokens.

Two numbers describe any such system:

- **τ (acceptance length)**: how many tokens are accepted per round, including the bonus token.
- **S (speedup)**: `S = τ / (draft cost + verify cost)`, both measured in units of one target step.

Everything interesting is a fight over those two terms.

## 2. What DFlash is, and why it beats EAGLE-3

EAGLE-3 is the strongest autoregressive drafter. It is a single small layer on top of the target's hidden states, and it drafts **sequentially**: d draft steps produce a tree of candidates, which the target then verifies in one pass. Sequential drafting means draft cost grows with how far ahead you look, `T_draft = d · t_step`.

DFlash replaces the sequential head with a **block diffusion drafter**: 5 Transformer layers (8 for the 30B coder model) that fill an entire block of 16 masked positions in **one forward pass**, conditioned on hidden states pulled from five layers of the target model. There is no denoising loop at inference; it is one pass, one shot.

| | EAGLE-3 | DFlash |
|---|---|---|
| Drafter | 1 layer, autoregressive | 5 layers, block diffusion |
| Draft depth per round | 6 to 8 (tree depth, EAGLE-3 p.12); DFlash runs its EAGLE-3 baseline at 7 steps (DFlash p.6) | **1 pass** |
| Draft cost | grows with lookahead, `d·t_step` | roughly flat in block width, `t_parallel` |
| Target conditioning | features as input | features injected into K/V of every draft layer |
| τ / speedup (Qwen3-8B, c=1) | 3.40 / 2.02× (DFlash Table 1, EAGLE-3 at 60 tokens); AdaFlash measures 4.40 / 2.34× for the same baseline | **8.01 / 5.1×** (DFlash Table 3, SGLang) |

The structural reason DFlash wins is not that diffusion is smarter. It is that **one wide pass is cheaper than several narrow ones**, so the same acceptance can be bought with less drafting compute. The magnitude is disputed in C2 below, because DFlash's drafter is also 5 to 8 times deeper than the head it is compared against. The direction of the advantage is not disputed.

## 3. Three critiques

All three come from DFlash's own tables.

**C1. The headline number is a batch-1 number.** At concurrency 1 DFlash reaches 5.1×; at concurrency 32 it reaches 2.8×, with τ unchanged at 8.01. That is **45.1% of its speedup lost to load**, and it is invisible in the abstract. Serving systems do not run at batch 1.

**C2. The EAGLE-3 comparison confounds parallelism with capacity, and the confound is smaller than it first looks.** DFlash's efficiency argument prices an autoregressive drafter at γ sequential steps, but its own EAGLE-3 baseline runs 7 steps for 10 to 60 tokens, and the DFlash drafter has 5 to 8 layers against EAGLE-3's single layer. Their own Table 6 ablates capacity at fixed parallelism (3, 5 and 8 layers at 4.69×, 4.71× and 4.64× on MATH-500), so capacity alone barely moves the number. What is never done is crossing capacity with drafter type: EAGLE-3 at 5 layers appears only in Table 9, a different experiment. So the honest form of the critique is narrow: "parallel beats sequential" is demonstrated, "one layer beats many" is not tested, and the headline multiplier depends on which EAGLE-3 configuration is chosen (2.02× at 60 tokens, higher at 10).

**C3. The authors' own future work picks the expensive knob.** DFlash closes by proposing adaptive block size. Shrinking the block to 8 costs **2.92 accepted tokens** (τ 8.01 → 5.09: two anchors from two tables, Table 3 average and Table 8's 8-layer MATH-500 ablation, not one measurement of one model). Truncating how much of the drafted block gets **verified**, at the same column count, costs at most about one token: τ 8.00 if acceptance rarely exceeds nine tokens, τ 7.21 under the tail we fit from the published calibration. Both reduce the target's work by the same amount. One of them throws away acceptance to get there.

## 4. What we propose

DFlash faces a two-knob budget problem every round, and the paper only ever turns one knob at a time:

- **b, the draft width**: how many tokens the drafter proposes.
- **L, the verify length**: how many of those proposed tokens are submitted to the target.

These are genuinely separable, because the target pass costs one dense forward over `L+1` columns while the drafter's cost is dominated by its weights and is roughly flat in width. So a system can **draft wide and verify narrow**.

**Our method (VLC, Verify-Length Control).** Draft the full 16-token block, then submit only the prefix worth verifying, chosen per round from the drafter's own confidence scores, which are already computed. Rule: extend the submitted prefix while the marginal expected acceptance gain exceeds the marginal verification cost; stop when it does not. No training, no new parameters, no extra forward passes, and the verifier is untouched, so the output stays lossless.

**Our ablation (W).** Serve the same checkpoint at block width 8, which the paper reports generalises from a block-16 checkpoint. This is the paper's own proposed direction, implemented as the comparison arm.

**The claim we test.** Under load, the two knobs are not equivalent: VLC keeps acceptance and cuts verification, while W cuts verification and acceptance together. If that is true, the paper's stated future work is dominated, and the right knob is verify length.

**What is already published, stated up front.** The marginal rule is LibraSpec's, training-free, and it has been run on DFlash. AdaFlash already truncates the draft before verification at concurrency, using a trained head rather than a rule. Neither has a block-width arm. Our claim is therefore a two-arm ablation at matched column count, not a new controller, and `docs/RELATED.md` records the page for each of those concessions.

## 5. Why this matters

**Narrowly:** a serving operator reading DFlash's abstract buys 5.1× and gets 2.8×. Knowing which knob to turn at batch 32 is worth more than any drafter-side trick, because the drafter is only about 6% of a round at batch 1 and less at batch 32. The verification pass is 90%+ of the cost and it is the only term with real headroom.

**Broadly:** the collapse under load is not a secret in this literature, it is a reported side effect that nobody repairs. XPress publishes 6.0× at batch 4 falling to 2.8× at batch 32; AdaFlash publishes 0.76× at concurrency 128, slower than plain autoregressive decoding; a block-diffusion tree drafter measured by a third party goes 3.14× to 0.84× from batch 1 to 64. Each paper reports the number and moves on. The one training-free controller that targets the verification side, LibraSpec, is evaluated single-stream. So the open question is not "does it collapse" but "does the cheap fix still work where the collapse happens", and that is measurable on one model with a controlled grid.

If it holds, the same argument applies to every block-diffusion drafter, because they all share the structure that makes it work: one wide drafter pass feeding one dense verification pass.

## 6. What results we will provide

Everything measurable is pre-registered, and the verdict comes from a script, not from our expectations. Thresholds are fixed before the first GPU run.

| ID | Question | Decides | Threshold |
|---|---|---|---|
| H1 | Is verification column-sensitive at batch 32? | whether the mechanism exists | `ms(9 cols)/ms(17 cols) ≤ 0.85` |
| H2 | Does truncation keep acceptance? | whether the gain survives | `τ ≥ 7.2` (≤10% loss) |
| H3 | Does VLC beat DFlash end-to-end? | the headline | `tok/s ≥ 1.15×` at c=32 |
| H4 | Does verify length beat block width? | our novel claim | `tok/s(VLC) ≥ 1.10× tok/s(W)` at c=32 |
| H5 | Any cost at batch 1? | deployability | `≥ 0.98×` at c=1 |
| H6 | Is output unchanged? | validity | greedy match `≥ 0.99` |

Deliverables from the runs: 51 measurements over `c ∈ {1,4,8,16,32}` on MATH-500 and MT-Bench with Qwen3-8B, one figure showing the gain of each knob against load, and a table of `τ`, `ms/round` p50/p95, and tok/s per cell. Two of the model's parameters (the share of verification cost that is pure weight read, and the share of acceptance clipped by truncation) are currently assumptions; the experiments measure them and we report the model's error either way.

**What we expect, stated as a hypothesis and not as evidence:** +15.6% at batch 1, about +29% at batch 8, about +35% at batch 32 for VLC; between −18% and −5% for block width 8. These come from a cost model anchored on the paper's two published points and calibrated against the one published measurement of this mechanism. They justify paying for the experiment; they do not predict its outcome, and the report will state the measured values whatever their sign.

**If it fails:** if verification turns out not to be column-sensitive (H1 fails), the method is unimplementable at serving batch, and the report becomes a measured negative result plus the three critiques. If the gain lands under 15% (H3 fails), the mechanism is real but too small to matter, which is also a result.

## 7. Deliverables, scope, and honesty

Per the guidelines: paper summary, critique, implemented improvement, in-class presentation, signed report with per-member contributions. The 13-day plan, GPU budget (about 10 A100-hours, roughly $15 to $25), and role split are in `docs/PLAN.md`; the experiment register is `docs/EXPERIMENTS.md`; the decision script is `scripts/decide.py`.

**Limits we will state in the report:** one target model family, greedy decoding in the headline cells, one serving backend, and a single-hardware regime map that may shift on other GPUs. Two of our three critiques rest on anchors taken from two different tables (τ 8.01 from Table 3, τ 5.09 from Table 8's 8-layer ablation), which the report will label as two anchors rather than one measurement. And verify-length selection is not our invention: LibraSpec publishes the marginal rule, training-free, on DFlash, and AdaFlash truncates the draft before verification at concurrency with a trained head. What is left is the block-width arm, at matched column count, above concurrency 1, which `docs/RELATED.md` shows none of the five related papers runs. We will say that in the introduction rather than let a marker say it for us.

## 8. References

- Chen, Liang, Liu. *DFlash: Block Diffusion for Flash Speculative Decoding.* arXiv:2602.06036, ICML 2026.
- Li, Wei, Zhang, Zhang. *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test.* arXiv:2503.01840.
- Lin et al. *LibraSpec: Dynamic Diffusion-Based Speculative Decoding via Marginal-Gain-Driven Optimization.* arXiv:2608.08721.
- Wang et al. *XPress: Parallel Refinement for Diffusion Drafters.* arXiv:2608.02438.
- Ringel, Romano. *Accelerating Speculative Decoding with Block Diffusion Draft Trees.* arXiv:2604.12989.
- Qian et al. *AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters.* arXiv:2607.19223.
