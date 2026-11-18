# Project plan — DFlash critique and verify-length control

Course: undergraduate ML group project. Deliverables: pick a paper, critique it, implement an improvement, present in class, submit a signed report with per-member contributions.

Paper under critique: **DFlash: Block Diffusion for Flash Speculative Decoding**, Chen, Liang, Liu, ICML 2026, arXiv 2602.06036v2.

Repo: `/home/ibrahim/uni/dflash-joint-budget`. Status: cost model complete and tested; GPU work not started.

---

## 1. Thesis

DFlash's own tables show that 45.1% of its speedup disappears between concurrency 1 and 32, and that its declared future work (adaptive block size) is the *expensive* way to spend less verification compute. The cheap way is to draft the full block and verify only the prefix worth verifying. The report measures both, head-to-head, above concurrency 1, which no paper in this family does.

This is a critique-first project. The improvement is small in mechanism and large in decision value: it tells a serving engineer which knob to turn.

## 2. Phase 1 — critique (three findings)

| # | Finding | Anchor |
|---|---|---|
| C1 | **The headline is a batch-1 number.** 5.1× at c=1, 2.8× at c=32, τ fixed at 8.01. Everything that degrades is in the verification term. 45.1% of the speedup is lost to load | Table 3, p.7 |
| C2 | **Parallelism is confounded with capacity.** A 5–8 layer drafter is compared against a 1-layer EAGLE-3 head; eq.(2) prices an AR drafter at γ·t_step while the paper's own EAGLE-3 runs 7 steps for 10–60 tokens (EAGLE-3 at 60 tokens scores 0.6× at c=32, which the "up to 2.5×" claim omits) | p.3 eq.2, p.8 Table 5 |
| C3 | **The declared future work is the wrong knob.** Adaptive block size spends acceptance it does not have to: b=8 gives τ 5.09 (a 2.92-token loss). Truncating verification to the same column count gives τ 8.00 (0.01-token loss) for approximately the same compute saving | p.9 future work; Table 8 τ at b=8 |

C3 is the finding that motivates the implementation, and it is the one a grader can check against the plan's arithmetic.

## 3. Phase 2 — method

**VLC (Verify-Length Control).** Serve one DFlash checkpoint at block width b=16. Each round:

1. Draft the full block (17 columns) with the existing drafter and target-context KV injection. Unchanged from DFlash.
2. Read the drafter's own per-position confidences `q_1..q_b` (already computed, no extra cost).
3. Choose the submitted length `L` maximising estimated speedup `τ̂(L) / (κ + v(L,c))`, where `τ̂(L) = min(E[τ], L+1)` under prefix gating and `v(L,c)` is the measured verification cost at `L+1` columns. Solved by a closed-form marginal criterion (LibraSpec Theorem 3.4) or a 17-entry lookup built from measurement.
4. Submit `L` tokens; the target verifies `L+1` columns in one pass; unchanged verifier, so output is lossless.

**Why not just adapt block width.** Because width and verify length are separable in this method: the drafter's cost is dominated by its weights and is roughly flat in columns, while the verifier's cost is dominated by how many columns it processes. Reducing width cuts both, but pays 2.92 accepted tokens for it. Truncation cuts the expensive one for free. The ablation (A3 below) measures exactly this.

**Novelty, stated honestly.** The truncation rule is LibraSpec's (2608.08721), training-free, already integrated with DFlash and DDTree. The contribution here is (a) the width-vs-length dominance test, (b) measurement above concurrency 1, which neither DFlash nor LibraSpec reports, (c) the regime map that tells a server operator when the lever is worth pulling. Claim it as a measurement and decision contribution, not as a new algorithm.

## 4. Phase 3 — experiments

All arms serve the same target (Qwen3-8B, bf16) and the released DFlash drafter. Decode: greedy, max 256 new tokens, 200 prompts per benchmark.

| ID | Arm | Draft width | Verify columns | Selection | Purpose |
|---|---|---|---|---|---|
| A0 | DFlash as published | 16 | 17 | none | baseline, must reproduce 5.1× / 2.8× |
| A1 | VLC-fixed | 16 | 9 | fixed L=8 | the method, simplest form |
| A2 | VLC-sweep | 16 | {7,9,11,13,17} | fixed | locates the optimum, tests the τ-cap assumption |
| A3 | Width reduction | 8 | 9 | fixed | the ablation that proves C3 |
| A4 | VLC-adaptive | 16 | chosen per round | marginal rule on q | the deployable form |
| A5 | AR reference | — | — | — | speedup denominators |

Grid: A0, A1, A3 across `c ∈ {1,4,8,16,32}` × two benchmarks (MATH-500 high-τ, MT-Bench low-τ) = 30 runs. A2 at c=32 on MATH-500 = 5 runs. A4 across `c ∈ {1,8,32}` × two benchmarks = 6 runs. A5 at all five concurrency levels, 100 prompts × 64 tokens = 10 runs. **51 runs total.**

Metrics per run: `E[τ]` (accepted tokens incl. bonus), `L` submitted (mean, histogram), end-to-end tok/s, `ms/round` p50 and p95, graph re-captures, and the verification cost share from a one-off nsys trace per arm.

Protocol rules: greedy outputs must match the AR baseline token-for-token on ≥99% of prompts (losslessness check); no table row may report tok/s without `E[τ]` and `ms/round`; three seeds on the four headline cells; prompts fixed and committed before any GPU run. Every run writes `results/<arm>_<bench>_c<c>_s<seed>.json` in the schema at `docs/EXPERIMENTS.md` §4, and the verdict is produced by `uv run python scripts/decide.py --results results --benchmark math500`, never typed by hand from a table.

## 5. GPU budget

| Item | Hardware | Hours |
|---|---|---|
| Harness setup: SGLang install, drafter wiring, prompt set, losslessness check | 1× A100 40/80 GB | 3.0 |
| P0 shape probe: `ms/round` at c=32 with 9 vs 17 verify columns | same | 0.5 |
| Main grid (30 runs × ~2 min average) | same | 1.0 |
| Sweep + adaptive + AR reference (21 runs) | same | 0.7 |
| Seeds and re-runs on 4 headline cells | same | 0.5 |
| Contingency (framework bugs, one failed session) | same | 4.0 |
| **Total** | **1× A100** | **~10 GPU-hours, 2 sessions of 5 h** |

Cost estimate: A100 40 GB at $1.29–1.89/h → **$13–19**; A100 80 GB at $1.49–2.29/h → **$15–23**. If the host offers an H100 at ≤$2.50/h, take it: verification is compute-bound at high batch, and the faster verifier makes the effect easier to see.

Everything else (cost model, analysis, figures, writing) runs on the laptop and costs nothing.

## 6. Projected gains

**Decision rule (binding).** The numbers in this section are **hypotheses under test**, not evidence. The project's verdict is produced by `scripts/decide.py` from measured `results/*.json` against thresholds pre-registered in `docs/EXPERIMENTS.md`, and no threshold may move after a run. A missing cell yields INCOMPLETE, never a result. Nothing here, and nobody's intuition about what is "too high", decides whether the method works; the measurements do.

From `analysis/cost_model.py`, anchored on the paper's two measured points (v(1)=1.471, v(32)=2.761 AR-step equivalents). Two parameters are not identified by the paper:

- **ρ**, the share of a batch-1 verification pass that is pure target-weight read (column-independent floor).
- **w_tail**, the share of blocks whose acceptance exceeds the truncation cap. Ignoring it overstates every gain, because truncation clips that tail. At w_tail = 0.10, capping at 9 columns costs 0.80 τ (8.01 → 7.21).

**ρ is fitted, not assumed.** Setting the model's c=1 prediction equal to LibraSpec's measured +15.6% over DFlash identifies ρ = 0.50 at w_tail = 0.10. One published measurement, one unknown, so the projection is a point estimate with a stated sensitivity band rather than a free parameter:

| Arm | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| A1 truncate to 9 columns | +15.6% | +25.5% | +29.3% | +32.4% | **+35.1%** |
| A3 block width 8 | −18.4% | −11.4% | −8.7% | −6.5% | **−4.6%** |

Sensitivity: the fit moves to ρ = 0.69 at w_tail = 0 (c=32: +40.1%), and to ρ = 0.20 at w_tail = 0.25 (c=32: +26.1%, with block-8 reaching +6.8%). The block-8 arm is positive only in that corner, where ρ is implausibly low and the clipped tail implausibly large.

Read these as: **the method is worth 25–35% at serving batch, roughly 15% at batch 1, and the width knob is neutral to harmful everywhere the fit is plausible.**

Two model results that shape the story:

- **The optimal cap is not identified by the model.** Under the fitted parameters it sits at 7 columns (τ 7.00, +49.8% at c=32) rather than 9 (+35.0%); with a coarser tail model it moves to 9. An earlier version of this plan claimed the optimum was regime-invariant, and that was a tautology: `τ(L) = min(τ_max, L+1)` forces the optimum onto the cap by construction. The A2 sweep must measure `τ(L)` directly, and the report may not state where the cap belongs until it does.
- **The width arm is dominated wherever the fit is plausible** (−18% at c=1 to −5% at c=32). That is the ablation's value: it converts the paper's own future-work sentence into a quantified mistake.

## 7. Risks and kill criteria

| # | Risk | Test | If it fires |
|---|---|---|---|
| R1 | The verify pass is not column-sensitive (padding to 17 columns regardless) | P0 probe: `ms/round` at 9 vs 17 columns must drop ≥15% at c=32 | Arm dead. Report the negative result: DFlash's verification is not column-dominated on this backend |
| R2 | Truncation costs more τ than the model assumes | A2 sweep: τ at L=8 must be ≥7.5 | Recompute with the measured τ(L) curve and report the corrected gain |
| R3 | τ at b=8 is transplanted from LLaMA-3.1-8B (Table 8) | A3 measures it directly on Qwen3-8B | Use our own number; if it differs by >10%, say the paper's generalisation claim is model-specific |
| R4 | The released DFlash drafter does not load, or SGLang pins a per-request verify length | Day-1 feasibility check | Fall back to the Transformers backend at c=1 only, and label the grid as c=1 |
| R5 | A100 numbers differ from the paper's B200 numbers | A0 must reproduce 5.1× at c=1 within 10% | Report our A0 as the baseline and rescale every ratio to it |

The project fails only if R1 and R4 both fire. In that case the report ships as the critique (C1, C2, C3) plus the cost model, with the empirical section labelled NOT MEASURED, and that is still a complete submission against the rubric's "theoretical" branch.

## 8. Deliverables, roles, timeline

| Rubric item | Deliverable | Owner | Evidence of contribution |
|---|---|---|---|
| Paper selection + summary | 2-page summary, why it is critiqueable | Member A | git blame on `docs/summary.md`, 6–8 h |
| Critiques | C1, C2, C3 with page cites and the C3 arithmetic | A + B | `docs/critique.md`, `analysis/cost_model.py` |
| Implementation | VLC serving path + losslessness check | B | commits to `src/vlc/`, 10–14 h |
| Experiments | 51 runs, `results/*.json`, one figure | C | `scripts/run_grid.py`, GPU session logs |
| Report | 6 sections, ≤6 tables, 1 figure, ≤4500 words | D | `docs/report.md` |
| Presentation | 10 min, the dominance table as slide 3 | all, C leads | `docs/slides.pdf` |

Contribution claims must be per-module with hours and commit ranges, because the rubric asks for detail and signatures.

| Days | Work |
|---|---|
| 1–2 | Repo setup, SGLang install on rented box, DFlash drafter loads, A0 reproduces c=1 |
| 3 | P0 shape probe (R1). **Stop and re-plan if it fails** |
| 4–5 | Main grid + sweep + adaptive, 2 GPU sessions |
| 6–7 | Analysis, figure, cost-model calibration against measured τ(L) |
| 8–10 | Report draft |
| 11 | Slides |
| 12–13 | Buffer, then signatures and submission |

## 9. Open items to verify before the first GPU hour

1. Does the released DFlash drafter support a per-request verify length in SGLang (`num_speculative_tokens` or equivalent)?
2. Is the released checkpoint the 5-layer Qwen3-8B block-16 drafter used in Table 3, or the b=10 LLaMA variant?
3. What is `ms/round` at c=32 for 17 columns on an A100, in absolute terms? This calibrates the whole projection.
4. Does LibraSpec report any concurrency above 1? If yes, our "nobody measures this" claim narrows to "nobody measures the head-to-head".
