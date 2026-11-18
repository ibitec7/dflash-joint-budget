# dflash-joint-budget

Course project (undergraduate ML, group): a critique of **DFlash** (arXiv 2602.06036, ICML 2026) plus one implemented improvement, tested empirically.

## What the project claims

DFlash drafts a block of γ tokens in one forward pass and verifies all of them in one target pass, reaching 5.1× at concurrency 1 on Qwen3-8B. Three things follow from its own tables:

1. **45.1% of that speedup disappears by concurrency 32** (5.1× → 2.8× at fixed τ = 8.01, Table 3). The paper reports batch-1 numbers as its headline.
2. **The comparison with EAGLE-3 confounds parallelism with draft capacity.** The DFlash drafter has 5–8 layers against EAGLE-3's 1-layer head, and the paper's own Table 6 contains the compute-matched row it never compares.
3. **The paper's declared future work is the wrong knob.** It proposes adaptive block size. Reducing the block to 8 costs 2.92 accepted tokens (τ 8.01 → 5.09). Truncating the *verification* to the same number of columns costs almost nothing (τ 8.00) for the same compute saving, because the target pass is one dense block whose cost scales with columns, not with how many drafted tokens you choose to submit.

The improvement is therefore a **verify-length controller**: draft the full block, submit only the prefix worth verifying, chosen per round from the drafter's own confidence. It needs no training and no new model. The block-width arm is run as the ablation that proves the claim above.

## Status

- Cost model: **done**, `analysis/cost_model.py`, with tests. Its numbers are hypotheses, not evidence.
- Decision protocol: **done**, `docs/EXPERIMENTS.md` plus `scripts/decide.py`, which turns measured `results/*.json` into a verdict against pre-registered thresholds.
- GPU experiments: **not started.** The local GPU is a 4 GB RTX 3050 and cannot host the 8B target.
- Projected gains are **upper bounds**, calibrated against the one published measurement of this mechanism (LibraSpec: +15.6% over DFlash on Qwen3-8B at concurrency 1). They exist to justify paying for the experiment, not to predict its outcome.

## Projected results

The model has two free parameters: ρ (share of a batch-1 verification pass that is pure target-weight read) and w_tail (share of blocks whose acceptance exceeds the truncation cap). ρ is fitted to the one published measurement of this mechanism, LibraSpec's +15.6% over DFlash at concurrency 1, which gives ρ = 0.50 at w_tail = 0.10.

| Arm | c=1 | c=8 | c=32 |
|---|---|---|---|
| Verify truncation to 9 columns (the method) | **+15.6%** | **+29.3%** | **+35.1%** |
| Block width 8 (the ablation) | −18.4% | −8.7% | −4.6% |

Fitted at c=1 by construction, so the two informative cells are c=8 and c=32. Sensitivity: at w_tail = 0.25 and ρ = 0.20 the block-8 arm reaches +6.8% and the truncation arm falls to +26.1% at c=32. The block-8 arm turns positive only in that corner.

The single measurement that decides everything is ρ. If verification turns out to be nearly all weight-read (ρ → 1), truncation saves almost nothing at any concurrency and the project ships as a negative result. `analysis/cost_model.py --fit` reproduces the table.

## Layout

```
analysis/cost_model.py     cost model, anchors, projection tables (has --selftest)
scripts/decide.py          applies the pre-registered rules to measured results
tests/                     cost-model algebra + the decision rules themselves
docs/PLAN.md               full plan: experiments, GPU budget, projections, roles
docs/EXPERIMENTS.md        hypothesis register H1-H7, thresholds, decision matrix
results/                   measurement JSON, one file per (arm, concurrency, benchmark, seed)
figures/                   the frontier figure for the report and the talk
```

## Setup

```bash
uv sync                      # python 3.12, torch 2.15+cu132, transformers 5.x
uv run python analysis/cost_model.py
uv run pytest -q
```

Serving dependencies (`sglang`, `vllm`) are **not** installed here: they pin their own torch/CUDA builds and only run on the rented GPU host. Install them there, not on the laptop.

## What this repo is not

- It is not a claim that verify-length truncation is new. LibraSpec (2608.08721) publishes exactly this rule, training-free, and integrates it with DFlash and DDTree. The contribution here is the head-to-head against the block-width knob, measured above concurrency 1, which neither paper reports.
- It is not a coverage claim. Every number in `docs/PLAN.md` marked PROJECTED is arithmetic from published anchors, not a measurement.
