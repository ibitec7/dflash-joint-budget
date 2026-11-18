# dflash-joint-budget

Course project (undergraduate ML, group): a critique of **DFlash** (arXiv 2602.06036, ICML 2026) plus one implemented improvement, tested empirically.

## What the project claims

DFlash drafts a block of γ tokens in one forward pass and verifies all of them in one target pass, reaching 5.1× at concurrency 1 on Qwen3-8B. Three things follow from its own tables:

1. **45.1% of that speedup disappears by concurrency 32** (5.1× → 2.8× at fixed τ = 8.01, Table 3). The paper reports batch-1 numbers as its headline.
2. **The comparison with EAGLE-3 confounds parallelism with draft capacity.** The DFlash drafter has 5–8 layers against EAGLE-3's 1-layer head, and the paper's own Table 6 contains the compute-matched row it never compares.
3. **The paper's declared future work is the wrong knob.** It proposes adaptive block size. Reducing the block to 8 costs 2.92 accepted tokens (τ 8.01 → 5.09). Truncating the *verification* to the same number of columns costs almost nothing (τ 8.00) for the same compute saving, because the target pass is one dense block whose cost scales with columns, not with how many drafted tokens you choose to submit.

The improvement is therefore a **verify-length controller**: draft the full block, submit only the prefix worth verifying, chosen per round from the drafter's own confidence. It needs no training and no new model. The block-width arm is run as the ablation that proves the claim above.

## Status

- Cost model: **done**, `analysis/cost_model.py`, with tests. All projections below are model output.
- GPU experiments: **not started.** The local GPU is a 4 GB RTX 3050 and cannot host the 8B target.
- Projected gains are **upper bounds**, calibrated against the one published measurement of this mechanism (LibraSpec: +15.6% over DFlash on Qwen3-8B at concurrency 1).

## Projected results (from the cost model, calibrated ~2× optimistic)

| Arm | c=1 | c=8 | c=32 |
|---|---|---|---|
| Verify truncation (the method) | +5 to +17% | +26 to +38% | +36 to +47% |
| After calibration (÷2, from the LibraSpec anchor) | **+2 to +9%** | **+13 to +19%** | **+18 to +24%** |
| Block width 8 (the ablation) | −8 to −27% | −16 to −22% | −5 to −17% |

Raw model at ρ = 0.9 (mostly memory floor): truncation +5.1% / +26.1% / +36.1%; at ρ = 0.3: +52.9% / +68.5% / +75.0%. Width arm at ρ = 0.7: −26.8% / −16.0% / −11.1%.

## Layout

```
analysis/cost_model.py     cost model, anchors, projection tables (has --selftest)
tests/test_cost_model.py   anchors, algebra, and the central dominance claim
docs/PLAN.md               full plan: experiments, GPU budget, projections, roles
results/                   measurement JSON, one file per (arm, concurrency, benchmark)
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
