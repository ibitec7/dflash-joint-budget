# Experiment register — what decides the project

**Rule: no projection decides anything.** Every number in `PLAN.md` §6 and in `README.md` is a
hypothesis with a decision rule attached below. The project's verdict is produced by
`scripts/decide.py` from measured `results/*.json`, not by the cost model, and not by anyone's
expectation. If a cell is missing, the verdict is INCOMPLETE and the report must say which
measurement is absent.

## 1. Measurements that are currently assumptions

Two numbers in the cost model are not in DFlash and were not measured by anyone we read. They are
measured here, and the fitted values (`ρ = 0.50`, `w_tail = 0.10`) are predictions, not inputs.

| Symbol | Meaning | How it is measured | Model's prediction |
|---|---|---|---|
| ρ | share of a batch-1 verification pass that is pure target-weight read, so invariant to column count | `ms/round` at c=32 for 17 vs 9 verify columns, same prompt set, same seed | 0.50 |
| w_tail | share of blocks whose acceptance exceeds the truncation cap (the tail that gets clipped) | `E[τ]` at 9 columns against `E[τ]` at 17, both measured | 0.10 |
| τ(L) curve | acceptance as a function of verify length | A2 sweep over columns {7,9,11,13,17} at c=32 | monotone, cap binds near 9 |

## 2. Hypothesis register

Each row states a prediction, the measured quantity, the threshold that decides it, and what the
report says if it fails. Thresholds are fixed **before** any GPU run and may not be moved
afterwards; if a threshold is wrong, the report says it was wrong and reports the measured value.

| ID | Hypothesis | Decides | Pass | Fail → report says |
|---|---|---|---|---|
| **H1** | The verification pass is column-sensitive at serving batch | whether the mechanism exists at all | `ms/round(A1)/ms/round(A0) ≤ 0.85` at c=32 | DFlash's verification is not column-dominated on this backend; the improvement is unimplementable at serving batch, and the measured ratio is the finding |
| **H2** | Truncating to 9 columns does not destroy acceptance | whether the gain survives the tail | `τ(A1) ≥ 7.2` at c=32 (≤10% loss vs 8.01) | truncation costs more acceptance than the model assumed; recompute with the measured `τ(L)` and report the corrected gain, not the projection |
| **H3** | The method beats DFlash end-to-end | the headline claim | `tok/s(A1)/tok/s(A0) ≥ 1.15` at c=32 | mechanism exists but the gain is under the useful threshold; report the measured number and the reason (verification share, overhead, padding) |
| **H4** | Verify length dominates draft width | the project's novel claim | `tok/s(A1) ≥ 1.10 · tok/s(A3)` at c=32 | the two knobs are equivalent; the dominance claim is withdrawn and the report presents them as equal-cost alternatives |
| **H5** | The method costs nothing at batch 1 | deployability | `tok/s(A1)/tok/s(A0) ≥ 0.98` at c=1 | per-round selection overhead is real; report it as the price of the method |
| **H6** | Output is unchanged | losslessness | `exact_match_ar(A1) ≥ 0.99` | the implementation changes model behaviour; the run is invalid and the bug is reported, not the result |
| **H7** | The controller beats the best fixed length | whether adaptivity is worth anything | `tok/s(A4) ≥ 1.05 · max_L tok/s(fixed L)` at c=32 | a fixed length is enough; the adaptive rule is reported as unnecessary complexity |

## 3. Decision matrix

Evaluated in order; the first matching row is the project verdict.

| Condition | Verdict | What ships |
|---|---|---|
| H6 fails on any arm | **INVALID RUN** | bug report; no results section |
| H1 fails | **NEGATIVE RESULT** | critique (C1–C3) + the measured column-insensitivity, which contradicts the method's premise |
| H1 ∧ H2 ∧ H3 ∧ H4 pass | **POSITIVE** | the method, the dominance table, the fitted ρ and w_tail against their measured values |
| H1 ∧ H2 pass, H3 fails | **MECHANISM ONLY** | the measurement contribution: the lever exists, the gain is below threshold |
| H1 ∧ H2 ∧ H3 pass, H4 fails | **POSITIVE, WEAKER CLAIM** | the method without the dominance claim |
| any required cell missing | **INCOMPLETE** | list of missing cells; no verdict |

## 4. Run protocol

Per run, write one JSON file to `results/` with these keys (validated by `scripts/decide.py`):

```json
{"arm": "A1", "benchmark": "math500", "concurrency": 32, "seed": 0,
 "draft_width": 16, "verify_columns": 9, "prompts": 200,
 "tau": 7.21, "tok_per_s": 151.0, "ar_tok_per_s": 110.0,
 "ms_per_round_p50": 1.90, "ms_per_round_p95": 2.31,
 "exact_match_ar": 0.997, "graph_recaptures": 0}
```

Rules that make the measurements trustworthy:

1. Prompts, seeds and decode parameters are fixed and committed **before** the first GPU run.
2. A0 must reproduce the paper's 5.1× at c=1 within 10%; if not, our A0 replaces it as the baseline and every ratio is against ours.
3. No row reports tok/s without `τ`, `ms/round` and the submitted length.
4. Three seeds on the four c=32 cells (A0, A1, A3, A4); report the spread, not just the mean.
5. `ms/round` is measured from the serving loop, not derived from throughput; p95 is reported because two-pass cycles widen latency tails.
6. One nsys trace per arm at c=32, so the verification share is observed rather than inferred.

## 5. What this register does not do

It does not settle whether +35% is "too much". That is not a question experiments answer directly —
they answer whether the measured gain is above a pre-registered usefulness threshold, and the
threshold is a judgement made now, in the open, at 15% (H3). Anything above it is a result;
anything below it is a narrower result. The projection's job was only to make the experiment worth
funding.
