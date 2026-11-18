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
| **H4 note (2026-11-18, before any VLC run)** | as scored, A1 and A3 were the same configuration | implementation | A1 ran `speculative_num_draft_tokens=9`, which sets the DRAFTER's block to 9 as well (`dflash_worker_v2.py:511-523`, `set_block_size(self.block_size)`), so the arm was block-9/verify-9 against block-8/verify-8. The measured 0.998 (default kernels) and 0.990 (deterministic) are a property of that coupling, not of the two knobs | H4 is re-scored as **INVALID-BY-CONFOUND**, not FAIL. The VLC test is re-specified below and its result replaces this row |
| **H5** | The method costs nothing at batch 1 | deployability | `tok/s(A1)/tok/s(A0) ≥ 0.98` at c=1 | per-round selection overhead is real; report it as the price of the method |
| **H6** | Output is unchanged | losslessness | `exact_match_ar(A1) ≥ 0.99` | the implementation changes model behaviour; the run is invalid and the bug is reported, not the result |
| **H7** | The controller beats the best fixed length | whether adaptivity is worth anything | `tok/s(A4) ≥ 1.05 · max_L tok/s(fixed L)` at c=32 | a fixed length is enough; the adaptive rule is reported as unnecessary complexity. **Conditional:** A4 needs E0 to confirm the backend exposes a per-request verify length. If it does not, H7 is N/A, not failed, and the report says the adaptive arm was unimplementable on this backend |

## 2b. H4-VLC, the test that actually separates the knobs (pre-registered; run, NOT SUPPORTED)

The drafter's block attention is **bidirectional** (`dflash.py:41 is_causal = False`, and the
drafting forward at `:244` passes `is_causal=False`), so the proposal at position i is computed
with the whole block in view: block 16 and block 9 produce different proposals at every position,
not just a longer suffix. Truncation is therefore a real alternative and not a relabelling of the
block-width knob.

**Definition.** VLC(b, k) = draft with block b, verify only the first k proposals.
`τ_k(b) = E[min(a, k)]` where `a` is the accepted prefix of the block-b draft.

**The one control, used by both rows.** D8 = **8 drafts into 9 verify columns** (`draft_width` 8 /
`verify_columns` 9, `results/A3_math500_c32_s0.json`). A round commits up to its verify-column count,
so the harness cell that realises D8 is `--narrow 9` and its tau is `τ_9(9)` = 4.381
(`experiments/standalone/accept_math500.json`). Both bars below name that arm and only that arm: an
8-column arm would verify one column fewer than the treatment and is not the registered control.

**Pre-registered test.** On identical contexts (AR-generated prefixes, so every arm sees the same
input tokens and the same target hidden states), measure `τ_k(16)` and `τ_k(9)` for k = 1..9.

| H4-VLC | statement | threshold | if it fails |
|---|---|---|---|
| acceptance | a wider draft improves first-k acceptance | `τ_9(16) ≥ 1.05 · τ_9(D8)` | the extra 7 draft positions buy nothing in the verified prefix; VLC is refuted with a mechanism, and block-width control is the method to report |
| end-to-end | the acceptance gain beats the extra drafter cost | `tok/s(VLC(16,9)) ≥ 1.10 · tok/s(D8)` at c=32 | acceptance gain exists but does not pay for the wider drafter pass; report the break-even width |

Cost model the second row tests, with all terms measurable:
`cost(b,k) = drafter(b) + target(k+1)` and VLC wins iff
`τ_k(16)/τ_k(9) > cost(16,k)/cost(9,k)`.

**Falsification is the point:** if `τ_9(16) < 1.05 · τ_9(D8)`, the honest report says the verify-length
knob is dominated by the block-width knob on a bidirectional-block drafter, and that is a finding.

**Outcome (measured, `experiments/standalone/accept_math500.json`; n = 32 frozen prompts, 7079 rounds,
one pass per drafter block).** Every number below is read from that file, or from
`results/A3_math500_c32_s0.json` and `results/A1_math500_c32_s0.json` for the round costs:

| H4-VLC | measured | bar | verdict |
|---|---|---|---|
| acceptance | tau ratio **1.018**, 95% CI [0.989, 1.049] (paired over 32 prompts; 20 wins / 12 losses, p = 0.215) | 1.05 | **FAIL** |
| end-to-end | predicted tok/s ratio **1.013** (tau 1.018 x cost factor 0.995, the cost factor being round(9 cols)/round(16 cols) = 45.07 / 45.29 ms at batch 1) | 1.10 | **FAIL** |
| overall | both rows fail; the 95% CI excludes 1.05 | | **NOT SUPPORTED** |

Verdict **NOT SUPPORTED** (RESULTS F15). A 9-column window returns 1.8% more accepted tokens from a
block-16 draft than from the 8-draft/9-column control, and at batch 1 the verify pass is memory-bound,
so the wide draft buys no round-time saving to multiply it by. Block-width control is the method to
report; the detail is in RESULTS F15 and F16.

## 3. Decision matrix

Evaluated in order; the first matching row is the project verdict.

| Condition | Verdict | What ships |
|---|---|---|
| H6 fails on any arm | **INVALID RUN** for every quality claim; throughput only within one kernel mode | bug report; no results section. What is still usable from a default-kernel grid is stated below the matrix |
| H1 fails | **NEGATIVE RESULT** | critique (C1–C3) + the measured column-insensitivity, which contradicts the method's premise |
| H1 ∧ H2 ∧ H3 ∧ H4 pass | **POSITIVE** | the method, the dominance table, the fitted ρ and w_tail against their measured values |
| H1 ∧ H2 pass, H3 fails | **MECHANISM ONLY** | the measurement contribution: the lever exists, the gain is below threshold |
| H1 ∧ H2 ∧ H3 pass, H4 fails | **POSITIVE, WEAKER CLAIM** | the method without the dominance claim |
| any required cell missing | **INCOMPLETE** | list of missing cells; no verdict |

**Why the default-kernel grid is not thrown out, and what it may not support.** H6 fails on that grid
by construction: two identical AR runs agree on only 27/64 = 0.422 of generations (RESULTS F6), so
`exact_match_ar` there measures backend nondeterminism, not model behaviour. That invalidates every
quality reading taken in the default-kernel mode, and every comparison that spans kernel modes (F8
moves the same win from +36% to +23%). It does not invalidate a throughput ratio taken **within** one
mode on one backend, which is all the c-sweep and F1-F4 claim: those numbers stand as throughput
measurements, labelled with their kernel mode, and none of them may be quoted as a losslessness or
quality result. The losslessness claim (H6) rests solely on the deterministic grid in `results_det/`,
where every arm matches AR on all 64 prompts (1.000) and A1/A3 read 0.9903.

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
2. A0 must reproduce the paper's 5.1× at c=1 within 10%, i.e. `tok/s(A0) / tok/s(A5) ≥ 4.59` at c=1 on the frozen prompts. **This check was run and it failed:** 3.39× with default kernels (301.44 / 88.91 tok/s, `results/A0_math500_c1_s0.json`, `results/A5_math500_c1_s0.json`) and 4.08× with batch-invariant kernels (119.65 / 29.33, `results_det/`). R5 therefore fired: our A0 replaces the paper's as the baseline and every ratio in this project is against ours, never against 5.1×.
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

### 2026-11-19 — executed vs declared geometry (before any VLC run)

`serve.py` passed only `speculative_num_draft_tokens = verify_columns - 1`, so `draft_width` was
never plumbed and every stored arm ran block == submitted. A1 and A3 are therefore one
configuration, so H4 is a repeat measurement of it: 0.998 with default kernels and 0.990 with
deterministic kernels, a noise floor of 0.2% and 1.0% respectively, and A0-vs-A3 is the paper's T8
diagonal at c=32, not a truncation test. The backend now refuses an inexpressible arm and records
`executed_draft_width`/`executed_verify_columns`.

H4-VLC has since been measured, by `experiments/standalone/run_accept.py` on 32 math500 prompts and
7079 rounds: acceptance 1.018 (95% CI [0.989, 1.049]) against a 1.05 bar and predicted tok/s 1.013
against 1.10. Both bars fail, verdict NOT SUPPORTED (RESULTS F15), and the nine-width sweep that
followed (RESULTS F17) says why: at a fixed verify length the drafter's width moves acceptance by
under 3% and not monotonically, while raising the window from 9 to 16 columns moves it 19.7%.
