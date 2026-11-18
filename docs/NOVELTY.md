# Novelty: what is defensibly new about verify-length control

Scope: the claim "truncating the verify window is a separate, and better, knob than shrinking the
draft block." Prior-work rows below were re-checked against the **arXiv abstract pages** on
2026-11-19 (not memory, not secondary summaries); the quoted sentence is the one each claim rests
on. Every number is from `docs/RESULTS.md` or from the DFlash paper. This file runs nothing.

## 1. What is not new

| Claim | Status | Source and the sentence it rests on |
|---|---|---|
| Verification cost grows with block size under serving batch; block reduction is a valid remedy | **NOT NEW** | DFlash, arXiv 2602.06036v2 (ICML 2026), §5.5.4: "large blocks can increase verification cost under compute-bound settings … reducing the block size in such cases can therefore yield better overall speedup. We leave adaptive block-size scheduling to future work." The problem statement, and the nominated fix, are the authors'. |
| Truncating the submitted draft before verification, at concurrency, to cut target cost | **NOT NEW — mechanism owned by AdaFlash** | AdaFlash, arXiv 2607.19223v3 (COLM'26 Workshop, Spotlight). Abstract: "an adaptive length head that dynamically adjusts the candidate length on the fly, substantially lowering the verification cost of the target model," and "especially significant gains under high-concurrency." It also ships a variable-length serving scheduler. Same mechanism, same target cost, swept at C ∈ {1,32,64,128}; that range is the abstract's, and no table in this repo verifies it, so it is carried here as a paper claim and not as a local result. |
| A training-free rule for *how many* drafted tokens to submit | **NOT NEW — rule owned by LibraSpec** | LibraSpec, arXiv 2608.08721v1. Abstract: "derive a marginal criterion that extends the speculative sequence only when its acceptance gain outweighs the additional verification cost," implemented as "a training-free and plug-and-play algorithm that iteratively determines the speculative length using drafter confidence scores." Per its full text it is already integrated with DFlash (`docs/RELATED.md`). |
| That the two knobs are *not* the same knob, tested against each other | **NOT CLAIMED BY EITHER, AND ONLY PARTLY VERIFIED** | Neither abstract mentions draft block width, and neither sets a width arm against its own truncation arm. **NOT VERIFIED beyond the pages read**: AdaFlash's appendices were read only where they touch DSpark (`docs/RELATED.md` §Provenance), so a width arm there would not have been seen. Treat "no width arm exists" as a working assumption, not a fact. |

Consequence: the **phenomenon** and the **mechanism** are both prior art. DFlash names the cost and
prescribes the wider knob; AdaFlash truncates at batch; LibraSpec supplies the selection rule. What is
left is not a method — it is one cell in the design space that no cited paper reports.

## 2. What is new

**One controlled cell, at equal verification work, that separates the two remedies: on a single frozen
DFlash block-16 drafter at c=32, drafting 16 and verifying only the first 9 columns versus shrinking
the draft block to 8 and verifying all 9, on byte-identical frozen prompts with the same τ ceiling.**

| | treatment | control |
|---|---|---|
| draft width γ | 16 | 8 (same b16 checkpoint, inference BS 8 — T8's generalizing direction) |
| verify columns k | **9** | **9** |
| τ ceiling | **9** | **9** |
| concurrency | 32 | 32 |
| prompts | frozen MATH-500, 32 (the executed run; 64 was the plan), shared byte-for-byte | identical |
| backend / kernel mode | one, named | identical |

Why the cell is not a relabelling of the paper's T8: block attention in DFlash is bidirectional, so
position *i* of a 16-block is computed with 16 positions in view while position *i* of an 8-block sees
8. The two arms therefore put **different tokens in the same 9 slots**. T8 (b16→b8) changes both knobs
at once — 17 columns to 9, *and* the proposal distribution — so it attributes the effect to neither.
The identifying cell is off the paper's diagonal; it does not exist in DFlash (there, τ ∈ [1, γ+1],
so columns are γ+1 by construction).

## 3. The honest weakness — stated before the run

**Pre-committed collapse condition.** The control is fixed in verify columns, once, and both bars use
it: **D8 = 8 drafts into 9 verify columns** (`draft_width` 8 / `verify_columns` 9,
`results/A3_math500_c32_s0.json`; in the standalone harness that is `--narrow 9`, tau cell `b9_k9` =
4.381). If `tok/s(γ=16, k=9) / tok/s(D8)` at c=32 lands within ±5% ([0.95, 1.05]), the contribution
collapses to exactly two things and no more: **(a) a replication** of AdaFlash's and LibraSpec's
already-published observation that shortening verification helps at batch, and **(b) a negative
result**: on a bidirectional-block drafter the two knobs are near-substitutes, so the block-size
remedy DFlash nominates and the truncation remedy we proposed buy the same thing. The report then says
that, in those words, and the title loses "better knob."

**The rule applied, so it cannot be read two ways.** Against D8 the measured tau ratio is **1.018**
(95% CI [0.989, 1.049]) with a cost factor of 0.995, hence a predicted tok/s ratio of **1.013**: inside
[0.95, 1.05], so the rule fires and the contribution collapses (RESULTS F15). Against a 7-draft /
8-column arm (`--narrow 8`) the tau ratio is **1.073** (CI [1.040, 1.104]), outside the band, so tau
alone would not trigger the collapse on that reading; that arm is not the registered control, because
it verifies one column fewer than VLC(16,9) and no engine arm corresponds to it. The registered reading
is the D8 one.

Three things make that outcome plausible rather than hypothetical, and none of them may be quietly
dropped:

1. **Our own pairs are already inside 5%.** The same-configuration A1/A3 pair (9 verify columns in
   both arms) reads **0.9980** with default kernels (`results/`) and **0.9903** with batch-invariant
   kernels (`results_det/`), both at 32 prompts. Those are block-k/verify-k pairs, not the treatment
   cell — a warning about the prior, not a result — but a reader will read them that way, so we state
   it first.
2. **The positive claim needs 1.10×, not 1.00×.** `docs/ABLATION-PROTOCOL.md` §3.4 fixes both bars
   against that same control, D8 = 8 drafts into 9 verify columns: `tok/s(VLC(16,9)) ≥ 1.10 ×
   tok/s(D8)` and `τ_9(16) ≥ 1.05 × τ_9(D8)`, where `τ_9(D8)` is the harness cell `b9_k9` = 4.381. A
   6% win fails.
3. **The treatment cell has run only in the standalone harness, never in the engine, and the public API
   cannot express it.** On SGLang 0.5.23 the block size and the verify length are one flag (RESULTS
   F5), so every engine "truncation" number is block-k/verify-k; the (16,9) cell is measured by
   `experiments/standalone/run_accept.py` (32 prompts, 7079 rounds, RESULTS F15). The result is
   therefore conditional on a patched draft worker, and must be labelled as such.

Even in the best case the honest title is "a second remedy for a cost the authors identified" — a
controlled comparison, not a new controller, and not a discovery of the batch collapse.

## 4. Positioning (paste-ready, two lines)

> DFlash (arXiv 2602.06036) names verification cost at serving batch and nominates block-size
> reduction as the remedy, and AdaFlash (arXiv 2607.19223) already truncates the submitted draft with a
> trained length head; neither tests the two remedies against each other.
> We hold verification work fixed at nine columns and τ ≤ 9 at c=32 and compare drafting 16 and
> verifying 9 against shrinking the block to 8 — isolating whether draft width and verify length are
> separable knobs, or one.

Provenance: abstracts of 2602.06036v2, 2607.19223v3, 2608.08721v1 retrieved 2026-11-19. Numbers from
`docs/RESULTS.md` (F1, F5, F9) and `docs/ABLATION-PROTOCOL.md` §1–§3.4. No experiment was run here.
