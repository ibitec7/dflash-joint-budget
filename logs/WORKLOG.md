## 2026-11-18 (UTC) — the acceptance instrument, and the first number out of it

commits: e3ba3c6, 8395501, c30b598, 1217f5b, d93b9f9, b155a86, 18fd968, 0196266

- what: three defects in run_accept.py, each found by running it. (1) the reference loop overshot
  the token budget by one. (2) the gate could pass while measuring nothing: at tau 1.03 both loops
  are plain greedy decoding and agree whatever the drafter does, so the gate now refuses a
  reference under 0.6x the engine's tau. (3) two implementations of the same loop: the separate
  draft/verify path reported tau 1.00 on every prompt while the reference loop accepted 3.73
  tokens per round on the same weights, an off-by-one in the verify positions. One validated loop
  now does both the gate and the measurement and records per-round draft and verify times.
- worked: the loop reproduces the checkpoint's own spec_generate token for token on both prompt
  conventions (raw and chat), and the first measurement gives tau_9(16)=3.948 against
  tau_9(9)=3.901, ratio 1.012, inside the pre-committed collapse band.
- partial: F14 is n=8 and has no CI. The harness sits 24% below the engine's absolute tau at block
  16 (4.45 against 5.861) because it decodes from its own greedy contexts and its widest block is
  one column narrower than the engine's; only the ratio travels, and the 32-prompt run decides.
- failed and why: E10 ran the whole benchmark on the off-by-one path and produced tau 1.03 for
  every block. The numbers were thrown away rather than explained away, and the gate that would
  have caught it did not exist yet.
- also: the A1/A0 ratio was being quoted as 1.409 from the two seeds that agree; s0 is 1.498 and
  the 64-prompt run 1.364, so the docs now carry the range 1.32-1.50.

## 2026-11-18 evening (UTC) — the surface, the composition, and the audit

commits: 72925a0, 1bfa599, 24ca5fb, 7d5da11

- what: two sweeps (nine drafter widths x every verify window, math500 and mtbench, 15594 and 21735
  rounds), a composition script that multiplies acceptance by the engine's round cost with a
  bootstrap CI, four figures, 13 tests, and an independent numeric audit of every claim in the four
  docs.
- worked: acceptance is a property of the verify window, not the drafter's width. At a fixed nine
  columns the nine blocks span under 3% on math500 (4.284 to 4.409, non-monotone) and 5.2% on
  mtbench, while raising the window from 9 to 16 columns at b16 is +19.7%. Composed over the
  shipped block-16 arm: 1.345 math500, 1.344 mtbench, CI [1.291, 1.406], acceptance factor 0.80 and
  cost factor 1.674.
- partial: the composition is not end-to-end. The driver indexes the commit chain with one
  block_size stride, allocates every per-round buffer at bs*block_size, and captures the draft path
  in a CUDA graph, so draft 16 / verify 9 has no expressible shape (RESULTS F18, with file:line).
  The model is calibrated against the engine: 0.7% error on default kernels, 14% under graphs, so
  the defended interval is 1.18x to 1.41x.
- failed and why: the first composition priced the wide arm by scaling its 16-column verify down to
  9 with a proportional model, inventing a 39% saving and reporting a cost factor of 1.596 that
  carried the throughput bar on its own. Measured flatness at batch 1 is 0.8%, so the model was
  replaced by the flat one and the direction of the cost factor corrected (control over treatment).
- also: the audit found 25 defects, including a control-arm substitution that would have flipped the
  tau bar (it was a unit confusion between verify columns and draft width, but vlc.py defaulted to
  the wrong arm and now does not), mode labels swapped on the 0.998/0.990 pair, a cross-k ratio
  quoted as equal-k, and H7 printed as NA where decide.py prints PASS. results_s, results_p64 and
  the n=8 acceptance pass are now in the repo, so the 1.32-1.50 seed spread is sourced.
