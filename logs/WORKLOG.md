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
