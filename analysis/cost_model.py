"""Cost model for the DFlash verify-length / draft-width decision.

Anchors are measured values from DFlash (arXiv 2602.06036). Two parameters are not
identified by the paper:

  rho    share of a batch-1 verification pass that is pure target-weight read
         (column-independent floor).
  w_tail share of blocks whose acceptance would exceed the truncated length, i.e.
         the right tail that truncation clips. Ignoring it overstates every gain.

rho is identified by the one published measurement of this mechanism (LibraSpec:
+15.6% over DFlash on Qwen3-8B at concurrency 1). `--fit` solves for it.

Run:      uv run python analysis/cost_model.py
Calibrate: uv run python analysis/cost_model.py --fit
Self-test: uv run python analysis/cost_model.py --selftest
"""

from __future__ import annotations

import math
import sys

# --- Measured anchors (DFlash 2602.06036v2) --------------------------------------
TAU_16 = 8.01        # Table 3, Qwen3-8B / MATH-500 / block 16, includes bonus token
S_C1 = 5.1           # Table 3, concurrency 1
S_C32 = 2.8          # Table 3, concurrency 32
TAU_8 = 5.09         # Table 8, b=16-trained checkpoint served at block 8
BLOCK = 16           # reference block width
ALPHA = 0.10         # drafter share of an AR step (inferred, bracket 0.05-0.20)

# --- Measured anchor for calibration (LibraSpec 2608.08721, abstract + Fig. 1) ---
LIBRASPEC_GAIN_C1 = 0.156   # 4.43x -> 5.12x, DFlash, Qwen3-8B, c=1

V_C1 = TAU_16 / S_C1 - ALPHA        # 1.471
V_C32 = TAU_16 / S_C32 - ALPHA      # 2.761
MAX_TAU = BLOCK + 1                 # 17


def v_verify(c: int, n_verify: int, rho: float) -> float:
    """Verification cost in units of one AR step.

    Split into a weight-read floor (independent of how many columns are verified)
    and a per-column part that also carries the batch-growth seen from c=1 to c=32.
    `n_verify` is the number of columns the target processes = tokens submitted + 1.
    """
    floor = rho * V_C1
    per_col_total = (1 - rho) * V_C1 + (V_C32 - V_C1) * math.log2(c) / 5.0
    return floor + (n_verify / MAX_TAU) * per_col_total


def tau_of(n_verify: int, w_tail: float = 0.0, tau_block: float = TAU_16) -> float:
    """Acceptance after truncating verification at `n_verify` columns.

    Two groups of blocks: a fraction `w_tail` would accept the full block (17), the
    rest accept `tau_lo`, chosen so the untruncated mean equals the measured tau.
    Truncation clips both groups at the cap, which is exactly the cost of truncating.
    """
    cap = min(n_verify, MAX_TAU)
    if tau_block != TAU_16:
        # A narrower drafter cannot exceed its own block; no tail above the cap.
        return min(tau_block, cap)
    if w_tail <= 0:
        return min(TAU_16, cap)
    if w_tail * MAX_TAU >= TAU_16:
        raise ValueError(f"w_tail={w_tail} implies a negative lower mean")
    tau_lo = (TAU_16 - w_tail * MAX_TAU) / (1 - w_tail)
    return (1 - w_tail) * min(tau_lo, cap) + w_tail * min(MAX_TAU, cap)


def speedup(c: int, n_verify: int, rho: float, w_tail: float = 0.0,
            tau_block: float = TAU_16) -> float:
    return tau_of(n_verify, w_tail, tau_block) / (ALPHA + v_verify(c, n_verify, rho))


def gain_pct(c: int, n_verify: int, rho: float, w_tail: float = 0.0,
             tau_block: float = TAU_16) -> float:
    base = speedup(c, MAX_TAU, rho, w_tail, tau_block)
    return (speedup(c, n_verify, rho, w_tail, tau_block) / base - 1) * 100


def fit_rho(target_gain: float, w_tail: float, lo: float = 0.0, hi: float = 0.95) -> float:
    """Solve for the memory-floor share that reproduces a measured c=1 gain."""
    def f(r: float) -> float:
        return gain_pct(1, 9, r, w_tail) / 100 - target_gain

    if f(lo) < 0 or f(hi) > 0:
        raise ValueError(f"gain {target_gain:.3f} not reachable for w_tail={w_tail}")
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def selftest() -> None:
    for rho in (0.3, 0.5, 0.7, 0.9):
        for w in (0.0, 0.1, 0.25):
            assert abs(v_verify(1, MAX_TAU, rho) - V_C1) < 1e-9
            assert abs(v_verify(32, MAX_TAU, rho) - V_C32) < 1e-9
            assert abs(tau_of(MAX_TAU, w) - TAU_16) < 1e-9
    assert abs(speedup(1, MAX_TAU, 0.7) - S_C1) < 1e-6
    assert abs(speedup(32, MAX_TAU, 0.7) - S_C32) < 1e-6
    assert tau_of(9, 0.0) == TAU_16 and tau_of(5, 0.0) == 5
    assert tau_of(9, 0.1) < TAU_16              # the tail is clipped
    assert tau_of(9, 0.25) < tau_of(9, 0.1)     # more tail, more loss
    print("selftest ok")


def width_gain_pct(c: int, rho: float) -> float:
    """Block-8 arm against the DFlash baseline. The baseline is always the full
    block (tau=8.01, 17 columns); only the arm changes width."""
    base = speedup(c, MAX_TAU, rho, 0.0, tau_block=TAU_16)
    narrow = speedup(c, 9, rho, 0.0, tau_block=TAU_8)
    return (narrow / base - 1) * 100


def table_truncation() -> None:
    print("Table 1. Gain from truncating verification to 9 columns (L=8), by (rho, w_tail)")
    print("         w_tail = share of blocks whose acceptance exceeds the cap")
    print(f"{'rho':>5} {'w':>5} {'tau(9)':>7} {'c=1':>8} {'c=8':>8} {'c=32':>8}")
    for rho in (0.3, 0.5, 0.7, 0.9):
        for w in (0.0, 0.05, 0.1, 0.25):
            print(f"{rho:>5} {w:>5} {tau_of(9, w):>7.2f} {gain_pct(1, 9, rho, w):>7.1f}% "
                  f"{gain_pct(8, 9, rho, w):>7.1f}% {gain_pct(32, 9, rho, w):>7.1f}%")
        print()


def table_length_sweep() -> None:
    print("Table 2. Where the cap should sit, given rho=0.5, w_tail=0.1")
    print("         The paper does not identify this; the A2 sweep must measure it.")
    print(f"{'cols':>5} {'tau':>6} {'c=1':>7} {'c=8':>7} {'c=32':>7}")
    for n in range(5, MAX_TAU + 1):
        print(f"{n:>5} {tau_of(n, 0.1):>6.2f} {gain_pct(1, n, 0.5, 0.1):>6.1f}% "
              f"{gain_pct(8, n, 0.5, 0.1):>6.1f}% {gain_pct(32, n, 0.5, 0.1):>6.1f}%")
    print()


def table_width() -> None:
    print("Table 3. Width arm: serving the b=16 checkpoint at block 8 (tau=5.09)")
    print(f"{'rho':>5} {'c':>4} {'S(b=16)':>8} {'S(b=8)':>7} {'gain%':>7}")
    for rho in (0.3, 0.5, 0.7, 0.9):
        for c in (1, 8, 32):
            narrow = speedup(c, 9, rho, 0.0, tau_block=TAU_8)
            print(f"{rho:>5} {c:>4} {speedup(c, MAX_TAU, rho):>8.2f} {narrow:>7.2f} "
                  f"{width_gain_pct(c, rho):>7.1f}")
        print()


def table_calibrated() -> None:
    print("Table 4. Calibrated projection: rho fitted to LibraSpec's measured +15.6% at c=1")
    print(f"{'w_tail':>6} {'rho*':>6} {'c=1':>8} {'c=8':>8} {'c=32':>8} {'width c=32':>11}")
    for w in (0.0, 0.05, 0.1, 0.25):
        r = fit_rho(LIBRASPEC_GAIN_C1, w)
        print(f"{w:>6} {r:>6.2f} {gain_pct(1, 9, r, w):>7.1f}% {gain_pct(8, 9, r, w):>7.1f}% "
              f"{gain_pct(32, 9, r, w):>7.1f}% {width_gain_pct(32, r):>10.1f}%")
    r = fit_rho(LIBRASPEC_GAIN_C1, 0.1)
    print(f"\nHeadline (w_tail=0.10, rho*={r:.2f}):")
    for c in (1, 4, 8, 16, 32):
        print(f"  c={c:<3} truncate-to-9 {gain_pct(c, 9, r, 0.1):+6.1f}%   "
              f"block-8 {width_gain_pct(c, r):+6.1f}%")


def main() -> None:
    print(f"anchors: tau={TAU_16} S(c=1)={S_C1} S(c=32)={S_C32} -> "
          f"v(1)={V_C1:.3f} v(32)={V_C32:.3f}\n")
    table_truncation()
    table_length_sweep()
    table_width()
    table_calibrated()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--fit" in sys.argv:
        table_calibrated()
    else:
        main()
