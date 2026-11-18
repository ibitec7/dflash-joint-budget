"""Cost model for the DFlash joint draft-width / verify-length budget.

Anchors are measured values from DFlash (arXiv 2602.06036), page-cited. Everything
else is derived arithmetic, printed as a range over the unknown split parameter
`rho` (share of a batch-1 verification pass that is pure target-weight read).

Run: uv run python analysis/cost_model.py
Self-test: uv run python analysis/cost_model.py --selftest
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

# --- Measured anchors (DFlash 2602.06036v2) --------------------------------------
TAU_16 = 8.01        # Table 3, Qwen3-8B / MATH-500 / block 16, includes bonus token
S_C1 = 5.1           # Table 3, concurrency 1
S_C32 = 2.8          # Table 3, concurrency 32
TAU_8 = 5.09         # Table 8, b=16-trained checkpoint served at block 8
BLOCK = 16           # reference block width
ALPHA = 0.10         # drafter share of an AR step (inferred, bracket 0.05-0.20)

V_C1 = TAU_16 / S_C1 - ALPHA        # 1.471
V_C32 = TAU_16 / S_C32 - ALPHA      # 2.761


def v_verify(c: int, n_verify: int, rho: float) -> float:
    """Verification cost in units of one AR step.

    Split into a weight-read floor (independent of how many columns are verified)
    and a per-column part that also carries the batch-growth seen from c=1 to c=32.
    `n_verify` is the number of columns the target processes = tokens submitted + 1.
    """
    floor = rho * V_C1
    per_col_total = (1 - rho) * V_C1 + (V_C32 - V_C1) * math.log2(c) / 5.0
    return floor + (n_verify / (BLOCK + 1)) * per_col_total


def tau_of(n_verify: int, tau_block: float = TAU_16) -> float:
    """Acceptance after truncating verification at n_verify columns.

    Upper bound: truncation cannot accept more than it verifies. It also assumes
    acceptance is untouched until the cap binds, which is optimistic by construction.
    """
    return min(tau_block, n_verify)


def speedup(c: int, n_verify: int, rho: float, tau_block: float = TAU_16) -> float:
    return tau_of(n_verify, tau_block) / (ALPHA + v_verify(c, n_verify, rho))


def best_length(c: int, rho: float, lo: int = 3, hi: int = BLOCK + 1) -> tuple[int, float]:
    grid = [(n, speedup(c, n, rho)) for n in range(lo, hi + 1)]
    return max(grid, key=lambda t: t[1])


def selftest() -> None:
    for rho in (0.3, 0.5, 0.7, 0.9):
        assert abs(v_verify(1, BLOCK + 1, rho) - V_C1) < 1e-9, rho
        assert abs(v_verify(32, BLOCK + 1, rho) - V_C32) < 1e-9, rho
    assert abs(speedup(1, BLOCK + 1, 0.7) - S_C1) < 1e-6
    assert abs(speedup(32, BLOCK + 1, 0.7) - S_C32) < 1e-6
    assert tau_of(9) == TAU_16 and tau_of(5) == 5
    print("selftest ok")


@dataclass
class Row:
    c: int
    rho: float
    n_star: int
    s_star: float
    s_stale: float

    @property
    def gain_pct(self) -> float:
        return (self.s_star / self.s_stale - 1) * 100


def main() -> None:
    print(f"anchors: tau={TAU_16} S(c=1)={S_C1} S(c=32)={S_C32} -> v(1)={V_C1:.3f} v(32)={V_C32:.3f}\n")

    print("Table 1. Optimal verify length vs concurrency (rho = memory-floor share)")
    print(f"{'rho':>5} {'c':>4} {'n*':>3} {'tau':>6} {'S*':>6} {'S@16':>6} {'gain%':>7}")
    for rho in (0.3, 0.5, 0.7, 0.9):
        for c in (1, 4, 8, 16, 32):
            n, s = best_length(c, rho)
            print(f"{rho:>5} {c:>4} {n:>3} {tau_of(n):>6.2f} {s:>6.2f} {speedup(c, 17, rho):>6.2f} "
                  f"{(s / speedup(c, 17, rho) - 1) * 100:>7.1f}")
        print()

    print("Table 2. The rule does not move with load; its benefit does")
    print("         (same optimal length at every c, because the tau cap binds first.")
    print("          What grows with load is the size of the saving, not the choice of n.)")
    print(f"{'rho':>5} {'n*':>3} {'c=1':>7} {'c=4':>7} {'c=8':>7} {'c=16':>7} {'c=32':>7}")
    for rho in (0.3, 0.5, 0.7, 0.9):
        n1, _ = best_length(1, rho)
        cells = []
        for c in (1, 4, 8, 16, 32):
            n, s = best_length(c, rho)
            assert n == n1, (rho, c, n, n1)
            cells.append((s / speedup(c, 17, rho) - 1) * 100)
        print(f"{rho:>5} {n1:>3} " + " ".join(f"{x:>6.1f}%" for x in cells))
    print()

    print("Table 3. Width arm only: serving the b=16 checkpoint at block 8 (tau=5.09)")
    print(f"{'rho':>5} {'c':>4} {'S(b=16)':>8} {'S(b=8)':>7} {'gain%':>7}")
    for rho in (0.3, 0.5, 0.7, 0.9):
        for c in (1, 8, 32):
            base = speedup(c, 17, rho)
            narrow = tau_of(9, tau_block=TAU_8) / (ALPHA + v_verify(c, 9, rho))
            print(f"{rho:>5} {c:>4} {base:>8.2f} {narrow:>7.2f} {(narrow / base - 1) * 100:>7.1f}")
        print()

    lo = (speedup(1, best_length(1, 0.9)[0], 0.9) / S_C1 - 1) * 100
    hi = (speedup(1, best_length(1, 0.3)[0], 0.3) / S_C1 - 1) * 100
    print("Calibration: LibraSpec measures +15.6% over DFlash (4.43x -> 5.12x, Qwen3-8B).")
    print(f"Model at c=1 gives +{lo:.1f}% (rho=0.9) to +{hi:.1f}% (rho=0.3) for the length arm,")
    print("so the model is optimistic by roughly 2x on the measured point. Treat every")
    print("number above as an upper bound, not a prediction.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
