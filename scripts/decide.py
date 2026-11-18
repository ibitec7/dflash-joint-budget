"""Apply the pre-registered decision rules to measured results.

Usage:
    uv run python scripts/decide.py --results results --benchmark math500
    uv run python scripts/decide.py --selftest

The cost model makes predictions; this script decides. It refuses to produce a verdict when a
required cell is missing, because a missing measurement is not a negative result.
Exit codes: 0 verdict produced, 1 incomplete, 2 invalid run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import cost_model as cm  # pyright: ignore[reportMissingImports]

REQUIRED = {
    "arm", "benchmark", "concurrency", "seed", "draft_width", "verify_columns", "prompts",
    "tau", "tok_per_s", "ar_tok_per_s", "ms_per_round_p50", "ms_per_round_p95",
    "exact_match_ar", "graph_recaptures",
}

# Pre-registered in docs/EXPERIMENTS.md §2. Do not move these after a run.
THRESHOLDS = {
    "H1_ms_ratio_max": 0.85,     # ms/round at 9 vs 17 columns, c=32
    "H2_tau_min": 7.2,           # tau at 9 columns, c=32
    "H3_gain_min": 1.15,         # tok/s vs DFlash, c=32
    "H4_dominance_min": 1.10,    # tok/s vs block-8, c=32
    "H5_c1_cost_max": 0.98,      # tok/s relative, c=1
    "H6_match_min": 0.99,        # greedy agreement with AR
    "H7_adaptive_min": 1.05,     # tok/s vs best fixed length, c=32
}


def mean(runs: list[dict], key: str) -> float:
    vals = [r[key] for r in runs]
    return sum(vals) / len(vals)


def cell(runs: list[dict], arm: str, c: int) -> list[dict]:
    return [r for r in runs if r["arm"] == arm and r["concurrency"] == c]


def ratio(runs: list[dict], arm: str, base: str, c: int, key: str) -> float | None:
    a, b = cell(runs, arm, c), cell(runs, base, c)
    if not a or not b:
        return None
    return mean(a, key) / mean(b, key)


def fit_w_tail(tau_measured: float, n_verify: int = 9, lo: float = 0.0,
               hi: float = 0.46) -> float | None:
    """Invert the model's tau(L) for the clipped-tail share. None if out of reach."""
    if tau_measured >= cm.TAU_16:
        return 0.0
    if cm.tau_of(n_verify, hi) > tau_measured:
        return None
    if cm.tau_of(n_verify, lo) < tau_measured:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if cm.tau_of(n_verify, mid) > tau_measured:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def evaluate(runs: list[dict]) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (verdict, [(id, PASS|FAIL|MISSING, detail)])."""
    out: list[tuple[str, str, str]] = []

    def add(hid: str, ok: bool | None, detail: str) -> None:
        out.append((hid, "MISSING" if ok is None else ("PASS" if ok else "FAIL"), detail))

    a1 = cell(runs, "A1", 32)
    a4 = cell(runs, "A4", 32)

    ms_ratio = ratio(runs, "A1", "A0", 32, "ms_per_round_p50")
    add("H1", None if ms_ratio is None else ms_ratio <= THRESHOLDS["H1_ms_ratio_max"],
        f"ms/round(A1)/ms/round(A0) = {ms_ratio:.3f}" if ms_ratio else "missing A0/A1 at c=32")

    tau_a1 = mean(a1, "tau") if a1 else None
    add("H2", None if tau_a1 is None else tau_a1 >= THRESHOLDS["H2_tau_min"],
        f"tau(A1) = {tau_a1:.2f} (min {THRESHOLDS['H2_tau_min']})" if tau_a1 else "missing A1 c=32")

    gain = ratio(runs, "A1", "A0", 32, "tok_per_s")
    add("H3", None if gain is None else gain >= THRESHOLDS["H3_gain_min"],
        f"tok/s ratio = {gain:.3f}" if gain else "missing A0/A1 at c=32")

    dom = ratio(runs, "A1", "A3", 32, "tok_per_s")
    add("H4", None if dom is None else dom >= THRESHOLDS["H4_dominance_min"],
        f"tok/s(A1)/tok/s(A3) = {dom:.3f}" if dom else "missing A3 at c=32")

    cost_c1 = ratio(runs, "A1", "A0", 1, "tok_per_s")
    add("H5", None if cost_c1 is None else cost_c1 >= THRESHOLDS["H5_c1_cost_max"],
        f"tok/s ratio at c=1 = {cost_c1:.3f}" if cost_c1 else "missing A0/A1 at c=1")

    if not a1:
        add("H6", None, "missing A1")
    else:
        worst = min(r["exact_match_ar"] for r in a1)
        add("H6", worst >= THRESHOLDS["H6_match_min"], f"min greedy agreement = {worst:.4f}")

    if not a4:
        add("H7", None, "missing A4 at c=32")
    else:
        best_fixed = max(mean(cell(runs, arm, 32), "tok_per_s")
                         for arm in ("A1", "A2") if cell(runs, arm, 32))
        r = mean(a4, "tok_per_s") / best_fixed
        add("H7", r >= THRESHOLDS["H7_adaptive_min"], f"tok/s(A4)/best fixed = {r:.3f}")

    got = {h: v for h, v, _ in out}
    required = ("H1", "H2", "H3", "H4", "H5", "H6")
    if any(got[h] == "MISSING" for h in required):
        return "INCOMPLETE", out
    if got["H6"] == "FAIL":
        return "INVALID RUN", out
    if got["H1"] == "FAIL":
        return "NEGATIVE RESULT", out
    if got["H2"] == "FAIL" or got["H3"] == "FAIL":
        return "MECHANISM ONLY", out
    if got["H4"] == "PASS":
        return "POSITIVE", out
    return "POSITIVE, WEAKER CLAIM", out


def report_fitted(runs: list[dict]) -> list[str]:
    """Compare the model's two predictions against what was measured."""
    lines = []
    ms_ratio = ratio(runs, "A1", "A0", 32, "ms_per_round_p50")
    if ms_ratio is not None:
        col_share = 1 - ms_ratio
        rho_hat = max(0.0, 1 - col_share * (cm.MAX_TAU / (cm.MAX_TAU - 9)))
        lines.append(f"rho:      predicted {0.50:.2f}   measured {rho_hat:.2f}  "
                     f"(column-scalable share of the round {col_share:.1%})")
    a1 = cell(runs, "A1", 32)
    if a1:
        tau_meas = mean(a1, "tau")
        w = fit_w_tail(tau_meas)
        got = f"{w:.2f}" if w is not None else "outside the model's reach"
        lines.append(f"w_tail:   predicted {0.10:.2f}   measured {got}  (tau at 9 columns {tau_meas:.2f})")
    return lines


def load(results_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        data = json.loads(p.read_text())
        missing = REQUIRED - data.keys()
        if missing:
            raise ValueError(f"{p.name}: missing keys {sorted(missing)}")
        runs.append(data)
    return runs


def selftest() -> None:
    base = {"benchmark": "math500", "seed": 0, "prompts": 200, "ar_tok_per_s": 110.0,
            "graph_recaptures": 0, "exact_match_ar": 0.997, "ms_per_round_p95": 2.3,
            "draft_width": 16, "verify_columns": 9}

    def run(arm, c, tau, tok, ms, **kw):
        return dict(base, arm=arm, concurrency=c, tau=tau, tok_per_s=tok,
                    ms_per_round_p50=ms, **kw)

    good = [run("A0", 32, 8.01, 110.0, 2.50), run("A1", 32, 7.21, 150.0, 1.90),
            run("A3", 32, 5.09, 105.0, 2.10), run("A4", 32, 7.30, 155.0, 1.85),
            run("A0", 1, 8.01, 112.0, 4.30), run("A1", 1, 7.21, 130.0, 3.90)]
    verdict, rows = evaluate(good)
    assert verdict == "POSITIVE", (verdict, rows)

    bad_ms = [run("A0", 32, 8.01, 110.0, 2.50), run("A1", 32, 8.01, 111.0, 2.48),
              run("A3", 32, 5.09, 105.0, 2.10), run("A0", 1, 8.01, 112.0, 4.30),
              run("A1", 1, 8.01, 112.0, 4.29)]
    assert evaluate(bad_ms)[0] == "NEGATIVE RESULT"

    leaky = [dict(r, exact_match_ar=0.95) for r in good]
    assert evaluate(leaky)[0] == "INVALID RUN"

    assert evaluate(good[:2])[0] == "INCOMPLETE"
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    results_dir = Path(args.results)
    if (results_dir / "SYNTHETIC_DATA_DO_NOT_USE").exists():
        print(f"refusing {results_dir}: it carries SYNTHETIC_DATA_DO_NOT_USE "
              "(dry-run plumbing records, not measurements)")
        return 3

    runs = [r for r in load(results_dir) if r["benchmark"] == args.benchmark]
    if not runs:
        print(f"no results for {args.benchmark} in {args.results}")
        return 1

    verdict, rows = evaluate(runs)
    print(f"\nverdict: {verdict}\n")
    for hid, status, detail in rows:
        print(f"  {hid}  {status:<7} {detail}")
    fitted = report_fitted(runs)
    if fitted:
        print("\nmodel vs measurement:")
        for line in fitted:
            print(f"  {line}")
    return {"INCOMPLETE": 1, "INVALID RUN": 2}.get(verdict, 0)


if __name__ == "__main__":
    raise SystemExit(main())
