"""Ablation tables for the verify-width and block-width arms (stdlib only).

Reads measured result JSONs and prints four tables. Nothing is modelled here: a cell is a
mean over seeds of a measured record, or the word "absent".

Run:  uv run python scripts/ablate.py --results results_det --benchmark math500
Test: uv run python scripts/ablate.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
from pathlib import Path

KS = (4, 6, 9, 12, 16)
BS = (6, 8, 12, 16)
AR = "A5"
BASE = "A0"
THRESHOLD = 1.10
ABSENT = "absent"
REGISTER = ["A0", "A1", "A2", "A3", "A4", "A5"]
V_RE = re.compile(r"^V(\d+)k(\d+)$")
B_RE = re.compile(r"^B(\d+)$")
REQUIRED = ("arm", "concurrency", "tau", "tok_per_s")


# ---------------------------------------------------------------- records / aggregation


def load_records(directory: Path, benchmark: str) -> list[dict]:
    """Load usable records; an unreadable one is reported by name, not swallowed."""
    records = []
    for path in sorted(directory.glob("*.json")):
        rec = json.loads(path.read_text())
        if benchmark and rec.get("benchmark") not in (None, benchmark):
            continue
        missing = [key for key in REQUIRED if key not in rec]
        if missing:
            print(f"NOTE skipped {path.name}: missing {', '.join(missing)}; "
                  f"has {', '.join(sorted(rec))}")
            continue
        records.append(rec)
    return records


def aggregate(records: list[dict]) -> dict:
    """(arm, concurrency) -> mean tau, tok/s, ms_per_round_p50, exact_match_ar over seeds."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for rec in records:
        groups.setdefault((str(rec["arm"]), int(rec["concurrency"])), []).append(rec)
    agg = {}
    for key, recs in groups.items():
        def mean(field: str):
            vals = [float(r[field]) for r in recs if r.get(field) is not None]
            return statistics.fmean(vals) if vals else None

        agg[key] = {"n": len(recs), "tau": mean("tau"), "tok": mean("tok_per_s"),
                    "ms": mean("ms_per_round_p50"), "exact": mean("exact_match_ar")}
    return agg


def get(agg: dict, arm: str, c: int, field: str):
    entry = agg.get((arm, c))
    return None if entry is None else entry[field]


def ratio(agg: dict, num: str, den: str, c: int, field: str):
    a, b = get(agg, num, c, field), get(agg, den, c, field)
    if a is None or b is None or not b:
        return None
    return a / b


def fmt(value, spec: str) -> str:
    return ABSENT if value is None else format(value, spec)


def width_cell(agg: dict, arm: str, c: int) -> str:
    tok, tau = get(agg, arm, c, "tok"), get(agg, arm, c, "tau")
    return ABSENT if tok is None or tau is None else f"{tok:.0f}/{tau:.2f}"


def speed_cell(agg: dict, arm: str, c: int) -> str:
    return (f"{fmt(ratio(agg, arm, AR, c, 'tok'), '.2f')}/"
            f"{fmt(ratio(agg, arm, BASE, c, 'tok'), '.2f')}")


# ---------------------------------------------------------------- tables


def render(title: str, legend: str, rows: list[tuple[str, str]], agg: dict,
           cs: list[int], cell) -> None:
    print(f"\n{title}")
    print(f"  {legend}")
    print(f"{'arm':<8}" + "".join(f"{f'c={c}':>17}" for c in cs))
    for label, arm in rows:
        print(f"{label:<8}" + "".join(f"{cell(agg, arm, c):>17}" for c in cs))


def breakeven(agg: dict, c: int, verbose: bool = True) -> list[dict]:
    """tau_k(V16k<k>)/tau_k(B<k>) against measured per-round cost and the 1.10 rule."""
    if verbose:
        print(f"\n[4] break-even at c={c} — VLC(k) vs block-k; WIN needs tok/s ratio > "
              f"{THRESHOLD:.2f}")
        print("  tau ratio = tau(V16k<k>)/tau(B<k>); cost ratio = ms_per_round_p50(V)/"
              "ms_per_round_p50(B)")
        print(f"{'k':<4}{'tau_V':>9}{'tau_B':>9}{'tau_ratio':>11}{'cost_ratio':>12}"
              f"{'tau>cost':>10}{'tok_ratio':>11}   verdict")
    rows = []
    for k in KS:
        v_arm, b_arm = f"V16k{k}", f"B{k}"
        tau_ratio = ratio(agg, v_arm, b_arm, c, "tau")
        cost_ratio = ratio(agg, v_arm, b_arm, c, "ms")
        tok_ratio = ratio(agg, v_arm, b_arm, c, "tok")
        if tok_ratio is None:
            verdict = "ABSENT"
        else:
            verdict = "WIN" if tok_ratio > THRESHOLD else "LOSE"
        pays = None if tau_ratio is None or cost_ratio is None else tau_ratio > cost_ratio
        row = {"k": k, "tau_ratio": tau_ratio, "cost_ratio": cost_ratio,
               "tok_ratio": tok_ratio, "verdict": verdict, "pays": pays,
               "tau_v": get(agg, v_arm, c, "tau"), "tau_b": get(agg, b_arm, c, "tau")}
        rows.append(row)
        if verbose:
            print(f"{k:<4}{fmt(row['tau_v'], '.3f'):>9}{fmt(row['tau_b'], '.3f'):>9}"
                  f"{fmt(tau_ratio, '.3f'):>11}{fmt(cost_ratio, '.3f'):>12}"
                  f"{'absent' if pays is None else ('yes' if pays else 'no'):>10}"
                  f"{fmt(tok_ratio, '.3f'):>11}   {verdict}")
    if verbose:
        print()
        for row in rows:
            k = row["k"]
            if row["verdict"] == "ABSENT":
                print(f"k={k:<3} ABSENT — V16k{k} or B{k} has no record at c={c}; no verdict")
            else:
                print(f"k={k:<3} {row['verdict']} — tok/s {row['tok_ratio']:.3f} "
                      f"(threshold {THRESHOLD:.2f})")
    return rows


def report(agg: dict, directory: Path, benchmark: str) -> None:
    present = sorted({arm for arm, _ in agg})
    cs = sorted({c for _, c in agg})
    headline = 32 if 32 in cs else (cs[-1] if cs else None)
    register = REGISTER + [f"V16k{k}" for k in KS] + [f"B{b}" for b in BS]
    missing = [arm for arm in register if arm not in present]

    print(f"== ablate.py — results={directory} benchmark={benchmark or 'any'}")
    print("nothing is projected: every cell is a measured record's mean over seeds, "
          "or 'absent'.")
    print(f"arms present: {', '.join(present) or 'none'}")
    print(f"arms absent : {', '.join(missing) or 'none'}")
    print(f"concurrency levels: {', '.join(str(c) for c in cs) or 'none'}")
    if headline is not None:
        exact = [f"{arm}={fmt(get(agg, arm, headline, 'exact'), '.3f')}" for arm in present]
        print(f"exact_match_ar at c={headline}: {', '.join(exact)}")

    render("[1] verify width — V16k<k>, cell = tok/s / tau",
           "rows = verify width k; fixed drafter width is not inferred from these cells",
           [(f"V16k{k}", f"V16k{k}") for k in KS], agg, cs, width_cell)
    render("[2] block width — B<b>, cell = tok/s / tau", "rows = block width b, verified in full",
           [(f"B{b}", f"B{b}") for b in BS], agg, cs, width_cell)
    render(f"[3] speedup, cell = tok/s(arm)/tok/s({AR}) / tok/s(arm)/tok/s({BASE})",
           "first number is the speedup over the autoregressive reference, second over the "
           "DFlash baseline; a self-comparison reads 1.00",
           [(arm, arm) for arm in present], agg, cs, speed_cell)

    if headline is None:
        print("\n[4] break-even — no records, nothing to compare")
    else:
        breakeven(agg, headline)


# ---------------------------------------------------------------- selftest


def selftest() -> None:
    records = [
        {"arm": "A0", "benchmark": "math500", "concurrency": 32, "tau": 5.50,
         "tok_per_s": 2000.0, "ms_per_round_p50": 44.0, "exact_match_ar": 1.0},
        {"arm": "A5", "benchmark": "math500", "concurrency": 32, "tau": 1.00,
         "tok_per_s": 1000.0, "ms_per_round_p50": 16.0, "exact_match_ar": 1.0},
        {"arm": "V16k9", "benchmark": "math500", "concurrency": 32, "tau": 4.40,
         "tok_per_s": 2400.0, "ms_per_round_p50": 36.0, "exact_match_ar": 1.0},
        {"arm": "B9", "benchmark": "math500", "concurrency": 32, "tau": 4.00,
         "tok_per_s": 2000.0, "ms_per_round_p50": 44.0, "exact_match_ar": 1.0},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for rec in records:
            path = Path(tmp, f"{rec['arm']}_math500_c{rec['concurrency']}_s0.json")
            path.write_text(json.dumps(rec))
        agg = aggregate(load_records(Path(tmp), "math500"))
    assert len(agg) == 4, agg
    assert agg[("V16k9", 32)]["tok"] == 2400.0
    assert agg[("B9", 32)]["tau"] == 4.00
    assert width_cell(agg, "V16k9", 32) == "2400/4.40"
    assert width_cell(agg, "V16k4", 32) == ABSENT, width_cell(agg, "V16k4", 32)
    assert speed_cell(agg, "V16k9", 32) == "2.40/1.20", speed_cell(agg, "V16k9", 32)
    assert speed_cell(agg, "A5", 32) == "1.00/0.50", speed_cell(agg, "A5", 32)
    assert speed_cell(agg, "B6", 32) == "absent/absent"

    rows = breakeven(agg, 32, verbose=False)
    assert [row["k"] for row in rows] == list(KS)
    win = rows[2]
    assert abs(win["tau_ratio"] - 1.10) < 1e-9 and abs(win["cost_ratio"] - 36 / 44) < 1e-9
    assert abs(win["tok_ratio"] - 1.20) < 1e-9 and win["pays"] is True
    assert win["verdict"] == "WIN", win
    assert rows[0]["verdict"] == "ABSENT" and rows[0]["tau_ratio"] is None

    lose_agg = {("V16k9", 32): {"n": 1, "tau": 4.4, "tok": 2100.0, "ms": 36.0, "exact": 1.0},
                ("B9", 32): {"n": 1, "tau": 4.0, "tok": 2000.0, "ms": 44.0, "exact": 1.0}}
    assert breakeven(lose_agg, 32, verbose=False)[2]["verdict"] == "LOSE"
    assert breakeven(lose_agg, 32, verbose=False)[2]["tok_ratio"] == 1.05
    print("SELFTEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--benchmark", default="math500")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    report(aggregate(load_records(args.results, args.benchmark)), args.results, args.benchmark)


if __name__ == "__main__":
    main()
