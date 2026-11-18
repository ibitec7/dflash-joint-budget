"""Ablation tables from measured result JSONs (stdlib only); nothing is modelled.

Run: uv run python scripts/ablate.py --results results_det --benchmark math500
     uv run python scripts/ablate.py --selftest
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

KS = (4, 6, 9, 12, 16)
BS = (6, 8, 12, 16)
AR, BASE, THRESHOLD, ABSENT = "A5", "A0", 1.10, "absent"
REQUIRED = ("arm", "concurrency", "tau", "tok_per_s")

def load_records(directory, benchmark):
    """Usable records only; an unusable one is named, never silently dropped."""
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        rec = json.loads(path.read_text())
        if benchmark and rec.get("benchmark") not in (None, benchmark):
            continue
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            print(f"NOTE skipped {path.name}: missing {', '.join(missing)}; has {', '.join(sorted(rec))}")
        else:
            records.append(rec)
    return records

def mean(recs, name):
    vals = [float(r[name]) for r in recs if r.get(name) is not None]
    return statistics.fmean(vals) if vals else None

def aggregate(records):
    """(arm, concurrency) -> mean tau, tok/s, ms_per_round_p50 over seeds."""
    groups = {}
    for rec in records:
        groups.setdefault((str(rec["arm"]), int(rec["concurrency"])), []).append(rec)
    return {key: {"tau": mean(recs, "tau"), "tok": mean(recs, "tok_per_s"),
                  "ms": mean(recs, "ms_per_round_p50")} for key, recs in groups.items()}

def field(agg, arm, c, name):
    entry = agg.get((arm, c))
    return None if entry is None else entry[name]

def ratio(agg, num, den, c, name):
    a, b = field(agg, num, c, name), field(agg, den, c, name)
    return None if a is None or b is None or not b else a / b

def fmt(value, spec):
    return ABSENT if value is None else format(value, spec)

def width_cell(agg, arm, c):
    tok, tau = field(agg, arm, c, "tok"), field(agg, arm, c, "tau")
    return ABSENT if tok is None or tau is None else f"{tok:.0f}/{tau:.2f}"

def speed_cell(agg, arm, c):
    return (f"{fmt(ratio(agg, arm, AR, c, 'tok'), '.2f')}/"
            f"{fmt(ratio(agg, arm, BASE, c, 'tok'), '.2f')}")

def render(title, arms, agg, cs, cell):
    print(f"\n{title}\n{'arm':<8}" + "".join(f"{f'c={c}':>17}" for c in cs))
    for arm in arms:
        print(f"{arm:<8}" + "".join(f"{cell(agg, arm, c):>17}" for c in cs))

def breakeven(agg, c, verbose=True):
    """tau_k(V16k<k>)/tau_k(B<k>) against measured per-round cost; 1.10 decides WIN/LOSE."""
    rows = []
    for k in KS:
        v, b = f"V16k{k}", f"B{k}"
        tok_r = ratio(agg, v, b, c, "tok")
        rows.append({"k": k, "tau_ratio": ratio(agg, v, b, c, "tau"),
                     "cost_ratio": ratio(agg, v, b, c, "ms"), "tok_ratio": tok_r,
                     "verdict": ABSENT if tok_r is None
                     else ("WIN" if tok_r > THRESHOLD else "LOSE")})
    if not verbose:
        return rows
    print(f"\n[4] break-even at c={c} — VLC(k) vs block-k; the pre-registered cost model needs "
          f"tau ratio > cost ratio, WIN needs tok/s ratio > {THRESHOLD:.2f}; k=4 has no block-k"
          f" arm, since b in {BS}")
    print("  tau ratio = tau(V16k<k>)/tau(B<k>); cost ratio = ms_p50(V16k<k>)/ms_p50(B<k>)")
    print(f"{'k':<4}{'tau_ratio':>11}{'cost_ratio':>12}{'tok_ratio':>11}")
    for r in rows:
        print(f"{r['k']:<4}{fmt(r['tau_ratio'], '.3f'):>11}"
              f"{fmt(r['cost_ratio'], '.3f'):>12}{fmt(r['tok_ratio'], '.3f'):>11}")
    print()
    for r in rows:
        why = (f"V16k{r['k']} or B{r['k']} has no record at c={c}; no verdict"
               if r["verdict"] == ABSENT
               else f"tok/s {r['tok_ratio']:.3f} (threshold {THRESHOLD:.2f})")
        print(f"k={r['k']:<3} {r['verdict']} — {why}")
    return rows

def report(agg, directory, benchmark):
    present = sorted({a for a, _ in agg})
    cs = sorted({c for _, c in agg})
    hl = 32 if 32 in cs else (cs[-1] if cs else None)
    register = [f"A{i}" for i in range(6)] + [f"V16k{k}" for k in KS] + [f"B{b}" for b in BS]
    absent = [a for a in register if a not in present]  # never estimated, only named
    print(f"== ablate.py — results={directory} benchmark={benchmark or 'any'}\n"
          f"nothing is projected: every cell is a mean over seeds of a measured record, or "
          f"'{ABSENT}'.\narms present: {', '.join(present) or 'none'}\n"
          f"arms absent : {', '.join(absent) or 'none'}\n"
          f"concurrency levels: {', '.join(map(str, cs)) or 'none'}")
    render("[1] verify width — V16k<k>; cell = tok/s / tau; rows = verify width k",
           [f"V16k{k}" for k in KS], agg, cs, width_cell)
    render("[2] block width — B<b>; cell = tok/s / tau; rows = block width b, verified in full",
           [f"B{b}" for b in BS], agg, cs, width_cell)
    render(f"[3] speedup; cell = tok/s(arm)/tok/s({AR}) / tok/s(arm)/tok/s({BASE}); "
           f"{AR} = autoregressive reference, {BASE} = DFlash baseline",
           present, agg, cs, speed_cell)
    breakeven(agg, hl) if hl is not None else print("\n[4] break-even — no records")

def selftest():
    # 1.20x on tau at c=32 is a WIN at 1.10; the hand-built agg below exercises LOSE.
    table = [("A0", 5.50, 2000.0, 44.0), ("A5", 1.00, 1000.0, 16.0),
             ("V16k9", 4.40, 2400.0, 36.0), ("B9", 4.00, 2000.0, 44.0)]
    records = [{"arm": a, "benchmark": "math500", "concurrency": 32, "tau": t,
                "tok_per_s": s, "ms_per_round_p50": m} for a, t, s, m in table]
    with tempfile.TemporaryDirectory() as tmp:
        for rec in records:
            Path(tmp, f"{rec['arm']}_math500_c32_s0.json").write_text(json.dumps(rec))
        agg = aggregate(load_records(tmp, "math500"))
    assert len(agg) == 4 and agg[("V16k9", 32)]["tok"] == 2400.0
    assert width_cell(agg, "V16k9", 32) == "2400/4.40" and width_cell(agg, "V16k4", 32) == ABSENT
    assert speed_cell(agg, "V16k9", 32) == "2.40/1.20" and speed_cell(agg, "B6", 32) == "absent/absent"
    assert speed_cell(agg, "A5", 32) == "1.00/0.50"
    rows = breakeven(agg, 32, verbose=False)
    assert rows[2]["k"] == 9 and rows[2]["verdict"] == "WIN" and rows[0]["verdict"] == ABSENT
    assert abs(rows[2]["tau_ratio"] - 1.10) < 1e-9 and abs(rows[2]["tok_ratio"] - 1.20) < 1e-9
    assert abs(rows[2]["cost_ratio"] - 36 / 44) < 1e-9   # 36/44 is the measured ms ratio
    lose = {("V16k9", 32): {"tau": 4.4, "tok": 2100.0, "ms": 36.0},
            ("B9", 32): {"tau": 4.0, "tok": 2000.0, "ms": 44.0}}
    assert breakeven(lose, 32, verbose=False)[2]["verdict"] == "LOSE"
    print("SELFTEST=PASS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ablation tables from result JSONs")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--benchmark", default="math500")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        report(aggregate(load_records(args.results, args.benchmark)), args.results,
               args.benchmark)
