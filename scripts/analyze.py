"""Tables and the gain-vs-concurrency figure, from measured `results/*.json`.

Usage:
    uv run python scripts/analyze.py --results results --out figures/gain_vs_concurrency.png
    uv run python scripts/analyze.py --results results --benchmark math500
    uv run python scripts/analyze.py --selftest

The verdict belongs to `scripts/decide.py`; this script only describes whatever grid it is given.
Nothing here invents measurements. `load` skips unreadable or incomplete records and says why, and
a missing cell is reported as missing rather than filled in. The synthetic records in
`synthetic_runs()` exist for `--selftest` and for the tests; they are written to a temporary
directory and never into `results/`.

The pre-registered H3 bar is a constant of the experiment, not a parameter of the analysis: it
comes from `docs/EXPERIMENTS.md` §2 and is asserted equal to `decide.THRESHOLDS["H3_gain_min"]`
by `tests/test_analyze.py`, so the two copies cannot drift quietly.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Arms in report order (docs/PLAN.md §experiments). A0 is the DFlash baseline that every
# percentage is computed against; A5 is the AR reference, i.e. the denominator of the speedup,
# not a DFlash-derived arm.
ARMS = ("A0", "A1", "A2", "A3", "A4", "A5")
BASELINE = "A0"
AR_REFERENCE = "A5"
CONCURRENCIES = (1, 4, 8, 16, 32)

# The grid docs/PLAN.md commits to. A2 is a c=32 sweep, A4 runs at three concurrencies.
PROTOCOL_GRID: dict[str, tuple[int, ...]] = {
    "A0": CONCURRENCIES,
    "A1": CONCURRENCIES,
    "A2": (32,),
    "A3": CONCURRENCIES,
    "A4": (1, 8, 32),
    "A5": CONCURRENCIES,
}

# H3: tok/s(A1)/tok/s(A0) >= 1.15 at c=32, fixed before any run (EXPERIMENTS §2/§3).
H3_RATIO = 1.15
H3_GAIN_PCT = (H3_RATIO - 1.0) * 100.0

# Keys of a complete record (EXPERIMENTS §4, the set scripts/decide.py validates).
REQUIRED = {
    "arm", "benchmark", "concurrency", "seed", "draft_width", "verify_columns", "prompts",
    "tau", "tok_per_s", "ar_tok_per_s", "ms_per_round_p50", "ms_per_round_p95",
    "exact_match_ar", "graph_recaptures",
}

NUMERIC = (
    "tau", "tok_per_s", "ar_tok_per_s", "ms_per_round_p50", "ms_per_round_p95",
    "exact_match_ar",
)

ARM_LABELS = {
    "A0": "A0 DFlash (baseline)",
    "A1": "A1 verify 9 cols (fixed)",
    "A2": "A2 verify-length sweep",
    "A3": "A3 block width 8",
    "A4": "A4 adaptive verify length",
    "A5": "A5 AR reference",
}


# --------------------------------------------------------------------------- loading


def _normalise(data: dict) -> dict:
    """Coerce the fields the analysis groups on. Raises ValueError if a value is unusable."""
    out = dict(data)
    out["arm"] = str(out["arm"])
    out["benchmark"] = str(out["benchmark"])
    out["concurrency"] = int(out["concurrency"])
    out["seed"] = int(out["seed"])
    for key in NUMERIC:
        out[key] = float(out[key])
    if out["concurrency"] <= 0:
        raise ValueError(f"concurrency must be positive, got {out['concurrency']}")
    return out


def load(results_dir: Path) -> list[dict]:
    """Read every `results/*.json` record, skipping the ones that cannot be used.

    Prints one line per skipped file stating what is missing. Returns [] for an empty,
    absent, or entirely unusable directory; never raises on bad input.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"no results directory: {results_dir}")
        return []

    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        print(f"no *.json records in {results_dir}")
        return []

    runs: list[dict] = []
    seen: dict[tuple[str, str, int, int], str] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {path.name}: unreadable ({exc})")
            continue
        if not isinstance(data, dict):
            print(f"skip {path.name}: not a JSON object")
            continue
        missing = sorted(REQUIRED - data.keys())
        if missing:
            print(f"skip {path.name}: missing keys {missing}")
            continue
        try:
            run = _normalise(data)
        except (TypeError, ValueError) as exc:
            print(f"skip {path.name}: bad value ({exc})")
            continue
        key = (run["benchmark"], run["arm"], run["concurrency"], run["seed"])
        if key in seen:
            print(f"warn {path.name}: duplicate of {seen[key]} for "
                  f"{key[1]} c={key[2]} seed={key[3]} (both kept in the mean)")
        else:
            seen[key] = path.name
        runs.append(run)

    if not runs:
        print(f"no usable records in {results_dir}")
    runs.sort(key=lambda r: (r["benchmark"], r["arm"], r["concurrency"], r["seed"]))
    return runs


# --------------------------------------------------------------------------- aggregation


def cell(runs: list[dict], arm: str, c: int) -> list[dict]:
    return [r for r in runs if r["arm"] == arm and r["concurrency"] == c]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def cell_mean(runs: list[dict], arm: str, c: int, key: str) -> float | None:
    vals = [r[key] for r in cell(runs, arm, c)]
    return _mean(vals) if vals else None


def seeds_present(runs: list[dict]) -> dict[str, dict[int, int]]:
    """{arm: {concurrency: number of distinct seeds}} — what the caption has to admit to."""
    raw: dict[str, dict[int, set[int]]] = {}
    for r in runs:
        raw.setdefault(r["arm"], {}).setdefault(r["concurrency"], set()).add(r["seed"])
    return {arm: {c: len(s) for c, s in sorted(cells.items())}
            for arm, cells in sorted(raw.items())}


def missing_cells(runs: list[dict]) -> list[tuple[str, int]]:
    """Protocol cells with no record at all, as (arm, concurrency)."""
    present = {(r["arm"], r["concurrency"]) for r in runs}
    return [(arm, c) for arm in ARMS for c in PROTOCOL_GRID.get(arm, ())
            if (arm, c) not in present]


def _single_benchmark(runs: list[dict]) -> str:
    names = sorted({r["benchmark"] for r in runs})
    if len(names) > 1:
        raise ValueError(f"runs mix benchmarks {names}; filter to one benchmark first")
    return names[0] if names else "unknown"


def _fmt(values: list[float], spec: str) -> str:
    if not values:
        return "n/a"
    mean = _mean(values)
    if len(values) == 1:
        return format(mean, spec)
    return f"{format(mean, spec)} ±{format(_sd(values), spec)}"


# --------------------------------------------------------------------------- table


def table(runs: list[dict]) -> str:
    """Markdown table of arm x concurrency: mean tok/s, tau and ms/round per cell.

    A cell with several seeds reports mean ±sd, and `n` counts them. Empty input returns the
    header only (an empty table); the caller decides what to say about it.
    """
    benches = sorted({r["benchmark"] for r in runs})
    multi = len(benches) > 1

    head = "| arm |" + (" bench |" if multi else "") + " c | tok/s | tau | ms/round p50 | n |"
    rule = "|:--|" + (":--|" if multi else "") + "--:|--:|--:|--:|--:|"
    lines = [head, rule]

    arms = [a for a in ARMS if any(r["arm"] == a for r in runs)]
    arms += sorted({r["arm"] for r in runs} - set(ARMS))
    for bench in benches or [None]:
        for arm in arms:
            for c in sorted({r["concurrency"] for r in runs
                             if r["arm"] == arm and (bench is None or r["benchmark"] == bench)}):
                rows = [r for r in runs if r["arm"] == arm and r["concurrency"] == c
                        and (bench is None or r["benchmark"] == bench)]
                tok = [r["tok_per_s"] for r in rows]
                tau = [r["tau"] for r in rows]
                ms = [r["ms_per_round_p50"] for r in rows]
                lines.append(
                    f"| {arm} |" + (f" {bench} |" if multi else "")
                    + f" {c} | {_fmt(tok, '.1f')} | {_fmt(tau, '.2f')} | "
                    f"{_fmt(ms, '.2f')} | {len(rows)} |"
                )

    if runs:
        missing = missing_cells(runs)
        if missing:
            listed = ", ".join(f"{a} c={c}" for a, c in missing)
            lines.append("")
            lines.append(f"_missing protocol cells: {listed}_")
    return "\n".join(lines)


# --------------------------------------------------------------------------- gain


def gain_vs_baseline(runs: list[dict]) -> dict[tuple[str, int], float]:
    """Percent tok/s over A0 at the same concurrency, keyed by (arm, concurrency).

    A cell whose baseline is absent has no percentage and is omitted — not defaulted to zero.
    All records must come from one benchmark; mixing them would subtract unlike runs.
    """
    _single_benchmark(runs)
    out: dict[tuple[str, int], float] = {}
    for arm in sorted({r["arm"] for r in runs}):
        for c in sorted({r["concurrency"] for r in runs if r["arm"] == arm}):
            base = cell_mean(runs, BASELINE, c, "tok_per_s")
            got = cell_mean(runs, arm, c, "tok_per_s")
            if base is None or got is None or base <= 0:
                continue
            out[(arm, c)] = (got / base - 1.0) * 100.0
    return out


# --------------------------------------------------------------------------- figure


def caption(runs: list[dict], gains: dict[tuple[str, int], float], note: str = "") -> str:
    """The caption that goes into both the PNG metadata and the .txt sidecar."""
    bench = _single_benchmark(runs) if runs else None
    seeds = seeds_present(runs)
    cs = sorted({c for _, c in gains}) or list(CONCURRENCIES)
    order = "/".join(str(c) for c in cs)
    seed_line = "; ".join(
        f"{arm} " + "/".join(str(seeds[arm].get(c, 0)) for c in cs)
        for arm in sorted(seeds)
    ) or "none"

    parts = [
        (f"Gain over DFlash ({BASELINE}) at equal concurrency, measured on {bench}." if bench
         else f"Gain over DFlash ({BASELINE}) at equal concurrency: no records were present."),
        (f"y = 100 * (tok/s(arm, c) / tok/s({BASELINE}, c) - 1); each point is the mean over "
         "the seeds present at that cell, and is labelled with the mean acceptance tau."),
        (f"Dashed line: pre-registered H3 threshold, tok/s(A1)/tok/s({BASELINE}) >= "
         f"{H3_RATIO:.2f} at c=32, i.e. +{H3_GAIN_PCT:.0f}% (docs/EXPERIMENTS.md section 2), "
         "fixed before any run."),
        f"Seeds present per arm (c={order}): {seed_line}.",
        (f"{AR_REFERENCE} is the AR reference and the denominator of the speedup, not a "
         "DFlash-derived arm, so it is not drawn as a gain line."),
    ]
    absent = [c for c in cs if cell_mean(runs, BASELINE, c, "tok_per_s") is None]
    if absent:
        parts.append(f"No {BASELINE} record at c={absent}: those cells have no percentage.")
    if note:
        parts.append(note)
    return "\n".join(parts)


def figure(runs: list[dict], out: Path, *, note: str = "") -> Path:
    """Write the gain-vs-concurrency PNG (and a .txt caption sidecar); return the PNG path.

    Empty input still produces a labelled figure saying there is nothing to plot, so the report
    build never has to special-case a missing run.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gains = gain_vs_baseline(runs) if runs else {}
    cap = caption(runs, gains, note=note)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)

    arms = [a for a in ARMS if a != AR_REFERENCE and any(k[0] == a for k in gains)]
    arms += sorted({k[0] for k in gains} - set(ARMS))

    xs_all = sorted({k[1] for k in gains}) or list(CONCURRENCIES)
    x_lo, x_hi = min(xs_all), max(xs_all)
    pad = 1.35
    ax.set_xlim(x_lo / pad, x_hi * pad)

    for i, arm in enumerate(arms):
        cs = sorted(c for (a, c) in gains if a == arm)
        ys = [gains[(arm, c)] for c in cs]
        style = {"color": f"C{i}", "marker": "osD^v"[i % 5], "ms": 5.5, "lw": 1.6}
        ax.plot(cs, ys, label=ARM_LABELS.get(arm, arm), **style)
        for j, (c, y) in enumerate(zip(cs, ys, strict=True)):
            tau = cell_mean(runs, arm, c, "tau")
            label = f"τ={tau:.2f}" if tau is not None else "τ n/a"
            ax.annotate(label, (c, y), textcoords="offset points",
                        xytext=(6, 7 if j % 2 == 0 else -12), fontsize=7,
                        color=style["color"], alpha=0.9)

    ax.axhline(0.0, color="0.65", lw=0.9, zorder=0)
    ax.axhline(H3_GAIN_PCT, color="crimson", ls="--", lw=1.4, zorder=1,
               label=f"H3 threshold +{H3_GAIN_PCT:.0f}% (pre-registered)")
    ax.text(x_hi * pad / 1.02, H3_GAIN_PCT + 0.8, f"H3 bar: +{H3_GAIN_PCT:.0f}%",
            color="crimson", fontsize=8, ha="right", va="bottom")

    if not gains:
        ax.text(0.5, 0.5, "no usable results — nothing to plot", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="0.35")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs_all)
    ax.set_xticklabels([str(c) for c in xs_all])
    ax.minorticks_off()
    ax.set_xlabel("concurrency (log2 scale)")
    ax.set_ylabel(f"tok/s gain over {BASELINE} at the same concurrency (%)")
    bench = _single_benchmark(runs) if runs else None
    title = "Verify-length / width arms against DFlash"
    ax.set_title(title + (f" — {bench}" if bench else ""), fontsize=10)
    ax.grid(True, axis="y", color="0.9", lw=0.7)
    if arms:
        # Below the axes, so the legend never sits on top of the A3 points.
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", metadata={
        "Title": title + (f" ({bench})" if bench else ""),
        "Description": cap,
        "Software": "dflash-joint-budget scripts/analyze.py",
    })
    plt.close(fig)

    sidecar = out.with_suffix(".txt")
    sidecar.write_text(cap + "\n")
    return out


# --------------------------------------------------------------------------- synthetic data


# Illustrative shapes only: A1/A4 near the cost model's projection, A3 below baseline,
# A5 the AR reference. Deterministic (no RNG) so the selftest output is reproducible.
_A0_TOK = {1: 110.0, 4: 196.0, 8: 268.0, 16: 318.0, 32: 352.0}
_ARM_MULT = {
    "A1": {1: 1.156, 4: 1.224, 8: 1.293, 16: 1.321, 32: 1.351},
    "A2": {1: 1.190, 4: 1.270, 8: 1.340, 16: 1.372, 32: 1.401},
    "A3": {1: 0.816, 4: 0.881, 8: 0.913, 16: 0.935, 32: 0.954},
    "A4": {1: 1.170, 4: 1.245, 8: 1.312, 16: 1.338, 32: 1.366},
}
_DFLASH_SPEEDUP = {1: 5.1, 4: 4.15, 8: 3.60, 16: 3.12, 32: 2.80}
_TAU = {"A0": 8.01, "A1": 7.21, "A2": 7.00, "A3": 5.09, "A4": 7.30, "A5": 1.00}
_A0_MS = {1: 4.30, 4: 3.40, 8: 3.00, 16: 2.70, 32: 2.50}
_MS_FACTOR = {"A0": 1.00, "A1": 0.78, "A2": 0.75, "A3": 0.86, "A4": 0.77, "A5": 3.60}
_COLS = {"A0": 17, "A1": 9, "A2": 7, "A3": 9, "A4": 9, "A5": 0}
_WIDTH = {"A0": 16, "A1": 16, "A2": 16, "A3": 8, "A4": 16, "A5": 0}
# Protocol rule 4: three seeds on the c=32 cells of A0, A1, A3, A4.
_THREE_SEEDS = {("A0", 32), ("A1", 32), ("A3", 32), ("A4", 32)}


def synthetic_runs() -> list[dict]:
    """Plausible records for all six arms at c in {1,4,8,16,32}. Tests and --selftest only.

    Never write these into `results/`: they are shaped after the projection, not measured.
    """
    runs: list[dict] = []
    for arm in ARMS:
        for c in CONCURRENCIES:
            seeds = (0, 1, 2) if (arm, c) in _THREE_SEEDS else (0,)
            for seed in seeds:
                jitter = 1.0 + 0.004 * (seed - 1)
                if arm == AR_REFERENCE:
                    tok = _A0_TOK[c] / _DFLASH_SPEEDUP[c]
                elif arm == BASELINE:
                    tok = _A0_TOK[c]
                else:
                    tok = _A0_TOK[c] * _ARM_MULT[arm][c]
                runs.append({
                    "arm": arm,
                    "benchmark": "math500",
                    "concurrency": c,
                    "seed": seed,
                    "draft_width": _WIDTH[arm],
                    "verify_columns": _COLS[arm],
                    "prompts": 200,
                    "tau": _TAU[arm] * (1.0 + 0.001 * (seed - 1)),
                    "tok_per_s": round(tok * jitter, 2),
                    "ar_tok_per_s": 21.6,
                    "ms_per_round_p50": round(_A0_MS[c] * _MS_FACTOR[arm] / jitter, 3),
                    "ms_per_round_p95": round(_A0_MS[c] * _MS_FACTOR[arm] * 1.22 / jitter, 3),
                    "exact_match_ar": 0.997,
                    "graph_recaptures": 0,
                })
    return runs


# --------------------------------------------------------------------------- selftest


def selftest() -> Path:
    """Build a full synthetic grid, plot it into a temp dir, and check the artifacts."""
    runs = synthetic_runs()
    expected_cells = len(ARMS) * len(CONCURRENCIES)   # six arms at every concurrency
    assert len({(r["arm"], r["concurrency"]) for r in runs}) == expected_cells, "grid incomplete"
    assert missing_cells(runs) == [], "synthetic grid does not cover the protocol cells"

    gains = gain_vs_baseline(runs)
    assert abs(gains[("A1", 1)] - 15.6) < 0.6, gains[("A1", 1)]
    assert abs(gains[("A1", 32)] - 35.1) < 0.6, gains[("A1", 32)]
    assert gains[("A1", 32)] >= H3_GAIN_PCT
    assert gains[("A3", 32)] < 0.0
    assert gains[("A5", 32)] < -50.0
    assert all(abs(gains[(BASELINE, c)]) < 1e-9 for c in CONCURRENCIES)

    text = table(runs)
    rows = [ln for ln in text.splitlines() if ln.startswith("| A")]
    assert len(rows) == expected_cells, (len(rows), expected_cells)

    out_dir = Path(tempfile.mkdtemp(prefix="analyze-selftest-"))
    png = figure(runs, out_dir / "gain_vs_concurrency.png", note="SYNTHETIC selftest data.")
    assert png.is_file()
    size = png.stat().st_size
    assert size > 10_000, size

    sidecar = png.with_suffix(".txt")
    assert sidecar.is_file()
    cap = sidecar.read_text()
    for needle in ("H3", "+15%", "Seeds present", "A5"):
        assert needle in cap, needle
    meta = png.read_bytes()
    for needle in (b"H3 threshold", b"Seeds present", b"+15%"):
        assert needle in meta, f"PNG metadata missing {needle!r}"

    empty = figure([], out_dir / "empty.png", note="Empty-input check.")
    assert empty.is_file() and empty.stat().st_size > 5_000

    seeds = seeds_present(runs)
    print(f"selftest ok: {len(runs)} records, {expected_cells} cells, "
          f"{sum(len(v) for v in seeds.values())} arm/concurrency groups, "
          f"max seeds per cell {max(n for v in seeds.values() for n in v.values())}")
    print(f"figure:  {png} ({size} bytes)")
    print(f"caption: {sidecar} ({len(cap)} chars)")
    print(f"empty-input figure: {empty} ({empty.stat().st_size} bytes)")
    print()
    print(text)
    print()
    for arm in ARMS:
        cells = " ".join(f"c={c}:{gains[(arm, c)]:+.1f}%" for (a, c) in gains if a == arm)
        print(f"  {arm}: {cells}")
    return png


# --------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description="Tables and the gain figure for the report.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/gain_vs_concurrency.png")
    ap.add_argument("--benchmark", default="math500",
                    help="benchmark to tabulate and plot (default: math500)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthesise a full grid in a temp dir and check the artifacts")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    all_runs = load(Path(args.results))
    runs = [r for r in all_runs if r["benchmark"] == args.benchmark]
    skipped = len(all_runs) - len(runs)
    if skipped:
        print(f"ignoring {skipped} record(s) not on {args.benchmark}")
    if not runs:
        print(f"no usable records for {args.benchmark} in {args.results}; nothing written")
        return 1

    print()
    print(table(runs))

    missing = missing_cells(runs)
    print()
    print(f"missing protocol cells ({len(missing)}): "
          + (", ".join(f"{a} c={c}" for a, c in missing) or "none"))

    gains = gain_vs_baseline(runs)
    print()
    for (arm, c), pct in sorted(gains.items()):
        mark = "  <-- H3 cell" if (arm, c) == ("A1", 32) else ""
        print(f"  {arm} c={c:<3} {pct:+7.1f}% vs {BASELINE}{mark}")

    out = figure(runs, Path(args.out))
    print()
    print(f"figure:  {out}")
    print(f"caption: {out.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
