"""Compose the acceptance surface with the engine's round cost into a tok/s ratio for VLC(16,9).

Two instruments are multiplied here, and that multiplication is an assumption, not a measurement:

* acceptance tau comes from the standalone harness (`experiments/standalone/accept_math500.json`,
  batch 1, one A100), where the round time is flat in columns and columns therefore cannot be
  priced;
* round cost comes from the engine (`results/*_math500_c32_s0.json`, c=32, default kernels), which
  does price them: 17 columns cost 50.74 ms, 9 columns cost 30.31 ms.

The composition is tok/s(VLC)/tok/s(arm) = [tau_9(16)/tau_arm] x [round(arm)/round(9 columns)].

Prints [1] every input with its file:key, [2] the composition against A0 (block 16, 17 columns)
and A1 (block 9, 9 columns), [3] a bootstrap CI over prompts for the tau-only factors, [4] the
engine's own A1/A0 cross-check from one instrument, and [5] the claim the numbers license plus the
line ``COMPOSED, NOT END-TO-END``.

A factor the files do not carry prints ``absent``; nothing is interpolated past a block's drafted
width, and any missing factor turns the verdict into ``no verdict``.

Run: uv run python scripts/vlc_tokps.py [experiments/standalone/accept_math500.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"
DEFAULT_PATH = Path("experiments/standalone/accept_math500.json")

# VLC(16,9): the wide block's proposals, verified through a 9-column window.
BLOCK, K = 16, 9
# The arms it is compared with. A0 verifies the whole wide block (17 columns on the engine, which
# counts the target's correction token); A1 is the one-flag narrow arm, block 8 / 9 columns as
# launched, labelled block 9 by its verify window.
A0_BLOCK, A0_K = 16, 16
A1_BLOCK, A1_K = 9, 9

BENCH, CONC, SEED_PREF = "math500", 32, 0
ENGINE_DIRS = (("default", ROOT / "results"), ("deterministic", ROOT / "results_det"))
CROSS_MODES = ("default", "deterministic")

TAU_BAR, TOK_BAR = 1.05, 1.10
RESAMPLES, BOOT_SEED = 10_000, 0
ABSENT = "absent"
ARMS = ("A0", "A1", "A5")
REQUIRED_ENGINE = ("arm", "benchmark", "concurrency", "seed", "ms_per_round_p50", "tok_per_s",
                   "tau", "verify_columns")

COLORS = {"A0": "#1f77b4", "A1": "#d62728", "A5": "#7f7f7f"}
MARKERS = {"A0": "o", "A1": "s", "A5": "D"}


# --- reading the sweep --------------------------------------------------------


def rel(path: str | Path | None) -> str:
    """Path relative to the repo root when possible, so every printed source is portable."""
    if path is None:
        return ABSENT
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def tau_key(b: int, k: int) -> str:
    return f"b{b}_k{k}"


def mean_tau(rows: list[dict], b: int, k: int) -> float | None:
    """mean min(accepted, k) over one block's rows; the definition accept.py uses."""
    vals = [min(int(r["accepted"]), k) for r in rows
            if r.get("block") == b and r.get("accepted") is not None]
    return statistics.fmean(vals) if vals else None


def tau_at(data: dict, b: int, k: int) -> tuple[float | None, str]:
    """tau_k(b) from the tau map, else computed from rows; ``absent`` past the drafted width.

    k = b + 1 is allowed: ``accepted`` counts the target's correction token, so a block of b drafts
    commits up to b + 1 tokens. Above that the file defines nothing and nothing is extrapolated.
    """
    if k > b + 1:
        return None, ABSENT
    val = (data.get("tau") or {}).get(tau_key(b, k))
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val), f"tau[{tau_key(b, k)}]"
    derived = mean_tau(data.get("rows") or [], b, k)
    return (derived, f"rows[block={b}] min(accepted,{k}) mean") if derived is not None \
        else (None, ABSENT)


def per_prompt(rows: list[dict], b: int, k: int) -> dict[tuple, float]:
    """(prompt, ctx_len) -> mean min(accepted, k) for one block."""
    grouped: dict[tuple, list[int]] = {}
    for r in rows:
        if r.get("block") != b or r.get("accepted") is None:
            continue
        grouped.setdefault((r.get("prompt"), r.get("ctx_len")), []).append(
            min(int(r["accepted"]), k))
    return {key: statistics.fmean(vals) for key, vals in grouped.items()}


def paired(num: dict[tuple, float], den: dict[tuple, float]) -> tuple[list[float], list[float]]:
    """Per-prompt numerator and denominator on the same (prompt, ctx_len); order-stable."""
    keys = sorted(set(num) & set(den), key=lambda t: (str(t[0]), str(t[1])))
    return [num[key] for key in keys], [den[key] for key in keys]


# --- statistics ---------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between order statistics, q in percent."""
    if not values:
        return None
    pos = (len(values) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_ratio_ci(num: list[float], den: list[float],
                       resamples: int = RESAMPLES, seed: int = BOOT_SEED) -> tuple:
    """Percentile CI for mean(num)/mean(den), resampling prompts with a fixed seed."""
    n = len(num)
    if n < 2:
        return (None, None), 0
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        bottom = sum(den[i] for i in idx)
        if bottom:
            draws.append(sum(num[i] for i in idx) / bottom)
    draws.sort()
    return (percentile(draws, 2.5), percentile(draws, 97.5)), len(draws)


# --- reading the engine records -----------------------------------------------


def load_engine(dirs=ENGINE_DIRS) -> dict[tuple, dict]:
    """(mode, arm, benchmark, concurrency, seed) -> record plus the file it came from."""
    out: dict[tuple, dict] = {}
    for mode, directory in dirs:
        if not directory.is_dir():
            print(f"absent: {rel(directory)}/ (directory missing)")
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".texts.json"):
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skip: {rel(path)} ({exc})")
                continue
            if not isinstance(data, dict):
                continue
            missing = [key for key in REQUIRED_ENGINE if key not in data]
            if missing:
                print(f"skip: {rel(path)} (missing keys: {missing})")
                continue
            key = (mode, str(data["arm"]), str(data["benchmark"]),
                   int(data["concurrency"]), int(data["seed"]))
            out[key] = {"rec": data, "path": path}
    return out


def pick(engine: dict[tuple, dict], mode: str, arm: str,
         conc: int = CONC) -> tuple[dict | None, Path | None]:
    """The record for (mode, arm) at the served concurrency, preferring the default seed."""
    seeds = sorted({s for (m, a, _b, c, s) in engine if m == mode and a == arm and c == conc})
    if not seeds:
        return None, None
    seed = SEED_PREF if SEED_PREF in seeds else seeds[0]
    hit = engine.get((mode, arm, BENCH, conc, seed))
    return (hit["rec"], hit["path"]) if hit else (None, None)


def draft_median_ms(data: dict, b: int) -> float | None:
    """Median draft pass of one block from the harness timings; the width penalty lives here."""
    vals = [float(t["draft_s"]) for t in data.get("timings") or []
            if t.get("block") == b and t.get("draft_s") is not None]
    return 1000.0 * statistics.median(vals) if vals else None


def harness_rounds_ms(data: dict) -> dict[int, tuple[float, float]]:
    """block -> (round ms, verify ms) medians, for the flatness panel."""
    out: dict[int, tuple[float, float]] = {}
    for b in sorted({int(t["block"]) for t in data.get("timings") or []
                     if t.get("block") is not None}):
        ts = [t for t in data["timings"] if t["block"] == b]
        draft = [float(t["draft_s"]) for t in ts if t.get("draft_s") is not None]
        verify = [float(t["verify_s"]) for t in ts if t.get("verify_s") is not None]
        if draft and verify:
            out[b] = (1000.0 * (statistics.median(draft) + statistics.median(verify)),
                      1000.0 * statistics.median(verify))
    return out


# --- composition --------------------------------------------------------------


CELLS = ((BLOCK, A0_K, "tau_16(16)"), (BLOCK, K, "tau_9(16)"), (A1_BLOCK, A1_K, "tau_9(9)"))


def compose(data: dict, engine: dict[tuple, dict]) -> dict:
    """Every factor of the composition, with the source of each, and the two directed results."""
    out: dict = {"cells": {}, "missing": [], "targets": {}}
    for b, k, label in CELLS:
        value, source = tau_at(data, b, k)
        out["cells"][label] = (value, f"{label.split('(')[0]}={value}" if value else ABSENT,
                               source)
        if value is None:
            out["missing"].append(f"{rel('experiments/standalone/accept_math500.json')}:"
                                  f"{tau_key(b, k)}")

    t1616 = out["cells"]["tau_16(16)"][0]
    t916 = out["cells"]["tau_9(16)"][0]
    t99 = out["cells"]["tau_9(9)"][0]

    rounds: dict[str, tuple[float | None, str]] = {}
    for arm in ("A0", "A1"):
        rec, path = pick(engine, "default", arm)
        if rec is None:
            rounds[arm] = (None, ABSENT)
            out["missing"].append(f"results/{arm}_{BENCH}_c{CONC}_s{SEED_PREF}.json:"
                                  "ms_per_round_p50")
            continue
        rounds[arm] = (float(rec["ms_per_round_p50"]),
                       f"{rel(path)}:ms_per_round_p50")
    out["rounds"] = rounds

    rows = data.get("rows") or []
    p916, p1616, p99 = per_prompt(rows, BLOCK, K), per_prompt(rows, BLOCK, A0_K), \
        per_prompt(rows, A1_BLOCK, A1_K)

    specs = {
        "A0": {"den_cell": t1616, "den_paired": paired(p916, p1616),
               "cost": (rounds["A0"][0] / rounds["A1"][0]
                        if rounds["A0"][0] and rounds["A1"][0] else None)},
        "A1": {"den_cell": t99, "den_paired": paired(p916, p99),
               "cost": (rounds["A1"][0] / rounds["A1"][0]
                        if rounds["A1"][0] else None)},
    }
    for target, spec in specs.items():
        num_p, den_p = spec["den_paired"]
        tau_cells = t916 / spec["den_cell"] if t916 and spec["den_cell"] else None
        tau_paired = (statistics.fmean(num_p) / statistics.fmean(den_p)
                      if num_p and statistics.fmean(den_p) else None)
        ci, resamples = bootstrap_ratio_ci(num_p, den_p)
        cost = spec["cost"]
        entry = {
            "tau_cells": tau_cells, "tau_paired": tau_paired, "cost": cost,
            "composed_cells": tau_cells * cost if tau_cells is not None and cost else None,
            "composed_paired": tau_paired * cost if tau_paired is not None and cost else None,
            "ci": ((ci[0] * cost if ci[0] is not None and cost else None),
                   (ci[1] * cost if ci[1] is not None and cost else None)),
            "ci_tau": ci, "resamples": resamples, "prompts": len(num_p),
            "wins": sum(1 for x, y in zip(num_p, den_p) if x > y),
            "ties": sum(1 for x, y in zip(num_p, den_p) if x == y),
            "losses": sum(1 for x, y in zip(num_p, den_p) if x < y),
        }
        out["targets"][target] = entry
        if entry["composed_paired"] is None and target not in out["missing"]:
            out["missing"].append(f"{target}: tau or round cost")

    out["draft_ms"] = {"b16": draft_median_ms(data, BLOCK), "b8": draft_median_ms(data, 8)}

    cross = []
    for mode in CROSS_MODES:
        seeds = sorted({s for (m, a, _b, c, s) in engine if m == mode and a in ARMS and c == CONC})
        for seed in seeds:
            a0 = engine.get((mode, "A0", BENCH, CONC, seed))
            a1 = engine.get((mode, "A1", BENCH, CONC, seed))
            a5 = engine.get((mode, "A5", BENCH, CONC, seed))
            if a0 is None or a1 is None:
                continue
            tau_a0, tau_a1 = float(a0["rec"]["tau"]), float(a1["rec"]["tau"])
            tok_a0, tok_a1 = float(a0["rec"]["tok_per_s"]), float(a1["rec"]["tok_per_s"])
            ms_a0 = float(a0["rec"]["ms_per_round_p50"])
            ms_a1 = float(a1["rec"]["ms_per_round_p50"])
            cross.append({
                "mode": mode, "seed": seed, "path_a0": a0["path"], "path_a1": a1["path"],
                "tau_a0": tau_a0, "tau_a1": tau_a1, "tok_a0": tok_a0, "tok_a1": tok_a1,
                "ms_a0": ms_a0, "ms_a1": ms_a1,
                "tau_ratio": tau_a1 / tau_a0 if tau_a0 else None,
                "cost_ratio": ms_a0 / ms_a1 if ms_a1 else None,
                "tok_ratio": tok_a1 / tok_a0 if tok_a0 else None,
                "tok_a5": float(a5["rec"]["tok_per_s"]) if a5 else None,
            })
    out["cross"] = cross

    out["verdict"], out["claims"] = verdict(out)
    return out


def verdict(out: dict) -> tuple[str, list[str]]:
    """The claims the numbers carry, and ``no verdict`` when any factor is missing."""
    if out["missing"]:
        return "no verdict", [f"missing input: {name}" for name in out["missing"]]

    claims: list[str] = []
    a0, a1 = out["targets"]["A0"], out["targets"]["A1"]
    low_a0 = min(a0["composed_cells"], a0["composed_paired"])
    side = "above" if low_a0 >= TOK_BAR else "below"
    claims.append(f"VLC(16,9) over A0 is {a0['composed_cells']:.3f}x on the pooled cells and "
                  f"{a0['composed_paired']:.3f}x on the paired reading, both {side} the "
                  f"{TOK_BAR:.2f} bar")
    lo, hi = a1["ci"]
    covers_parity = lo is not None and hi is not None and lo <= 1.0 <= hi
    beats = a1["composed_paired"] >= TAU_BAR and not covers_parity
    claims.append(
        f"VLC(16,9) over A1 is {a1['composed_paired']:.3f}x, CI "
        f"[{lo:.3f}, {hi:.3f}], which {'covers' if covers_parity else 'excludes'} 1.0: "
        + ("parity, not a win" if covers_parity else ("a win" if beats else "not a win")))
    verdicts = [a1["composed_paired"] >= TAU_BAR, not covers_parity]
    head = "supported" if all(verdicts) else "not supported"
    tail = (f"VLC vs A1: {head} ({a1['composed_paired']:.3f}, "
            + f"CI [{lo:.3f}, {hi:.3f}])")
    head_a0 = f"VLC vs A0: {a0['composed_cells']:.3f} pooled / {a0['composed_paired']:.3f} paired; "
    return head_a0 + tail, claims


# --- printing -----------------------------------------------------------------


def fmt(value, spec: str = ".3f") -> str:
    return ABSENT if value is None else format(value, spec)


def render(data: dict, engine: dict[tuple, dict], path: str | Path) -> dict:
    a = compose(data, engine)
    print(f"== vlc_tokps.py: {path}")
    print("acceptance from the standalone harness (batch 1); round cost from the engine (c=32). "
          "No factor is interpolated; a factor the files lack prints 'absent'.")

    print("\n[1] inputs, each with the file:key it was read from")
    print(f"  {'input':<38}{'value':>10}  source")
    sweep = rel(path)
    for b, k, label in CELLS:
        value, _txt, source = a["cells"][label]
        print(f"  {label + ' acceptance cell':<38}{fmt(value):>10}  {sweep}:{source}")
    print(f"  {'round(17 cols) c=32 ms/round':<38}{fmt(a['rounds']['A0'][0]):>10}  "
          f"{a['rounds']['A0'][1]}")
    print(f"  {'round(9 cols) c=32 ms/round':<38}{fmt(a['rounds']['A1'][0]):>10}  "
          f"{a['rounds']['A1'][1]}")
    rec_a0, path_a0 = pick(engine, "default", "A0")
    rec_a1, path_a1 = pick(engine, "default", "A1")
    for rec, pth, arm in ((rec_a0, path_a0, "A0"), (rec_a1, path_a1, "A1")):
        if rec is None:
            print(f"  {f'tau({arm}) engine c=32':<38}{ABSENT:>10}  results/{arm}_{BENCH}_c{CONC}")
            continue
        print(f"  {f'tau({arm}) engine c=32':<38}{float(rec['tau']):>10.3f}  {rel(pth)}:tau")
        print(f"  {f'tok/s({arm}) engine c=32':<38}{float(rec['tok_per_s']):>10.2f}  "
              f"{rel(pth)}:tok_per_s")
    print(f"  {'draft pass median, b=8 (harness)':<38}{fmt(a['draft_ms']['b8']):>10}  "
          f"{sweep}:timings[block=8].draft_s median")
    print(f"  {'draft pass median, b=16 (harness)':<38}{fmt(a['draft_ms']['b16']):>10}  "
          f"{sweep}:timings[block=16].draft_s median")
    if a["draft_ms"]["b8"] and a["draft_ms"]["b16"]:
        ratio = a["draft_ms"]["b16"] / a["draft_ms"]["b8"]
        print(f"  width penalty on the draft pass (b16/b8) = {ratio:.3f}; the composition does "
              f"not apply it, and it is the size of the error that leaves out.")

    print("\n[2] composition: tok/s ratio = [acceptance] x [round(arm) / round(9 columns)]")
    print(f"  {'factor':<42}{'vs A0 (17 cols)':>17}{'vs A1 (9 cols)':>17}")
    print(f"  {'acceptance, pooled cells':<42}"
          f"{fmt(a['targets']['A0']['tau_cells']):>17}{fmt(a['targets']['A1']['tau_cells']):>17}")
    print(f"  {'acceptance, paired over prompts':<42}"
          f"{fmt(a['targets']['A0']['tau_paired']):>17}"
          f"{fmt(a['targets']['A1']['tau_paired']):>17}")
    print(f"  {'round cost ratio':<42}{fmt(a['targets']['A0']['cost']):>17}"
          f"{fmt(a['targets']['A1']['cost']):>17}")
    print(f"  {'composed, pooled cells':<42}"
          f"{fmt(a['targets']['A0']['composed_cells']):>17}"
          f"{fmt(a['targets']['A1']['composed_cells']):>17}")
    print(f"  {'composed, paired':<42}{fmt(a['targets']['A0']['composed_paired']):>17}"
          f"{fmt(a['targets']['A1']['composed_paired']):>17}")
    print("  pooled cells weight every round equally, the paired reading every prompt equally, "
          "so they differ")
    print("  the verdict is read at the pessimistic (paired) one.")
    print(f"  round(9)/round(9) for the A1 target is an identity on one record "
          f"({a['rounds']['A1'][1]}), not a measurement.")

    print(f"\n[3] bootstrap CI over prompts for the tau-only factors ({RESAMPLES} resamples, "
          f"random.Random(seed={BOOT_SEED}), percentile)")
    for target in ("A0", "A1"):
        t = a["targets"][target]
        den = "tau_16(16)" if target == "A0" else "tau_9(9)"
        lo, hi = t["ci_tau"]
        print(f"  tau_9(16)/{den}: {fmt(t['tau_paired'])}  95% CI [{fmt(lo)}, {fmt(hi)}]  "
              f"n={t['prompts']} prompts  (wins {t['wins']}, ties {t['ties']}, "
              f"losses {t['losses']})")
        mids = "covers 1.0" if lo is not None and lo <= 1.0 <= hi else "excludes 1.0"
        print(f"  composed over {target} at cost {fmt(t['cost'])}: "
              f"{fmt(t['composed_paired'])}  95% CI [{fmt(t['ci'][0])}, {fmt(t['ci'][1])}]  "
              f"({mids}; the cost factor is a point, so the interval is the tau interval scaled)")
        if t["prompts"] < 2:
            print(f"  {ABSENT}: need >= 2 prompts measured under both blocks for a paired CI")

    print(f"\n[4] engine-only cross-check, both terms from one instrument (c={CONC}, {BENCH})")
    print(f"  {'mode':<14}{'seed':>5}{'tau A0':>9}{'tau A1':>9}{'tau A1/A0':>11}"
          f"{'round A0/A1':>13}{'tok/s A1/A0':>13}{'A1/A0 vs A5':>13}")
    for row in a["cross"]:
        tok_vs_ar = (row["tok_ratio"] / (row["tok_a5"] / row["tok_a0"])
                     if row["tok_a5"] and row["tok_a0"] else None)
        print(f"  {row['mode']:<14}{row['seed']:>5}{row['tau_a0']:>9.3f}{row['tau_a1']:>9.3f}"
              f"{fmt(row['tau_ratio']):>11}{fmt(row['cost_ratio']):>13}"
              f"{fmt(row['tok_ratio']):>13}{fmt(tok_vs_ar):>13}")
    for row in a["cross"]:
        product = (row["tau_ratio"] or 0) * (row["cost_ratio"] or 0)
        print(f"  {row['mode']} s{row['seed']}: tau ratio x round-cost ratio = "
              f"{fmt(row['tau_ratio'])} x {fmt(row['cost_ratio'])} = {fmt(product)} against a "
              f"measured tok/s ratio of {fmt(row['tok_ratio'])}; the {fmt(row['tok_ratio'])} is "
              f"measured, the product is the model, and the gap is the engine's aggregate "
              f"accounting.")
    if not a["cross"]:
        print(f"  {ABSENT}: no (mode, seed) has both an A0 and an A1 record at c={CONC}")

    print("\n[5] what the numbers license")
    for claim in a["claims"]:
        print(f"  - {claim}")
    for row in a["cross"]:
        print(f"  - engine-only A1/A0, {row['mode']} kernels: {fmt(row['tok_ratio'])}")
    print("  The truncation knob reaches the same place the one-flag width change reaches, about")
    print("  1.4x over the deployed block-16 arm; a claim that VLC beats A1 is not supported.")
    print("\nCOMPOSED, NOT END-TO-END")
    print("  Acceptance is measured at batch 1 and round cost at batch 32, in two separate "
          "runs; one end-to-end run of draft 16 with a 9-column verify window on the engine would "
          "make it end-to-end, and this backend cannot launch it, because one flag sets both the "
          "drafter block and the verify width (docs/RESULTS.md, Correction 2026-11-19).")
    print(f"verdict: {a['verdict']}")
    return a


# --- figure -------------------------------------------------------------------


def figure(data: dict, a: dict, path: str | Path,
           engine: dict[tuple, dict]) -> tuple[Path, Path] | None:
    """fig5: the two factors with their instruments, then the composed ratio and its CI."""
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    })
    sweep = rel(path)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3))

    # (a) acceptance surface: what the 9-column window throws away.
    ax = axes[0]
    for b, colour, ls, label in ((BLOCK, COLORS["A0"], "-", "block 16 (wide)"),
                                 (A1_BLOCK, COLORS["A1"], "--", "block 9 (narrow)")):
        pts = [(k, tau_at(data, b, k)[0]) for k in range(1, b + 1)]
        pts = [(k, v) for k, v in pts if v is not None]
        if not pts:
            print(f"absent: {sweep}:tau[b{b}_k*] (no cells; panel (a) line skipped)")
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, ls=ls, lw=1.5,
                marker="o", ms=3.4, label=label)
    ax.axvspan(K, BLOCK + 0.5, color="0.90", zorder=0)
    ax.axvline(K, color="0.3", lw=1.0, ls=":")
    for b, colour, dy in ((BLOCK, COLORS["A0"], 8), (A1_BLOCK, COLORS["A1"], -12)):
        value = tau_at(data, b, K)[0]
        if value is None:
            continue
        ax.plot([K], [value], marker="o", ms=5.0, mfc="white", mec=colour, mew=1.4, zorder=5)
        ax.annotate(f"{value:.2f}", (K, value), textcoords="offset points", xytext=(7, dy),
                    fontsize=7.0, color=colour)
    wide = tau_at(data, BLOCK, A0_K)[0]
    if wide is not None:
        ax.plot([A0_K], [wide], marker="o", ms=5.0, mfc="white", mec=COLORS["A0"], mew=1.4,
                zorder=5)
        ax.annotate(f"{wide:.2f}", (A0_K, wide), textcoords="offset points", xytext=(-4, 7),
                    fontsize=7.0, color=COLORS["A0"])
    ax.set_xlabel("verify columns k")
    ax.set_ylabel("tau_k(b), accepted tokens per round")
    ax.set_title("acceptance: standalone harness, batch 1", fontsize=9.0)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.grid(True, axis="y", color="0.9", lw=0.6)
    ax.text(K + 0.35, 1.15, "columns a 9-wide window discards", fontsize=6.2, color="0.4")

    # (b) round cost against columns, from both instruments.
    ax = axes[1]
    rounds = harness_rounds_ms(data)
    if rounds:
        xs = sorted(rounds)
        ax.plot(xs, [rounds[b][0] for b in xs], color="0.45", lw=1.3, ls="--", marker="s",
                ms=4.0, label="harness, batch 1 (flat: cannot price a column)")
        for b in xs:
            ax.annotate(f"{rounds[b][0]:.1f}", (b, rounds[b][0]), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=6.2, color="0.35")
    points = []
    for arm in ("A1", "A0"):
        rec, _pth = pick(engine, "default", arm)
        if rec is None:
            print(f"absent: results/{arm}_{BENCH}_c{CONC}:ms_per_round_p50 (panel (b) skipped)")
            continue
        points.append((int(rec["verify_columns"]), float(rec["ms_per_round_p50"])))
    if len(points) == 2:
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], color=COLORS["A0"], lw=1.6,
                marker="o", ms=5.0, label=f"engine, c={CONC}, default kernels")
        for k, v in points:
            ax.annotate(f"{v:.1f} ms @ {k} cols", (k, v), textcoords="offset points",
                        xytext=(-6, 7), ha="right", fontsize=6.6, color=COLORS["A0"])
    ax.set_xlabel("verify columns")
    ax.set_ylabel("ms per round")
    ax.set_title("cost: only the engine prices a column", fontsize=9.0)
    ax.legend(fontsize=6.6, frameon=False, loc="lower right")
    ax.grid(True, axis="y", color="0.9", lw=0.6)

    # (c) the composition itself, against both arms and the engine's own cross-check.
    ax = axes[2]
    rows = []
    for target, depth in (("A0", 3), ("A1", 2)):
        t = a["targets"][target]
        if t["composed_paired"] is None:
            print(f"absent: composed over {target} (panel (c) row skipped)")
            continue
        lo, hi = t["ci"]
        ax.errorbar([t["composed_paired"]], [depth + 0.16],
                    xerr=[[t["composed_paired"] - lo], [hi - t["composed_paired"]]],
                    fmt="o", ms=6.2, color=COLORS[target], capsize=4, lw=1.4, zorder=5)
        ax.plot([t["composed_cells"]], [depth - 0.16], marker="o", ms=5.4, mfc="white",
                mec=COLORS[target], mew=1.3, ls="none", zorder=5)
        rows.append((depth, f"VLC(16,9) vs {target}"))
    det_offset = 0
    for row in a["cross"]:
        if row["tok_ratio"] is None:
            continue
        marker = "D" if row["mode"] == "default" else "d"
        ax.plot([row["tok_ratio"]], [0.7 - det_offset], marker=marker, ms=5.0, ls="none",
                mfc="none", mec="0.35", mew=1.3, zorder=5)
        ax.annotate(f"{row['tok_ratio']:.2f} {row['mode']}", (row["tok_ratio"], 0.7 - det_offset),
                    textcoords="offset points", xytext=(7, -2), fontsize=6.4, color="0.3")
        det_offset += 0.34
    rows.append((0.7, "engine-only A1/A0 cross-check"))
    ax.axvline(1.0, color="0.25", lw=1.0, ls=":")
    ax.axvline(TOK_BAR, color="0.25", lw=1.0, ls="--")
    ax.annotate(f"bar {TOK_BAR:.2f}", (TOK_BAR, 3.55), fontsize=6.4, color="0.3", rotation=90,
                va="top", ha="right")
    ax.annotate("parity", (1.0, 3.55), fontsize=6.4, color="0.3", rotation=90, va="top", ha="right")
    ax.set_yticks([depth for depth, _ in rows])
    ax.set_yticklabels([label for _, label in rows], fontsize=7.6)
    ax.set_ylim(-0.15, 3.7)
    ax.set_xlim(0.95, 1.62)
    ax.set_xlabel("tok/s ratio (filled = paired, hollow = pooled cells, bar = 95% CI)")
    ax.set_title("composed result, beside the one measured ratio", fontsize=9.0)
    ax.grid(True, axis="x", color="0.9", lw=0.6)

    fig.suptitle(
        f"VLC(16,9) throughput composition: acceptance from the batch-1 harness, round cost from "
        f"the c={CONC} engine (default kernels)\n"
        "Two instruments multiplied; the 1.4x is composed, not measured, and it is not a claim "
        "over the narrow arm",
        fontsize=9.4,
    )
    fig.text(
        0.01, -0.06,
        f"Composed, not end-to-end: tau from {sweep} (batch 1), ms/round from "
        f"results/*_{BENCH}_c{CONC}_s{SEED_PREF}.json (batch {CONC}).\nMultiplying them assumes "
        "the harness tau survives the engine's scheduler and that a 9-column verify costs the same "
        "whichever block drafted it; the harness draft pass is\n"
        f"{fmt(a['draft_ms']['b8'], '.2f')} ms at 8 slots and {fmt(a['draft_ms']['b16'], '.2f')} "
        "ms at 16, so the width term omitted from the cost is about 1%. Deterministic-kernel "
        "points are drawn hollow and are not comparable\nwith default-kernel ones. No end-to-end "
        "run of draft 16 / verify 9 exists: one SGLang flag sets both, so this backend cannot "
        "launch it (docs/RESULTS.md).",
        fontsize=6.2, color="0.35", ha="left", va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUTDIR / "fig5_tokps_composition.pdf", OUTDIR / "fig5_tokps_composition.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {rel(pdf)}, {rel(png)}")
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose VLC(16,9) tok/s from two instruments")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--no-figure", action="store_true", help="print only, write no figure")
    args = parser.parse_args()
    if not args.path.exists():
        print(f"== vlc_tokps.py: {args.path}")
        print(f"{ABSENT}: no file at {args.path}")
        print("verdict: no verdict")
        return
    data = load(args.path)
    engine = load_engine()
    a = render(data, engine, args.path)
    if not args.no_figure:
        if a["missing"]:
            print(f"figure not written: {len(a['missing'])} factor(s) absent, so the composition "
                  "has nothing to draw")
            return
        figure(data, a, args.path, engine)


if __name__ == "__main__":
    main()
