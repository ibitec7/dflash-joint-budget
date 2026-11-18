"""VLC ablation: wide draft truncated to a narrow verify window, from one accept.json.

Prints [1] the tau table over drafter blocks x verify columns, [2] the decisive paired
contrast tau_9(b=16) vs tau_9(b=8) with a paired bootstrap CI over prompts, [3] the cost
model from the measured timings, and [4] the numeric conditions that would falsify the run.

Every printed number is read from the file (or, where the file's ``tau`` map lacks a key,
computed from its ``rows``); a missing number prints ``absent`` and the run ends with
``no verdict``. Nothing is interpolated past a block's drafted width.

Run: uv run python scripts/vlc.py experiments/standalone/accept.json
     uv run python scripts/vlc.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
from pathlib import Path

KS = (5, 8, 9, 12, 16)
WIDE, NARROW, K_STAR = 16, 8, 9
TAU_BAR, TOK_BAR = 1.05, 1.10
RESAMPLES, SEED = 10_000, 0
ABSENT, NA = "absent", "n/a (block < k)"
DEFAULT_PATH = Path("experiments/standalone/accept.json")


# --- reading the file ---------------------------------------------------------


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def tau_key(b: int, k: int) -> str:
    return f"b{b}_k{k}"


def mean_tau(rows: list[dict], b: int, k: int) -> float | None:
    """mean min(accepted, k) over the rows of one block; the definition accept.py uses."""
    vals = [min(int(r["accepted"]), k) for r in rows
            if r.get("block") == b and r.get("accepted") is not None]
    return statistics.fmean(vals) if vals else None


def cell(data: dict, b: int, k: int) -> tuple[float | None, str]:
    """(value, source); source is one of tau / rows / na / absent."""
    if k > b:
        return None, "na"
    tmap = data.get("tau") or {}
    val = tmap.get(tau_key(b, k))
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val), "tau"
    derived = mean_tau(data.get("rows") or [], b, k)
    return (derived, "rows") if derived is not None else (None, "absent")


def blocks_in(data: dict) -> list[int]:
    """Blocks named by rows or by the tau map, widest first."""
    found = {int(r["block"]) for r in data.get("rows") or [] if r.get("block") is not None}
    for key in data.get("tau") or {}:
        if key.startswith("b") and "_k" in key:
            head = key[1:].split("_k", 1)[0]
            if head.isdigit():
                found.add(int(head))
    return sorted(found, reverse=True)


def per_prompt(rows: list[dict], b: int, k: int) -> dict[tuple, float]:
    """(prompt, ctx_len) -> mean min(accepted, k) for one block."""
    grouped: dict[tuple, list[int]] = {}
    for r in rows:
        if r.get("block") != b or r.get("accepted") is None:
            continue
        grouped.setdefault((r.get("prompt"), r.get("ctx_len")), []).append(
            min(int(r["accepted"]), k))
    return {key: statistics.fmean(vals) for key, vals in grouped.items()}


def paired_sets(data: dict, k: int) -> tuple[list[float], list[float]]:
    """Per-prompt values for both blocks on the same (prompt, ctx_len); order-stable."""
    rows = data.get("rows") or []
    wide, narrow = per_prompt(rows, WIDE, k), per_prompt(rows, NARROW, k)
    keys = sorted(set(wide) & set(narrow), key=lambda t: (str(t[0]), str(t[1])))
    return [wide[key] for key in keys], [narrow[key] for key in keys]


# --- statistics ---------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between order statistics, q in percent."""
    if not values:
        return None
    pos = (len(values) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_ratio_ci(wide: list[float], narrow: list[float],
                       resamples: int = RESAMPLES, seed: int = SEED) -> tuple:
    """Percentile CI for mean(wide)/mean(narrow), resampling prompts with a fixed seed."""
    n = len(wide)
    if n < 2:
        return (None, None), 0
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        den = sum(narrow[i] for i in idx)
        if den:
            draws.append(sum(wide[i] for i in idx) / den)
    draws.sort()
    return (percentile(draws, 2.5), percentile(draws, 97.5)), len(draws)


def timing_medians(data: dict, b: int) -> tuple:
    """(median draft_s, median verify_s, median columns) over the block's timing rows."""
    ts = [t for t in data.get("timings") or [] if t.get("block") == b]
    draft = [float(t["draft_s"]) for t in ts if t.get("draft_s") is not None]
    verify = [float(t["verify_s"]) for t in ts if t.get("verify_s") is not None]
    cols = [float(t["columns"]) for t in ts if t.get("columns") is not None]
    return (statistics.median(draft) if draft else None,
            statistics.median(verify) if verify else None,
            statistics.median(cols) if cols else None)


def round_s(data: dict, b: int, k: int) -> float | None:
    """median draft_s(b) + median verify_s(b) * k / columns(b); linear in columns."""
    d, v, c = timing_medians(data, b)
    if d is None or v is None or not c:
        return None
    return d + v * k / c


# --- analysis -----------------------------------------------------------------


def analyse(data: dict) -> dict:
    out: dict = {"blocks": blocks_in(data), "cells": {}, "sources": {}}
    for b in out["blocks"]:
        for k in KS:
            val, src = cell(data, b, k)
            out["cells"][(b, k)] = val
            out["sources"][(b, k)] = src

    out["tau_wide"] = out["cells"].get((WIDE, K_STAR))
    out["tau_narrow"] = out["cells"].get((NARROW, K_STAR))
    out["tau_ratio_table"] = (out["tau_wide"] / out["tau_narrow"]
                              if out["tau_wide"] is not None and out["tau_narrow"] else None)

    wide, narrow = paired_sets(data, K_STAR)
    out["paired_n"] = len(wide)
    out["paired_wins"] = sum(1 for x, y in zip(wide, narrow) if x > y)
    out["paired_ties"] = sum(1 for x, y in zip(wide, narrow) if x == y)
    out["paired_losses"] = sum(1 for x, y in zip(wide, narrow) if x < y)
    mean_wide, mean_narrow = (statistics.fmean(wide), statistics.fmean(narrow)) if wide else (0, 0)
    out["paired_ratio"] = mean_wide / mean_narrow if mean_narrow else None
    out["ci"], out["resamples"] = bootstrap_ratio_ci(wide, narrow)
    out["tau_factor"] = out["paired_ratio"]

    out["medians"] = {b: timing_medians(data, b) for b in (WIDE, NARROW)}
    out["round_wide"] = round_s(data, WIDE, K_STAR)
    out["round_narrow"] = round_s(data, NARROW, K_STAR)
    out["cost_factor"] = (out["round_narrow"] / out["round_wide"]
                          if out["round_narrow"] and out["round_wide"] else None)
    out["tok_ratio"] = (out["tau_factor"] * out["cost_factor"]
                        if out["tau_factor"] is not None and out["cost_factor"] else None)

    out["tau_ok"] = None if out["tau_factor"] is None else out["tau_factor"] >= TAU_BAR
    out["tok_ok"] = None if out["tok_ratio"] is None else out["tok_ratio"] >= TOK_BAR
    if out["tau_ok"] is None or out["tok_ok"] is None:
        out["verdict"] = "no verdict"
    else:
        out["verdict"] = "SUPPORTED" if out["tau_ok"] and out["tok_ok"] else "NOT SUPPORTED"
    out["tau_needed"] = TOK_BAR / out["cost_factor"] if out["cost_factor"] else None
    return out


# --- printing -----------------------------------------------------------------


def fmt(value, spec: str = ".3f") -> str:
    return ABSENT if value is None else format(value, spec)


def tri(flag: bool | None) -> str:
    return ABSENT if flag is None else ("passes" if flag else "fails")


def render(data: dict, path: str | Path) -> dict:
    a = analyse(data)
    print(f"== vlc.py — {path}")
    print("cells are the file's tau map (computed from rows where that key is missing); "
          "no value is interpolated past a block's drafted width.")
    missing = [k for k in ("rows", "tau", "timings") if k not in data]
    if missing:
        print(f"missing top-level keys: {', '.join(missing)} — printed as '{ABSENT}'; no verdict")
    cfg = data.get("config")
    print(f"config: {json.dumps(cfg, sort_keys=True) if cfg else ABSENT}")

    print(f"\n[1] tau table — cell = mean tau_k(b) = E[min(accepted, k)]; rows = drafter block, "
          f"columns = verify columns k; k > b is '{NA}'")
    print(f"{'b':>4}" + "".join(f"{f'k={k}':>17}" for k in KS))
    for b in a["blocks"]:
        cells = []
        for k in KS:
            val, src = a["cells"][(b, k)], a["sources"][(b, k)]
            cells.append(NA if src == "na" else (ABSENT if val is None else f"{val:.3f}"))
        print(f"{b:>4}" + "".join(f"{c:>17}" for c in cells))
    derived = [tau_key(b, k) for b in a["blocks"] for k in KS
               if a["sources"][(b, k)] == "rows"]
    if derived:
        print(f"  note: tau key absent, computed from rows: {', '.join(derived)}")
    if not a["blocks"]:
        print(f"  {ABSENT}: neither rows nor tau keys name a block")

    print(f"\n[2] decisive contrast at k={K_STAR}: tau_{K_STAR}(b={WIDE}) vs "
          f"tau_{K_STAR}(b={NARROW})")
    print(f"  table cells   tau_{K_STAR}({WIDE}) = {fmt(a['tau_wide'])}   "
          f"tau_{K_STAR}({NARROW}) = {fmt(a['tau_narrow'])}   "
          f"ratio = {fmt(a['tau_ratio_table'])}")
    print(f"  paired on (prompt, ctx_len) present in both blocks: {a['paired_n']} prompts")
    print(f"  paired ratio  = {fmt(a['paired_ratio'])}   "
          f"95% CI [{fmt(a['ci'][0])}, {fmt(a['ci'][1])}]   "
          f"({a['resamples']} resamples, random.Random(seed={SEED}), percentile)")
    if a["paired_n"] < 2:
        print(f"  {ABSENT}: need >= 2 prompts measured under both blocks for a paired CI")
    print(f"  per-prompt    wide>narrow {a['paired_wins']}, tie {a['paired_ties']}, "
          f"wide<narrow {a['paired_losses']}")

    print("\n[3] cost model — round(b,k) = median draft_s(b) + median verify_s(b) * k / "
          "columns(b)")
    print("  assumption, stated not derived: verify time is linear in columns over the range "
          f"measured; k={K_STAR} at b={NARROW} extrapolates past its measured columns.")
    for b in (WIDE, NARROW):
        d, v, c = a["medians"][b]
        print(f"  b={b:<2}: median draft_s = {fmt(d, '.5f')}   median verify_s = "
              f"{fmt(v, '.5f')}   median columns = {fmt(c, '.0f')}")
    print(f"  round({WIDE},{K_STAR}) = {fmt(a['round_wide'], '.5f')} s   "
          f"round({NARROW},{K_STAR}) = {fmt(a['round_narrow'], '.5f')} s")
    print(f"  predicted tok/s ratio ({WIDE},{K_STAR})/({NARROW},{K_STAR}) = "
          f"tau factor {fmt(a['tau_factor'])} x cost factor "
          f"round({NARROW},{K_STAR})/round({WIDE},{K_STAR}) {fmt(a['cost_factor'])} = "
          f"{fmt(a['tok_ratio'])}")
    if a["tok_ratio"] and a["tok_ratio"] > 0 and a["tau_factor"] and a["cost_factor"]:
        whole = math.log(a["tok_ratio"])
        acc_share = math.log(a["tau_factor"]) / whole if whole else 0.0
        print(f"  split of the log ratio: acceptance {acc_share:+.1%}, "
              f"cost {1 - acc_share:+.1%} of the {'gain' if whole > 0 else 'loss'}")

    print("\n[4] what would falsify this — conditions evaluated on this file, not asserted")
    print(f"  needs tau_{K_STAR}({WIDE})/tau_{K_STAR}({NARROW}) >= {TAU_BAR:.2f}; "
          f"observed {fmt(a['tau_factor'])} -> {tri(a['tau_ok'])}")
    print(f"  needs predicted tok/s ratio >= {TOK_BAR:.2f}; "
          f"observed {fmt(a['tok_ratio'])} -> {tri(a['tok_ok'])}")
    if a["tau_needed"] is not None:
        print(f"  at the measured cost factor {fmt(a['cost_factor'])}, acceptance would have to "
              f"reach {fmt(a['tau_needed'])} to clear {TOK_BAR:.2f} "
              f"({TOK_BAR:.2f} / {fmt(a['cost_factor'])})")
    print(f"  verdict: {a['verdict']}")
    return a


# --- selftest -----------------------------------------------------------------


def synth(acc_wide: list[int], acc_narrow: list[int], draft=(0.010, 0.008),
          verify=(0.020, 0.010), ctx_len: int = 160) -> dict:
    """One synthetic accept.json: paired prompts, per-block timings, full tau map."""
    rows, timings, tau = [], [], {}
    for i, (aw, an) in enumerate(zip(acc_wide, acc_narrow)):
        for slot, (b, acc) in enumerate(((WIDE, aw), (NARROW, an))):
            rows.append({"prompt": i, "ctx_len": ctx_len, "block": b, "accepted": acc,
                         "drafts_matched": acc - 1, "agree": [True] * (acc - 1)})
            timings.append({"prompt": i, "block": b, "draft_s": draft[slot],
                            "verify_s": verify[slot], "columns": b})
    for vals, b in ((acc_wide, WIDE), (acc_narrow, NARROW)):
        for k in KS:
            if k <= b:
                tau[tau_key(b, k)] = statistics.fmean(min(a, k) for a in vals)
    return {"rows": rows, "tau": tau, "timings": timings, "config": {"synthetic": True}}


SUPPORTING = synth([12, 10, 14, 9, 11, 13, 15, 8], [6, 5, 7, 4, 6, 8, 5, 4])
FALSIFYING = synth([4, 5, 5, 4, 5, 5, 4, 5], [5, 5, 5, 5, 5, 5, 5, 5])


def selftest() -> None:
    cases = (("supporting", SUPPORTING, "SUPPORTED"), ("falsifying", FALSIFYING, "NOT SUPPORTED"))
    verdicts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, data, want in cases:
            path = Path(tmp, f"accept_{name}.json")
            path.write_text(json.dumps(data))
            got = analyse(load(path))
            assert got["verdict"] == want, (name, got["verdict"])
            assert got["paired_n"] == 8, (name, got["paired_n"])
            lo, hi = got["ci"]
            assert lo is not None and lo <= hi, (name, got["ci"])
            assert got["tok_ratio"] is not None
            verdicts[name] = got["verdict"]
    assert verdicts["supporting"] != verdicts["falsifying"]
    assert analyse(SUPPORTING)["ci"] == analyse(SUPPORTING)["ci"]  # deterministic seed
    assert analyse(SUPPORTING)["ci"][0] > 1.0                      # CI clears parity
    assert analyse({"rows": [], "tau": {}, "timings": []})["verdict"] == "no verdict"
    assert analyse({})["verdict"] == "no verdict"
    nothing = analyse({"rows": [], "tau": {}, "timings": []})
    assert nothing["tau_factor"] is None and nothing["tok_ratio"] is None
    print("SELFTEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLC table, paired contrast, cost model")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.path.exists():
        print(f"== vlc.py — {args.path}")
        print(f"{ABSENT}: no file at {args.path}")
        print("verdict: no verdict")
        return
    render(load(args.path), args.path)


if __name__ == "__main__":
    main()
