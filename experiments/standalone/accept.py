"""Acceptance accounting for the verify-length experiment (CPU only, no torch).

One verify pass over ``prefix + drafts`` yields a per-position agreement vector, so every
truncation width k is readable from the same pass: ``tau_k = E[min(a, k)]`` where ``a`` is the
longest accepted prefix. That is what lets VLC(b=16, k=9) be compared against block-9 on
identical contexts instead of on two server configurations.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

__all__ = ["agreement", "tau_table", "load_rows", "summarise"]


def agreement(drafts: list[int], verified: list[int]) -> list[bool]:
    """Per-position agreement; both lists are the k candidate tokens, in order."""
    if len(drafts) != len(verified):
        raise ValueError(f"length mismatch: {len(drafts)} drafts vs {len(verified)} verified")
    return [d == v for d, v in zip(drafts, verified)]


def accepted_prefix(agree: list[bool]) -> int:
    """Length of the longest accepted prefix. Greedy: the first disagreement stops acceptance."""
    n = 0
    for ok in agree:
        if not ok:
            break
        n += 1
    return n


def tau_table(rows: list[dict], widths: list[int]) -> dict[tuple[int, int], float]:
    """(block, k) -> mean min(accepted prefix, k) over all rows with that block.

    Each row is one ``prefix + drafts(block)`` measurement, so ``k`` may be any width up to the
    block. A width above the block is absent rather than clamped: extrapolating past the drafted
    tokens would invent data.
    """
    out: dict[tuple[int, int], float] = {}
    for b in {r["block"] for r in rows}:
        sub = [r for r in rows if r["block"] == b]
        for k in widths:
            if k > b:
                continue
            out[(b, k)] = statistics.fmean(min(r["accepted"], k) for r in sub)
    return out


def summarise(rows: list[dict], k: int = 9) -> str:
    """The H4-VLC acceptance comparison, as text a report can quote."""
    if not rows:
        return "no rows"
    table = tau_table(rows, widths=[k])
    blocks = sorted({r["block"] for r in rows})
    brief = " ".join(f"b{b}={table[(b, k)]:.2f}" for b in blocks if (b, k) in table)
    if 16 not in blocks or k not in [kk for (bb, kk) in table if bb == 16]:
        return f"k={k}: {brief} (no block-16 row at this width)"
    base = min((b for b in blocks if b != 16 and (b, k) in table), default=None)
    if base is None:
        return f"k={k}: {brief}"
    ratio = table[(16, k)] / table[(base, k)]
    verdict = "WIN" if ratio >= 1.05 else "LOSE"
    return f"k={k}: {brief} | tau_{k}(16)/tau_{k}(b{base}) = {ratio:.3f} {verdict} (bar 1.05)"


def load_rows(path: str | Path) -> list[dict]:
    """Rows from one JSON file or every *.json under a directory."""
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    rows: list[dict] = []
    for f in files:
        data = json.loads(f.read_text())
        rows.extend(data if isinstance(data, list) else data.get("rows", []))
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=9)
    a = ap.parse_args()
    print(summarise(load_rows(a.path), a.k))
