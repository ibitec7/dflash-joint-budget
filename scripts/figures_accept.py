"""Acceptance-surface figures from the standalone acceptance harness.

Usage:
    uv run python scripts/figures_accept.py
    uv run python scripts/figures_accept.py --math500 experiments/standalone/accept_math500_sweep.json

Each input file is the JSON the harness writes: ``{"tau": {"b<block>_k<k>": float}, "rows": [...],
"timings": [...], "config": {...}}``. Nothing here re-derives the ``tau`` table; the cells are read as
measured. ``fig4`` therefore plots only cells that exist in the file, and a block whose own drafted
width is smaller than the verify window is not a missing cell, it is a structural absence (block 8
cannot deliver 9 verify columns) and the series simply terminates there.

What the figures may claim:

* fig4 is tau (accepted tokens per round) against drafter block width b, one line per verify window
  k in {5, 8, 9, 12, 16}. The lines are measurements on a shared grid, not a fitted curve: no line is
  smoothed, and no line is extended past the largest block that can still deliver its k. Because
  k <= b is a hard constraint, the k=12 and k=16 lines can only start at b=12 and b=16, and the
  only like-for-like cross-block comparison is at k <= 8. The title says so instead of quoting a
  ratio that mixes two different k values.
* fig6 pairs prompts: for each prompt, tau_9 = mean over that prompt's rounds of min(accepted, 9),
  once from block 16 and once from block 9. Points are one per prompt, the diagonal is parity, and
  the subtitle counts wins/losses/ties. The footer carries the paired ratio with a bootstrap CI over
  prompts. Per-prompt means weight every prompt equally; the tau table weights every round equally,
  so the two figures do not share an absolute level (see the footer note).

Every series that is skipped, every cell that is absent, and every k value present in the file but
not in the requested window set is printed with its key. Nothing is interpolated, and a gap in the
measurements is drawn as a gap.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"
STANDALONE = ROOT / "experiments" / "standalone"

# Verify windows the scientific claim is about. A file may contain a finer sweep; those extra k
# values are reported as unplotted rather than silently dropped (and never drawn).
K_WINDOWS = (5, 8, 9, 12, 16)

KEY_RE = re.compile(r"^b(\d+)_k(\d+)$")

# (panel label, candidate files, in preference order). The first file that exists is used; the
# finer sweep supersedes the coarse file for the same benchmark so a block is never drawn twice.
DATASETS: tuple[tuple[str, tuple[Path, ...]], ...] = (
    ("math500", (STANDALONE / "accept_math500_sweep.json", STANDALONE / "accept_math500.json")),
    ("mtbench", (STANDALONE / "accept_mtbench_sweep.json", STANDALONE / "accept_mtbench.json")),
)

# The pre-registered cells: the verdict is read at (b=16, k=9) against the (b=9, k=9) control.
VERDICT_BLOCK, VERDICT_K = 16, 9
CONTROL_BLOCK, CONTROL_K = 9, 9

COL_K = {5: "#7f7f7f", 8: "#1f77b4", 9: "#d62728", 12: "#2ca02c", 16: "#9467bd"}
MARK_K = {5: "o", 8: "s", 9: "^", 12: "D", 16: "P"}

RNG_SEED = 0
N_BOOT = 4000


@dataclass
class Dataset:
    """One harness JSON, reduced to the two things the figures read."""

    label: str
    path: Path
    tau: dict[tuple[int, int], float] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    unplotted_k: set[int] = field(default_factory=set)

    @property
    def blocks(self) -> list[int]:
        return sorted({b for b, _ in self.tau} | {int(r["block"]) for r in self.rows})

    def rel(self) -> str:
        try:
            return str(self.path.relative_to(ROOT))
        except ValueError:
            return str(self.path)


# --------------------------------------------------------------------------- loading


def _rel(path: Path) -> str:
    """Path relative to the repo when it lives inside it, absolute otherwise (CLI can pass /tmp)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_dataset(label: str, candidates: tuple[Path, ...], inv: dict[str, list[str]]) -> Dataset | None:
    """First existing candidate, parsed. Prints why a file or a cell was skipped."""
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        names = " or ".join(_rel(p) for p in candidates)
        print(f"absent: {names} (no file; {label} panel skipped)")
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"skip: {_rel(path)} ({exc})")
        return None
    if not isinstance(data, dict):
        print(f"skip: {_rel(path)} (top level is {type(data).__name__}, not an object)")
        return None

    ds = Dataset(label=label, path=path)
    raw_tau = data.get("tau", {})
    if not isinstance(raw_tau, dict):
        print(f"skip: {_rel(path)} tau is {type(raw_tau).__name__}, not an object")
        raw_tau = {}
    for key, value in raw_tau.items():
        m = KEY_RE.match(str(key))
        if m is None:
            print(f"skip: {_rel(path)} tau key {key!r} (not b<block>_k<k>)")
            continue
        b, k = int(m.group(1)), int(m.group(2))
        if not isinstance(value, (int, float)) or value <= 0:
            print(f"skip: {_rel(path)} {key} (unusable tau {value!r})")
            continue
        if k > b:
            print(f"skip: {_rel(path)} {key} (k exceeds the block's own column count)")
            continue
        ds.tau[(b, k)] = float(value)
        if k not in K_WINDOWS:
            ds.unplotted_k.add(k)

    rows = data.get("rows", [])
    if isinstance(rows, list):
        for r in rows:
            try:
                b = int(r["block"])
                float(r["accepted"])
                int(r["prompt"])
            except (KeyError, TypeError, ValueError):
                print(f"skip: {_rel(path)} one rows entry (needs block/accepted/prompt)")
                continue
            ds.rows.append(r)
    elif rows:
        print(f"skip: {_rel(path)} rows is {type(rows).__name__}, not a list")

    if not ds.tau and not ds.rows:
        print(f"skip: {_rel(path)} (no usable tau cells and no usable rows)")
        return None
    inv["files"].append(f"{label}={ds.rel()} blocks={ds.blocks}")
    return ds


def missing_cells(ds: Dataset, k_windows) -> list[tuple[int, int]]:
    """(block, k) cells the figure needs, that the file does not have, for blocks that can host k."""
    out = []
    for k in k_windows:
        for b in ds.blocks:
            if b < k:
                continue
            if (b, k) not in ds.tau:
                out.append((b, k))
    return out


# --------------------------------------------------------------------------- shared style


def _save(fig, stem: str) -> tuple[Path, Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")
    return pdf, png


def _cell_text(k: int) -> str:
    return f"k={k}"


# --------------------------------------------------------------------------- fig 4


def fig4(datasets: list[Dataset], inv: dict[str, list[str]]) -> None:
    panels = [ds for ds in datasets if ds.tau]
    for ds in datasets:
        if not ds.tau:
            print(f"absent: {ds.rel()} tau (no measured cells; fig4 panel skipped)")
    if not panels:
        print("fig4_tau_vs_block: skipped, no tau cells in any input")
        return

    nrow = len(panels)
    fig, axes = plt.subplots(
        nrow, 1, figsize=(7.4, 4.5 * nrow + 0.5), squeeze=False, sharex=True,
    )
    for ax, ds in zip(axes[:, 0], panels, strict=True):
        peak = 0.0
        for k in K_WINDOWS:
            pts = [(b, ds.tau[(b, k)]) for b in ds.blocks if (b, k) in ds.tau and b >= k]
            if not pts:
                print(f"absent: {ds.rel()} no b*_k{k} cell at any block (line {_cell_text(k)} skipped)")
                continue
            pts.sort()
            # Split into contiguous runs so a missing cell is a gap, never a straight segment.
            runs: list[list[tuple[int, float]]] = [[pts[0]]]
            for prev, cur in itertools.pairwise(pts):
                gap = any((b, k) not in ds.tau for b in ds.blocks if prev[0] < b < cur[0])
                if gap:
                    runs.append([cur])
                else:
                    runs[-1].append(cur)
            color, marker = COL_K[k], MARK_K[k]
            for i, run in enumerate(runs):
                xs = [p[0] for p in run]
                ys = [p[1] for p in run]
                if len(run) > 1:
                    ax.plot(
                        xs, ys, color=color, lw=1.5, ls="-", zorder=3,
                        label=_cell_text(k) if i == 0 else None,
                    )
                ax.plot(
                    xs, ys, marker=marker, ms=5.0, linestyle="none", mfc="white",
                    mec=color, mew=1.3, zorder=4,
                    label=_cell_text(k) if len(run) == 1 and i == 0 else None,
                )
            peak = max(peak, *(v for _, v in pts))
            end_b, end_v = pts[-1]
            ax.annotate(
                f"{end_v:.2f}", (end_b, end_v), textcoords="offset points", xytext=(5, 4),
                fontsize=6.4, color=color, zorder=5,
                bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.75},
            )
            inv["fig4"].append(
                f"{ds.label} k={k} b={[p[0] for p in pts]} "
                f"n={len(pts)} y={pts[0][1]:.3f}..{pts[-1][1]:.3f}"
            )
            if len(pts) == 1:
                inv["fig4"].append(
                    f"{ds.label} k={k} single measured block (no line: a segment needs 2 cells)"
                )

        # Structural absences are inventoried once per panel, not as a fake missing cell per key.
        for k in K_WINDOWS:
            too_wide = [b for b in ds.blocks if b < k]
            if too_wide:
                print(
                    f"n/a: {ds.rel()} k={k} at b={too_wide} (block b cannot deliver k>b; "
                    "series terminates there, no clamp, no extrapolation)"
                )
        if ds.unplotted_k:
            extra = ",".join(str(k) for k in sorted(ds.unplotted_k))
            print(
                f"unplotted: {ds.rel()} k={{{extra}}} present but outside the drawn windows "
                f"{list(K_WINDOWS)} (kept out of fig4; no series merged across k)"
            )

        # Annotation of the two pre-registered cells, when the file has them.
        v = ds.tau.get((VERDICT_BLOCK, VERDICT_K))
        c = ds.tau.get((CONTROL_BLOCK, CONTROL_K))
        if (VERDICT_BLOCK, VERDICT_K) in ds.tau:
            ax.annotate(
                f"verdict cell: b={VERDICT_BLOCK}, k={VERDICT_K}\n(truncated at {VERDICT_K} columns)",
                xy=(VERDICT_BLOCK, v), xytext=(-8, 16), textcoords="offset points",
                fontsize=6.3, ha="right", va="bottom", color="#111111",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#333333",
                            "shrinkA": 2, "shrinkB": 4},
                bbox={"boxstyle": "round,pad=0.25", "fc": "#fff8e1", "ec": "#b8860b", "lw": 0.7},
                zorder=6,
            )
        if (CONTROL_BLOCK, CONTROL_K) in ds.tau:
            ax.annotate(
                f"control: b={CONTROL_BLOCK}, k={CONTROL_K}\ntau={c:.2f}",
                xy=(CONTROL_BLOCK, c), xytext=(-9, 14), textcoords="offset points",
                fontsize=6.3, ha="right", va="bottom", color="#111111",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#333333",
                            "shrinkA": 2, "shrinkB": 4},
                zorder=6,
            )
        if (VERDICT_BLOCK, VERDICT_BLOCK) in ds.tau:
            ax.annotate(
                f"k={VERDICT_BLOCK} exists only at b={VERDICT_BLOCK}\n(no other block drafts that many)",
                xy=(VERDICT_BLOCK, ds.tau[(VERDICT_BLOCK, VERDICT_BLOCK)]),
                xytext=(-8, 8), textcoords="offset points",
                fontsize=6.0, ha="right", va="bottom", color=COL_K[VERDICT_BLOCK],
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COL_K[VERDICT_BLOCK],
                            "shrinkA": 2, "shrinkB": 5},
                zorder=6,
            )
        if not (VERDICT_BLOCK, VERDICT_K) in ds.tau or not (CONTROL_BLOCK, CONTROL_K) in ds.tau:
            print(
                f"absent: {ds.rel()} verdict/control annotation incomplete "
                f"(b{VERDICT_BLOCK}_k{VERDICT_K}={v}, b{CONTROL_BLOCK}_k{CONTROL_K}={c})"
            )

        ax.set_ylim(0, peak * 1.18 if peak else None)
        ax.set_xlim(min(ds.blocks) - 1.6, max(ds.blocks) + 1.0)
        ax.set_ylabel("acceptance length tau (accepted tokens per round)")
        ax.set_title(f"{ds.label}: measured cells only, one line per verify window", fontsize=9.5)
        ax.grid(True, axis="y", color="0.9", lw=0.6)
        ax.legend(fontsize=7.0, frameon=False, loc="upper left", title="verify window k",
                  title_fontsize=7.0)
        ax.margins(x=0.06)

    axes[-1, 0].set_xlabel("drafter block width b (tokens drafted per round)")
    axes[-1, 0].set_xticks(sorted({b for ds in panels for b in ds.blocks}))
    fig.suptitle(
        "tau vs drafter block width: below the verify window the width does nothing,\n"
        "and k>=12 exists only where the block can host it (k<=b), so no cross-block ratio is "
        "available at k=16",
        fontsize=10,
    )
    fig.text(
        0.01, -0.01,
        "tau_k = mean over rounds of min(accepted prefix, k), k <= b, read directly from the harness "
        "tau table: no interpolation, no smoothing, no\nclamping of k past b. A line stops at the "
        "largest block that can deliver its k, and a missing measured cell is drawn as a gap "
        "(printed\nabove). k=16 is measured at b=16 only, so the 5.28-vs-4.20 comparison is k=16 "
        "against k=8, not two widths at equal k.",
        fontsize=6.2, color="0.35", ha="left", va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig4_tau_vs_block")


# --------------------------------------------------------------------------- fig 6


def per_prompt_tau9(ds: Dataset, block: int) -> dict[int, tuple[float, int]]:
    """prompt -> (mean over its rounds of min(accepted, 9), number of rounds)."""
    acc: dict[int, list[int]] = {}
    for r in ds.rows:
        if int(r["block"]) != block:
            continue
        acc.setdefault(int(r["prompt"]), []).append(min(int(r["accepted"]), 9))
    return {p: (statistics.fmean(v), len(v)) for p, v in acc.items()}


def _bootstrap_ratio(xs: list[float], ys: list[float], n: int = N_BOOT, seed: int = RNG_SEED):
    """Percentile CI for sum(y)/sum(x) under paired resampling of prompts."""
    rng = random.Random(seed)
    idx = range(len(xs))
    ratios = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        sx = sum(xs[i] for i in pick)
        sy = sum(ys[i] for i in pick)
        if sx > 0:
            ratios.append(sy / sx)
    ratios.sort()
    if not ratios:
        return (float("nan"), float("nan"))
    lo = ratios[int(0.025 * (len(ratios) - 1))]
    hi = ratios[int(0.975 * (len(ratios) - 1))]
    return (lo, hi)


def fig6(datasets: list[Dataset], inv: dict[str, list[str]]) -> None:
    panels = [ds for ds in datasets if ds.rows]
    for ds in datasets:
        if not ds.rows:
            print(f"absent: {ds.rel()} rows (no per-prompt rows; fig6 panel skipped)")
    if not panels:
        print("fig6_paired_scatter: skipped, no per-prompt rows in any input")
        return

    nrow = len(panels)
    fig, axes = plt.subplots(nrow, 1, figsize=(5.4, 4.6 * nrow + 0.4), squeeze=False)
    for ax, ds in zip(axes[:, 0], panels, strict=True):
        hi_block, lo_block = VERDICT_BLOCK, CONTROL_BLOCK
        ymap = per_prompt_tau9(ds, hi_block)
        xmap = per_prompt_tau9(ds, lo_block)
        for b, m in ((hi_block, ymap), (lo_block, xmap)):
            if not m:
                print(f"absent: {ds.rel()} rows with block={b} (fig6 needs both block {lo_block} "
                      f"and block {hi_block}; pair cannot be formed)")
        prompts = sorted(set(xmap) & set(ymap))
        only_x = sorted(set(xmap) - set(ymap))
        only_y = sorted(set(ymap) - set(xmap))
        for p in only_x:
            print(f"absent: {ds.rel()} rows prompt={p} block={hi_block} (no pair; prompt dropped)")
        for p in only_y:
            print(f"absent: {ds.rel()} rows prompt={p} block={lo_block} (no pair; prompt dropped)")
        if not prompts:
            print(f"absent: {ds.rel()} no paired prompts; fig6 panel skipped")
            ax.set_axis_off()
            continue

        xs = [xmap[p][0] for p in prompts]
        ys = [ymap[p][0] for p in prompts]
        wins = sum(1 for x, y in zip(xs, ys, strict=True) if y > x + 1e-12)
        losses = sum(1 for x, y in zip(xs, ys, strict=True) if y < x - 1e-12)
        ties = len(prompts) - wins - losses
        ratio = sum(ys) / sum(xs) if sum(xs) else float("nan")
        lo, hi = _bootstrap_ratio(xs, ys)

        top = max(max(xs), max(ys)) * 1.06
        ax.plot([0, top], [0, top], color="0.3", lw=1.0, ls=":", zorder=2, label="parity (y = x)")
        for x, y in zip(xs, ys, strict=True):
            color = "#2ca02c" if y > x + 1e-12 else ("#d62728" if y < x - 1e-12 else "0.5")
            ax.plot([x], [y], marker="o", ms=4.2, linestyle="none", mfc="none",
                    mec=color, mew=1.2, zorder=4)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal", adjustable="box")

        rounds_x = sum(m[1] for m in xmap.values())
        rounds_y = sum(m[1] for m in ymap.values())
        ax.set_title(
            f"{ds.label}: tau_9 at block {hi_block} vs block {lo_block}, one point per prompt\n"
            f"n={len(prompts)} prompts; block {hi_block} wins {wins}, loses {losses}, ties {ties}",
            fontsize=9.0,
        )
        ax.set_xlabel(f"tau_9 at block {lo_block} (accepted tokens per round, prompt mean)")
        ax.set_ylabel(f"tau_9 at block {hi_block} (accepted tokens per round, prompt mean)")
        ax.grid(True, color="0.92", lw=0.6)
        ax.legend(fontsize=7.0, frameon=False, loc="upper left")

        fig.text(
            0.01, -0.01 - 0.02 * (nrow - 1),
            f"Paired ratio tau_9(block {hi_block}) / tau_9(block {lo_block}) = {ratio:.3f} "
            f"[95% CI {lo:.3f}, {hi:.3f}], bootstrap over prompts, "
            f"{N_BOOT} resamples, seed {RNG_SEED}.\n"
            f"Each point is mean over that prompt's own rounds (block {lo_block}: {rounds_x} rounds, "
            f"block {hi_block}: {rounds_y} rounds), so prompts\nare weighted equally here; the tau "
            "table in fig4 weights rounds equally, which is why the pooled levels differ.\n"
            "Green = block 16 higher, red = lower, grey dot = tie. Identity line is parity, not a "
            "fit; no regression is drawn through the points.",
            fontsize=6.2, color="0.35", ha="left", va="top",
        )
        inv["fig6"].append(
            f"{ds.label} n_prompts={len(prompts)} win/loss/tie={wins}/{losses}/{ties} "
            f"ratio={ratio:.4f} ci=[{lo:.4f},{hi:.4f}] mean_x={statistics.fmean(xs):.4f} "
            f"mean_y={statistics.fmean(ys):.4f}"
        )

    fig.suptitle(
        "Per-prompt paired acceptance at a 9-column verify window: does the wider drafter block "
        "buy anything?",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "fig6_paired_scatter")


# --------------------------------------------------------------------------- audit


def audit(datasets: list[Dataset]) -> None:
    """Print the comparisons the caption wants, next to whether the data can supply them."""
    for ds in datasets:
        if not ds.tau:
            continue
        print(f"AUDIT {ds.label} ({ds.rel()})")
        blocks = ds.blocks
        for k in K_WINDOWS:
            cells = [(b, ds.tau[(b, k)]) for b in blocks if (b, k) in ds.tau]
            if len(cells) >= 2:
                lo_b, lo_v = cells[0]
                hi_b, hi_v = cells[-1]
                print(
                    f"AUDIT   k={k}: like-for-like b{lo_b}={lo_v:.3f} vs b{hi_b}={hi_v:.3f} "
                    f"({(hi_v / lo_v - 1) * 100:+.1f}%)"
                )
            elif cells:
                print(f"AUDIT   k={k}: only b{cells[0][0]}={cells[0][1]:.3f}; no equal-k pair")
        wide = ds.tau.get((16, 16))
        small = ds.tau.get((8, 8))
        if wide is not None and small is not None:
            print(
                f"AUDIT   cross-k (NOT like-for-like): tau_16(b16)={wide:.3f} vs "
                f"tau_8(b8)={small:.3f} = {(wide / small - 1) * 100:+.1f}%; the '25.9%' caption "
                "compares two different k values"
            )
        ratio = ds.tau.get((16, 9)), ds.tau.get((9, 9))
        if all(v is not None for v in ratio):
            r = ratio[0] / ratio[1] - 1
            print(
                f"AUDIT   verdict cell tau_9(b16)={ratio[0]:.3f} vs tau_9(b9)={ratio[1]:.3f} "
                f"= {r * 100:+.2f}% (pre-registered bar is +5%)"
            )


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--math500", type=Path, default=None,
        help="override the math500 JSON (default: the sweep file if present, else accept_math500.json)",
    )
    parser.add_argument(
        "--mtbench", type=Path, default=None,
        help="override the mtbench JSON (default: the sweep file if present, else the coarse file)",
    )
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    })

    inv: dict[str, list[str]] = {"files": [], "fig4": [], "fig6": []}
    datasets: list[Dataset] = []
    for label, candidates in DATASETS:
        override = {"math500": args.math500, "mtbench": args.mtbench}.get(label)
        paths = (override,) if override is not None else candidates
        ds = load_dataset(label, paths, inv)
        if ds is not None:
            datasets.append(ds)
        else:
            print(f"absent: {label} (row skipped in both figures)")

    for ds in datasets:
        for b, k in missing_cells(ds, K_WINDOWS):
            print(f"absent: {ds.rel()} b{b}_k{k} (block {b} can host k={k} but the file has no cell)")

    if not datasets:
        print("INVENTORY files_read=0 | fig4: none | fig6: none")
        return

    audit(datasets)
    fig4(datasets, inv)
    fig6(datasets, inv)

    parts = [f"{name}: " + ("; ".join(inv[name]) if inv[name] else "none") for name in inv]
    print(f"INVENTORY files_read={len(inv['files'])} ({'; '.join(inv['files'])}) | " + " | ".join(parts))
    # Rows exist only where the harness stored them; a tau-only file renders fig4 and skips fig6.
    if not any(ds.rows for ds in datasets):
        print("INVENTORY fig6: no per-prompt rows in any input (scatter not drawn)")


if __name__ == "__main__":
    main()
