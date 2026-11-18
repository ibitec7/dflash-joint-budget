"""Three ablation figures, drawn only from measured `results/` and `results_det/` records.

Usage:
    uv run python scripts/figures.py

`results/` holds the default-kernel runs and `results_det/` the deterministic-kernel runs. Both are
read; nothing is computed across the two, because they are different measurements of the same grid
and their absolute tokens/s move by more than the effects being plotted.

What the figures may and may not claim:

* fig1 shows throughput against concurrency with default kernels solid and deterministic kernels
  dashed. The title states that distinction and claims no speedup.
* fig2 is a paired ratio, tok/s(arm) / tok/s(A5) at the same (benchmark, concurrency, seed, kernel
  mode). Pairing is what makes it readable at one seed, since A5 absorbs the machine's noise.
* fig3 is acceptance length tau against concurrency, one panel per kernel mode.

A series with no records is skipped and the missing filename is printed; nothing is interpolated.
When a series has more than one seed, the line follows the seed mean and every seed is drawn as its
own marker, so a mean is never shown on its own.

The A1 and A3 records are the same launched configuration (draft block 8 / 9 verify columns): one
SGLang flag sets both the drafter block and the verify width, so A1's declared "draft 16" never
reached the backend. See docs/RESULTS.md, "Correction (2026-11-19)". The labels say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"

# (mode name, directory). Order fixes legend and panel order.
MODES: tuple[tuple[str, Path], ...] = (
    ("default", ROOT / "results"),
    ("deterministic", ROOT / "results_det"),
)

ARMS = ("A0", "A1", "A3", "A5")
BENCHES = ("math500", "mtbench")
CONCURRENCIES = (1, 4, 8, 16, 32)
AR_REFERENCE = "A5"

# Short enough to fit a legend, specific enough not to repeat the retracted labels.
SHORT = {
    "A0": "block 16/17",
    "A1": "block 8/9",
    "A3": "block 8/9",
    "A5": "AR",
}

# Extra label displacement per arm, in points, so the two arms that share a configuration
# (A1 and A3) do not print identical tau text on top of each other.
LABEL_NUDGE = {"A0": 0, "A1": 0, "A3": 15, "A5": 0}

COLORS = {"A0": "#1f77b4", "A1": "#d62728", "A3": "#2ca02c", "A5": "#7f7f7f"}
MARKERS = {"A0": "o", "A1": "s", "A3": "^", "A5": "D"}
LINESTYLES = {"default": "-", "deterministic": "--"}
MODE_TEXT = {"default": "default kernels", "deterministic": "deterministic kernels"}

REQUIRED = {
    "arm", "benchmark", "concurrency", "seed", "draft_width", "verify_columns", "prompts",
    "tau", "tok_per_s", "ar_tok_per_s", "ms_per_round_p50", "ms_per_round_p95",
    "exact_match_ar", "graph_recaptures",
}


# --------------------------------------------------------------------------- loading


def load_records(dirs) -> tuple[dict[tuple, dict], dict[str, int]]:
    """Read every record from the given directories, keyed by (mode, bench, arm, c, seed)."""
    records: dict[tuple, dict] = {}
    read_counts = {mode: 0 for mode, _ in dirs}
    for mode, directory in dirs:
        if not directory.is_dir():
            print(f"absent: {directory}/ (directory missing)")
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skip: {path.relative_to(ROOT)} ({exc})")
                continue
            missing = REQUIRED.difference(data)
            if missing:
                print(f"skip: {path.relative_to(ROOT)} (missing keys: {sorted(missing)})")
                continue
            if not isinstance(data["tok_per_s"], (int, float)) or data["tok_per_s"] <= 0:
                print(f"skip: {path.relative_to(ROOT)} (unusable tok_per_s)")
                continue
            key = (
                mode,
                str(data["benchmark"]),
                str(data["arm"]),
                int(data["concurrency"]),
                int(data["seed"]),
            )
            records[key] = data
            read_counts[mode] += 1
    return records, read_counts


def series(records: dict[tuple, dict], mode: str, bench: str, arm: str) -> dict:
    """concurrency -> {seed: record} for one (mode, bench, arm)."""
    out: dict[int, dict[int, dict]] = {}
    for (m, b, a, c, s), rec in records.items():
        if m == mode and b == bench and a == arm:
            out.setdefault(c, {})[s] = rec
    return dict(sorted(out.items()))


def report_absence(records, mode: str, bench: str, arm: str, directory: Path) -> None:
    """Print the filename of every grid cell with no record; the caller then skips the series."""
    points = series(records, mode, bench, arm)
    if not points:
        print(f"absent: {directory.name}/{arm}_{bench}_c*_s*.json (no records for this series)")
        return
    seeds = sorted({s for by_seed in points.values() for s in by_seed})
    for conc in CONCURRENCIES:
        if conc in points:
            continue
        for seed in seeds:
            print(f"absent: {directory.name}/{arm}_{bench}_c{conc}_s{seed}.json")


def _tau_text(taus: list[float]) -> str:
    """The label itself is just the number; the axis note says these labels are tau."""
    if max(taus) - min(taus) < 5e-3:
        return f"{taus[0]:.2f}"
    return f"{taus[0]:.2f}-{taus[-1]:.2f}"


def _log2_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(CONCURRENCIES))
    ax.set_xticklabels([str(c) for c in CONCURRENCIES])
    ax.set_xlim(0.8, 40)
    ax.minorticks_off()
    ax.set_xlabel("concurrency (clients), log scale base 2")


# --------------------------------------------------------------------------- fig 1


def fig1(records, dirs, inv: dict[str, list[str]]) -> Path | None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    drawn = 0
    peak = 0.0
    for mode, directory in dirs:
        for arm in ARMS:
            report_absence(records, mode, "math500", arm, directory)
            points = series(records, mode, "math500", arm)
            if not points:
                continue
            xs, ys, labels = [], [], []
            for conc, by_seed in points.items():
                values = [rec["tok_per_s"] for rec in by_seed.values()]
                xs.append(conc)
                ys.append(sum(values) / len(values))
                taus = sorted(rec["tau"] for rec in by_seed.values())
                labels.append((conc, sum(values) / len(values), _tau_text(taus)))
                peak = max(peak, *values)
                for seed, rec in sorted(by_seed.items()):
                    ax.plot(
                        [conc], [rec["tok_per_s"]],
                        marker=MARKERS[arm], ms=4.6, linestyle="none",
                        mfc=COLORS[arm] if len(by_seed) == 1 else "none",
                        mec=COLORS[arm], mew=1.1, zorder=4,
                    )
            ax.plot(
                xs, ys, color=COLORS[arm], linestyle=LINESTYLES[mode], lw=1.5,
                label=f"{arm}: {SHORT[arm]}, {MODE_TEXT[mode]}", zorder=3,
            )
            # Rotated so labels of arms that sit at nearly the same throughput do not collide
            # horizontally; default-kernel points label upward, deterministic ones downward.
            upward = mode == "default"
            nudge = LABEL_NUDGE[arm]
            for conc, value, text in labels:
                ax.annotate(
                    text, (conc, value), textcoords="offset points",
                    xytext=(0, (3 + nudge) if upward else -(3 + nudge)),
                    ha="center", va="bottom" if upward else "top",
                    rotation=90, rotation_mode="anchor",
                    fontsize=5.6, color=COLORS[arm],
                    bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.75},
                    zorder=5,
                )
            inv["fig1"].append(f"{arm}/{mode} n={sum(len(v) for v in points.values())}")
            drawn += len(points)
    if not drawn:
        plt.close(fig)
        print("fig1_throughput_vs_concurrency: skipped, no math500 records")
        return None

    _log2_axis(ax)
    ax.set_ylim(0, peak * 1.12 if peak else None)
    ax.set_ylabel("tokens / s (within one kernel mode only)")
    ax.set_title(
        "math500 throughput vs concurrency: solid = default kernels, dashed = deterministic kernels\n"
        "Two separate measurement series; values are not comparable across modes",
        fontsize=10,
    )
    ax.grid(True, axis="y", color="0.9", lw=0.6)
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.01, -0.02,
        "Numbers beside each point are that point's acceptance length tau. A1 and A3 execute the "
        "same configuration (block 8 / 9 verify columns):\none SGLang flag sets both the drafter "
        "block and the verify width, so A1's declared draft 16 never reached the backend "
        "(docs/RESULTS.md,\nCorrection 2026-11-19). Markers are one per seed; the line is the "
        "seed mean.",
        fontsize=6.2, color="0.35", ha="left", va="top",
    )
    return _save(fig, "fig1_throughput_vs_concurrency")


# --------------------------------------------------------------------------- fig 2


def fig2(records, dirs, inv: dict[str, list[str]]) -> Path | None:
    default_dir = dict(dirs)["default"]
    panels = []
    for bench in BENCHES:
        has_arm = any(series(records, "default", bench, arm) for arm in ARMS)
        if not has_arm:
            print(f"absent: {default_dir.name}/*_{bench}_c*_s*.json (no records; panel skipped)")
            continue
        panels.append(bench)
    if not panels:
        print("fig2_paired_ratio: skipped, no default-kernel records")
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels) + 1.6, 4.3), squeeze=False,
    )
    for ax, bench in zip(axes[0], panels, strict=True):
        for arm in ARMS:
            report_absence(records, "default", bench, arm, default_dir)
            if arm == AR_REFERENCE:
                continue
            points = series(records, "default", bench, arm)
            reference = series(records, "default", bench, AR_REFERENCE)
            if not points or not reference:
                continue
            xs, ys, last = [], [], None
            for conc, by_seed in points.items():
                ratios = []
                for seed, rec in sorted(by_seed.items()):
                    ref = reference.get(conc, {}).get(seed)
                    if ref is None:
                        print(
                            f"absent: {default_dir.name}/{AR_REFERENCE}_{bench}_c{conc}_s{seed}.json"
                            f" (no pair for {arm})"
                        )
                        continue
                    ratio = rec["tok_per_s"] / ref["tok_per_s"]
                    ratios.append(ratio)
                    ax.plot(
                        [conc], [ratio], marker=MARKERS[arm], ms=4.6, linestyle="none",
                        mfc=COLORS[arm] if len(by_seed) == 1 else "none",
                        mec=COLORS[arm], mew=1.1, zorder=4,
                    )
                if not ratios:
                    continue
                xs.append(conc)
                ys.append(sum(ratios) / len(ratios))
                last = (conc, ys[-1])
            if xs:
                ax.plot(
                    xs, ys, color=COLORS[arm], lw=1.5,
                    label=f"{arm}: {SHORT[arm]}", zorder=3,
                )
                if last is not None:
                    ax.annotate(
                        f"{last[1]:.2f}x", last, textcoords="offset points",
                        xytext=(7, 6 if arm == "A3" else 2),
                        fontsize=6.5, color=COLORS[arm],
                    )
                inv["fig2"].append(
                    f"{arm}/{bench} n={sum(1 for _ in series(records, 'default', bench, arm))}"
                )

        ax.axhline(1.0, color="0.25", lw=1.0, ls=":", zorder=2)
        _log2_axis(ax)
        ax.set_title(f"{bench}, default kernels", fontsize=9)
        ax.grid(True, axis="y", color="0.9", lw=0.6)
        ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    axes[0][0].set_ylabel("tok/s(arm) / tok/s(A5) at the same (benchmark, c, seed)")

    fig.suptitle(
        "Paired ratio to autoregressive decode (A5) at equal concurrency; dotted line = parity (1.0)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig2_paired_ratio")


# --------------------------------------------------------------------------- fig 3


def fig3(records, dirs, inv: dict[str, list[str]]) -> Path | None:
    panels = [(mode, directory) for mode, directory in dirs
              if any(series(records, mode, "math500", arm) for arm in ARMS)]
    if not panels:
        print("fig3_tau_vs_concurrency: skipped, no math500 records in either mode")
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.3 * len(panels) + 1.4, 4.2), squeeze=False, sharey=True,
    )
    for ax, (mode, directory) in zip(axes[0], panels, strict=True):
        for arm in ARMS:
            report_absence(records, mode, "math500", arm, directory)
            points = series(records, mode, "math500", arm)
            if not points:
                continue
            xs, ys = [], []
            for conc, by_seed in points.items():
                taus = [rec["tau"] for rec in sorted(by_seed.values(), key=lambda r: r["seed"])]
                xs.append(conc)
                ys.append(sum(taus) / len(taus))
                ax.plot(
                    [conc] * len(taus), taus, marker=MARKERS[arm], ms=4.4, linestyle="none",
                    mfc=COLORS[arm] if len(taus) == 1 else "none",
                    mec=COLORS[arm], mew=1.1, zorder=4,
                )
            ax.plot(
                xs, ys, color=COLORS[arm], linestyle=LINESTYLES[mode], lw=1.5,
                label=f"{arm}: {SHORT[arm]}", zorder=3,
            )
            inv["fig3"].append(f"{arm}/{mode} n={sum(len(v) for v in points.values())}")
        _log2_axis(ax)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("acceptance length tau (tokens per round)")
        ax.set_title(MODE_TEXT[mode], fontsize=9.5)
        ax.grid(True, axis="y", color="0.9", lw=0.6)
        ax.legend(fontsize=6.8, frameon=False, loc="lower left")

    fig.suptitle(
        "Acceptance length vs concurrency, math500, one panel per kernel mode (separate runs)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig3_tau_vs_concurrency")


# --------------------------------------------------------------------------- output


def _save(fig, stem: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {pdf.relative_to(ROOT)}, {png.relative_to(ROOT)}")
    return png


def main() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    })
    records, read_counts = load_records(MODES)
    inv: dict[str, list[str]] = {"fig1": [], "fig2": [], "fig3": []}

    fig1(records, MODES, inv)
    fig2(records, MODES, inv)
    fig3(records, MODES, inv)

    counts = ", ".join(f"{mode}={n}" for mode, n in read_counts.items())
    parts = [f"{name}: " + ("; ".join(inv[name]) if inv[name] else "none") for name in inv]
    print(
        f"INVENTORY files_read={sum(read_counts.values())} ({counts}) | "
        + " | ".join(parts)
    )


if __name__ == "__main__":
    main()
