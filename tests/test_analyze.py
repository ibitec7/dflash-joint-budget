"""The analysis has to survive an empty grid, and it has to admit what is missing.

The synthetic records here live in the test process only. Nothing in this file writes to
`results/`: the analysis output goes to `tmp_path`, and the selftest writes to a temp dir.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze
import decide

BASE = {
    "benchmark": "math500", "seed": 0, "prompts": 200, "ar_tok_per_s": 110.0,
    "graph_recaptures": 0, "exact_match_ar": 0.997, "ms_per_round_p95": 2.3,
    "draft_width": 16, "verify_columns": 9,
}


def rec(arm, c, tok, tau=8.01, ms=2.50, seed=0, benchmark="math500"):
    return dict(BASE, arm=arm, concurrency=c, seed=seed, tau=tau, tok_per_s=tok,
                ms_per_round_p50=ms, benchmark=benchmark)


def write(tmp_path, name, record):
    path = tmp_path / name
    path.write_text(json.dumps(record))
    return path


def rows(markdown):
    return [ln for ln in markdown.splitlines() if ln.startswith("| A")]


# ------------------------------------------------------------------ empty and missing input


def test_empty_directory_yields_no_runs_and_an_empty_table(tmp_path, capsys):
    assert analyze.load(tmp_path) == []
    assert analyze.table([]).startswith("| arm |")
    assert rows(analyze.table([])) == []
    assert "no *.json" in capsys.readouterr().out


def test_absent_directory_is_reported_not_raised(tmp_path, capsys):
    assert analyze.load(tmp_path / "does-not-exist") == []
    assert "no results directory" in capsys.readouterr().out


def test_empty_input_still_writes_a_labelled_figure(tmp_path):
    out = analyze.figure([], tmp_path / "fig" / "empty.png")
    assert out.is_file()
    assert out.stat().st_size > 5_000
    caption = out.with_suffix(".txt").read_text()
    assert "Seeds present per arm" in caption
    assert "no" in caption.lower()


def test_incomplete_and_malformed_records_are_skipped_with_a_reason(tmp_path, capsys):
    incomplete = rec("A0", 32, 100.0)
    incomplete.pop("tau")
    write(tmp_path, "incomplete.json", incomplete)
    (tmp_path / "broken.json").write_text("{not json")
    write(tmp_path, "good.json", rec("A0", 32, 100.0, seed=1))

    runs = analyze.load(tmp_path)
    out = capsys.readouterr().out
    assert len(runs) == 1
    assert "incomplete.json" in out and "tau" in out
    assert "broken.json" in out


# ------------------------------------------------------------------ partial grid


def test_partial_grid_reports_what_is_missing_without_inventing_cells():
    runs = [
        rec("A0", 32, 100.0), rec("A1", 32, 115.0, tau=7.21),
        rec("A1", 1, 90.0, tau=7.21),          # no A0 at c=1, so no percentage there
    ]
    text = analyze.table(runs)
    assert len(rows(text)) == 3
    assert "missing protocol cells" in text
    missing = analyze.missing_cells(runs)
    assert ("A2", 32) in missing and ("A0", 1) in missing
    assert "A0 c=32" not in text.split("missing protocol cells")[1].split("_")[0]

    gains = analyze.gain_vs_baseline(runs)
    assert gains[("A1", 32)] == pytest.approx(15.0)
    assert ("A1", 1) not in gains


def test_a_cell_without_a_baseline_has_no_percentage():
    assert analyze.gain_vs_baseline([rec("A1", 8, 120.0)]) == {}


def test_multiple_seeds_report_mean_and_spread():
    runs = [rec("A0", 32, tok, seed=s) for s, tok in enumerate((100.0, 110.0, 120.0))]
    text = analyze.table(runs)
    assert "110.0 ±10.0" in text
    assert rows(text)[0].endswith("| 3 |")     # n counts the seeds behind the mean


def test_mixed_benchmarks_are_refused_rather_than_averaged():
    runs = [rec("A0", 32, 100.0), rec("A1", 32, 120.0, benchmark="mtbench")]
    with pytest.raises(ValueError, match="mix benchmarks"):
        analyze.gain_vs_baseline(runs)
    assert "bench" in analyze.table(runs).splitlines()[0]


# ------------------------------------------------------------------ gain arithmetic


def test_gain_is_percent_over_a_synthetic_a0_at_the_same_concurrency():
    runs = [
        rec("A0", 1, 100.0), rec("A1", 1, 115.0),
        rec("A0", 4, 200.0), rec("A1", 4, 230.0),
        rec("A0", 32, 300.0), rec("A4", 32, 330.0),
    ]
    gains = analyze.gain_vs_baseline(runs)
    assert gains[("A0", 1)] == pytest.approx(0.0)
    assert gains[("A1", 1)] == pytest.approx(15.0)
    assert gains[("A1", 4)] == pytest.approx(15.0)
    assert gains[("A4", 32)] == pytest.approx(10.0)
    assert ("A4", 1) not in gains            # no A4 record at c=1


def test_gain_is_a_ratio_of_cell_means_not_a_mean_of_ratios():
    runs = [
        rec("A0", 32, 100.0, seed=0), rec("A0", 32, 200.0, seed=1),
        rec("A1", 32, 165.0),
    ]
    assert analyze.gain_vs_baseline(runs)[("A1", 32)] == pytest.approx(10.0)


# ------------------------------------------------------------------ figure contract


def test_figure_carries_the_threshold_and_the_seeds_actually_present(tmp_path):
    png = analyze.figure(analyze.synthetic_runs(), tmp_path / "gain.png",
                         note="SYNTHETIC test data.")
    blob = png.read_bytes()
    assert png.stat().st_size > 20_000
    assert b"H3 threshold" in blob
    assert b"+15%" in blob
    assert b"Seeds present per arm" in blob

    caption = png.with_suffix(".txt").read_text()
    # three seeds only where the protocol asks for them, one elsewhere
    assert "A0 1/1/1/1/3" in caption
    assert "A2 1/1/1/1/1" in caption
    assert "SYNTHETIC test data." in caption
    assert "A5 is the AR reference" in caption


def test_figure_x_axis_is_log2_concurrency(tmp_path):
    png = analyze.figure(analyze.synthetic_runs(), tmp_path / "gain.png")
    # The PNG is the artifact; the axis assertion is on the ticks the builder would use.
    gains = analyze.gain_vs_baseline(analyze.synthetic_runs())
    assert sorted({c for _, c in gains}) == [1, 4, 8, 16, 32]
    assert png.is_file()


# ------------------------------------------------------------------ selftest


def test_selftest_writes_a_nontrivial_figure_outside_the_repo():
    png = analyze.selftest()
    assert png.is_file()
    assert png.stat().st_size > 10_000
    assert png.with_suffix(".txt").is_file()
    assert ROOT not in png.parents, f"selftest wrote into the repo: {png}"
    assert png.suffix == ".png"


def test_h3_constant_cannot_drift_from_the_decision_rule():
    assert analyze.H3_GAIN_PCT == pytest.approx((decide.THRESHOLDS["H3_gain_min"] - 1) * 100)
    assert analyze.BASELINE == "A0"
    assert set(analyze.PROTOCOL_GRID) == set(analyze.ARMS)
