"""The decision rules are the project's referee; they are tested like code."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import decide

BASE = {
    "benchmark": "math500", "seed": 0, "prompts": 200, "ar_tok_per_s": 110.0,
    "graph_recaptures": 0, "exact_match_ar": 0.997, "ms_per_round_p95": 2.3,
    "draft_width": 16, "verify_columns": 9,
}


def run(arm, c, tau, tok, ms, **kw):
    return dict(BASE, arm=arm, concurrency=c, tau=tau, tok_per_s=tok, ms_per_round_p50=ms, **kw)


GOOD = [
    run("A0", 32, 8.01, 110.0, 2.50), run("A1", 32, 7.21, 150.0, 1.90),
    run("A3", 32, 5.09, 105.0, 2.10), run("A4", 32, 7.30, 155.0, 1.85),
    run("A0", 1, 8.01, 112.0, 4.30), run("A1", 1, 7.21, 130.0, 3.90),
]


def test_headline_run_is_a_positive_result():
    verdict, rows = decide.evaluate(GOOD)
    assert verdict == "POSITIVE"
    assert {h for h, v, _ in rows if v == "PASS"} >= {"H1", "H2", "H3", "H4"}


def test_column_insensitive_verification_is_a_negative_result():
    flat = [run("A0", 32, 8.01, 110.0, 2.50), run("A1", 32, 8.01, 111.0, 2.48),
            run("A3", 32, 5.09, 105.0, 2.10), run("A0", 1, 8.01, 112.0, 4.30),
            run("A1", 1, 8.01, 112.0, 4.29)]
    assert decide.evaluate(flat)[0] == "NEGATIVE RESULT"


def test_losslessness_failure_invalidates_the_run():
    leaky = [dict(r, exact_match_ar=0.95) for r in GOOD]
    assert decide.evaluate(leaky)[0] == "INVALID RUN"


def test_missing_cell_never_becomes_a_verdict():
    assert decide.evaluate(GOOD[:2])[0] == "INCOMPLETE"


def test_gain_below_threshold_is_mechanism_only():
    weak = [dict(r) for r in GOOD]
    for r in weak:
        if r["arm"] == "A1" and r["concurrency"] == 32:
            r["tok_per_s"] = 118.0     # +7% over A0, below the 15% bar
    assert decide.evaluate(weak)[0] == "MECHANISM ONLY"


def test_fitted_parameters_are_reported_not_assumed():
    lines = decide.report_fitted(GOOD)
    joined = "\n".join(lines)
    assert "rho" in joined and "w_tail" in joined


def test_loader_rejects_incomplete_records(tmp_path):
    bad = dict(GOOD[0])
    bad.pop("tau")
    (tmp_path / "r.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing keys"):
        decide.load(tmp_path)


def test_loader_accepts_a_full_record(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps(GOOD[0]))
    assert len(decide.load(tmp_path)) == 1
