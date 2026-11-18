"""The recorder's only job is to emit a record the decider accepts; test that, not the wiring.

Run: uv run pytest -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import decide

from src import metrics

THREE_ROUNDS = ((8, 9, 1.88), (7, 9, 1.90), (9, 9, 1.92))


def recorder(
    arm: str = "A1",
    benchmark: str = "math500",
    concurrency: int = 32,
    seed: int = 0,
    draft_width: int = 16,
    verify_columns: int = 9,
    prompts: int = 200,
) -> metrics.RunRecorder:
    return metrics.RunRecorder(
        arm=arm,
        benchmark=benchmark,
        concurrency=concurrency,
        seed=seed,
        draft_width=draft_width,
        verify_columns=verify_columns,
        prompts=prompts,
    )


def finished(rounds=THREE_ROUNDS) -> dict:
    rec = recorder()
    for accepted, cols, ms in rounds:
        rec.add_round(accepted, cols, ms)
    return rec.finish(tok_per_s=151.0, ar_tok_per_s=110.0, exact_match_ar=0.997)


def test_schema_keys_match_the_decider():
    assert metrics.SCHEMA_KEYS == frozenset(decide.REQUIRED)
    assert len(metrics.SCHEMA_KEYS) == 14
    assert {"tau", "ms_per_round_p50", "ms_per_round_p95"} <= metrics.SCHEMA_KEYS


def test_record_has_exactly_the_schema_keys():
    record = finished()
    assert set(record) == metrics.SCHEMA_KEYS
    assert record["prompts"] == 200 and record["concurrency"] == 32
    assert record["tau"] == pytest.approx(8.0)
    assert record["graph_recaptures"] == 0        # default, not omitted


def test_tau_counts_the_bonus_token():
    """A round that accepts nothing still emits the bonus token: tau == 1, not 0."""
    rec = recorder()
    rec.add_round(1, 9, 2.0)
    assert rec.finish(tok_per_s=120.0, ar_tok_per_s=110.0, exact_match_ar=1.0)["tau"] == 1.0

    full = recorder()
    for _ in range(2):
        full.add_round(9, 9, 2.0)                 # every column accepted, bonus included
    assert full.finish(tok_per_s=120.0, ar_tok_per_s=110.0, exact_match_ar=1.0)["tau"] == 9.0


def test_single_round_gives_p50_equal_to_p95():
    record = finished(rounds=((8, 9, 2.34),))
    assert record["ms_per_round_p50"] == 2.34
    assert record["ms_per_round_p95"] == 2.34
    assert record["tau"] == 8.0


def test_percentiles_interpolate_over_rounds():
    record = finished(rounds=tuple((8, 9, ms) for ms in (1.0, 2.0, 3.0, 4.0)))
    assert record["ms_per_round_p50"] == pytest.approx(2.5)
    assert record["ms_per_round_p95"] == pytest.approx(3.85)


def test_missing_key_raises_value_error_naming_it(monkeypatch):
    """finish() must refuse a record the decider would reject, and say which key is absent."""
    monkeypatch.setattr(metrics, "SCHEMA_KEYS", metrics.SCHEMA_KEYS | {"gpu_hours"})
    with pytest.raises(ValueError, match="gpu_hours"):
        finished()


def test_validate_record_rejects_missing_and_extra_keys():
    record = finished()
    partial = dict(record)
    partial.pop("tau")
    with pytest.raises(ValueError, match="tau"):
        metrics.validate_record(partial)
    with pytest.raises(ValueError, match="outside the results schema"):
        metrics.validate_record(dict(record, notes="looked fast"))


def test_round_trip_through_decide_load(tmp_path):
    record = finished()
    rec = recorder()
    for accepted, cols, ms in THREE_ROUNDS:
        rec.add_round(accepted, cols, ms)
    rec.finish(tok_per_s=151.0, ar_tok_per_s=110.0, exact_match_ar=0.997)

    out = tmp_path / "results" / "a1_c32_s0.json"
    rec.to_json(out)                              # creates the parent directory

    assert out.exists()
    assert json.loads(out.read_text()) == record
    assert decide.load(tmp_path / "results") == [record]


def test_written_cells_produce_a_verdict(tmp_path):
    """End to end: six recorded cells load and decide, with no hand-written JSON."""
    cells = [
        ("A0", 32, 8, 110.0, 2.50),
        ("A1", 32, 7, 151.0, 1.90),
        ("A3", 32, 5, 105.0, 2.10),
        ("A4", 32, 7, 155.0, 1.85),
        ("A0", 1, 8, 112.0, 4.30),
        ("A1", 1, 7, 130.0, 3.90),
    ]
    for seed, (arm, concurrency, tau_floor, tok, ms) in enumerate(cells):
        rec = recorder(arm=arm, concurrency=concurrency, seed=seed,
                       draft_width=8 if arm == "A3" else 16)
        for accepted in (tau_floor, tau_floor + 1):
            rec.add_round(accepted, 9, ms)
        rec.finish(tok_per_s=tok, ar_tok_per_s=110.0, exact_match_ar=0.997)
        rec.to_json(tmp_path / "results" / f"{arm}_c{concurrency}_s{seed}.json")

    runs = decide.load(tmp_path / "results")
    assert len(runs) == 6
    verdict, rows = decide.evaluate(runs)
    assert verdict == "POSITIVE", rows
    assert {hid for hid, status, _ in rows if status == "PASS"} >= {"H1", "H3", "H6"}


def test_to_json_before_finish_is_an_error(tmp_path):
    rec = recorder()
    rec.add_round(8, 9, 2.0)
    with pytest.raises(ValueError, match="finish"):
        rec.to_json(tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_finish_without_rounds_is_an_error():
    with pytest.raises(ValueError, match="add_round"):
        recorder().finish(tok_per_s=151.0, ar_tok_per_s=110.0, exact_match_ar=0.997)


def test_accepted_outside_the_round_is_rejected():
    """Catches a caller who passed draft-only acceptances: 0 is impossible, > columns too."""
    rec = recorder()
    with pytest.raises(ValueError, match="bonus"):
        rec.add_round(0, 9, 2.0)
    with pytest.raises(ValueError, match="bonus"):
        rec.add_round(10, 9, 2.0)
    with pytest.raises(ValueError, match="ms"):
        rec.add_round(8, 9, 0.0)
    assert rec.rounds == 0


def test_finish_rejects_impossible_throughput_and_agreement():
    rec = recorder()
    rec.add_round(8, 9, 2.0)
    with pytest.raises(ValueError, match="tok_per_s"):
        rec.finish(tok_per_s=0.0, ar_tok_per_s=110.0, exact_match_ar=0.997)
    with pytest.raises(ValueError, match="exact_match_ar"):
        rec.finish(tok_per_s=151.0, ar_tok_per_s=110.0, exact_match_ar=1.2)
    with pytest.raises(ValueError, match="finite"):
        rec.finish(tok_per_s=151.0, ar_tok_per_s=float("nan"), exact_match_ar=0.997)


def test_summarize_reports_mean_median_and_spread():
    records = []
    for seed, (tau, ms) in enumerate([(7, 1.90), (7, 1.80), (8, 2.10)]):
        rec = recorder(seed=seed)
        rec.add_round(tau, 9, ms)
        records.append(rec.finish(tok_per_s=151.0, ar_tok_per_s=110.0, exact_match_ar=0.997))

    summary = metrics.summarize(records)
    assert summary["n"] == 3
    assert summary["tau_mean"] == pytest.approx(22 / 3)
    assert summary["tau_median"] == 7.0
    assert (summary["tau_min"], summary["tau_max"]) == (7.0, 8.0)
    assert summary["ms_per_round_p50_median"] == pytest.approx(1.90)


def test_summarize_needs_a_record():
    with pytest.raises(ValueError, match="at least one record"):
        metrics.summarize([])
    with pytest.raises(ValueError, match="tau"):
        metrics.summarize([{"arm": "A1"}])
