"""The grid runner must produce schema-valid records and never let synthetic ones through."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decide
import run_grid

import metrics


def _run(tmp_path, *extra):
    out = tmp_path / "grid"
    rc = run_grid.main(["--dry-run", "--prompts", "2", "--concurrency", "1",
                        "--benchmarks", "math500", "--out", str(out), *extra])
    return rc, out.with_name(out.name + "_dryrun")


def test_dry_run_writes_schema_valid_records(tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 4                     # A0, A1, A3, A5 at c=1
    for f in files:
        metrics.validate_record(json.loads(f.read_text()))


def test_dry_run_leaves_a_synthetic_marker(tmp_path):
    _, out = _run(tmp_path)
    assert (out / "SYNTHETIC_DATA_DO_NOT_USE").exists()


def test_decide_refuses_synthetic_directory(tmp_path, capsys, monkeypatch):
    _, out = _run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["decide.py", "--results", str(out), "--benchmark", "math500"])
    assert decide.main() == 3
    assert "SYNTHETIC_DATA_DO_NOT_USE" in capsys.readouterr().out


def test_a4_is_refused_until_the_backend_probe(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_grid.main(["--dry-run", "--prompts", "1", "--concurrency", "1",
                       "--benchmarks", "math500", "--arms", "A4",
                       "--out", str(tmp_path / "g")])
    assert "E0" in str(exc.value)


def test_cells_skip_the_sweep_arm_away_from_serving_batch():
    got = run_grid.cells(("A0", "A2"), (1, 32))
    assert ("A0", 1) in got and ("A2", 32) in got and ("A2", 1) not in got
