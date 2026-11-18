"""E0 probe: marker plumbing and verdict logic, both exercised without a GPU.

The verdict logic is tested against throwaway packages on disk rather than mocks of
`inspect`, so the same code path runs as on the box: import a module, read its fields, grep
its source. A test that only asserts "probe_q2 was called" would not catch the case where a
real request object exposes `verify_len` and the probe still says NO.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import e0_probe


def _pkg(tmp_path, monkeypatch, name, body, sub=None):
    """Write a one-module package on disk and put it on sys.path; return its module name."""
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body)
    if sub is not None:
        (pkg / "sub.py").write_text(sub)
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


# ------------------------------------------------------------------ dry-run markers


def test_dry_run_prints_the_three_answer_lines(capsys):
    rc = e0_probe.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "E0-Q1: YES (dry-run synthetic)" in out
    assert "E0-Q2: YES (dry-run synthetic)" in out
    assert "E0-Q3: " in out and "x (dry-run synthetic) at batch=32" in out
    assert "E0-DONE: rc=0" in out


def test_dry_run_labels_every_verdict_synthetic(capsys):
    e0_probe.main(["--dry-run"])
    out = capsys.readouterr().out
    verdicts = [ln for ln in out.splitlines() if ln.startswith(("E0-Q1:", "E0-Q2:", "E0-Q3:"))]
    assert len(verdicts) == 3
    for line in verdicts:
        assert "dry-run" in line, f"unlabelled dry-run verdict: {line}"


def test_dry_run_writes_json_out(tmp_path):
    out = tmp_path / "e0.json"
    assert e0_probe.main(["--dry-run", "--json-out", str(out)]) == 0
    assert '"dry_run": true' in out.read_text()


# ------------------------------------------------------------------ Q2 verdicts


def test_q2_unknown_when_the_package_cannot_be_imported(monkeypatch):
    monkeypatch.setattr(
        e0_probe,
        "PROBE_TARGETS",
        (("absent.SamplingParams", "e0_absent_module", ("SamplingParams",), "request"),),
    )
    r = e0_probe.probe_q2()
    assert r["answer"] == "UNKNOWN"
    assert "importable" in r["reason"]
    assert r["rows"][0]["status"].startswith("import e0_absent_module failed")


def test_q2_yes_when_the_request_object_declares_a_verify_field(tmp_path, monkeypatch):
    name = _pkg(
        tmp_path,
        monkeypatch,
        "e0fake_req",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass SamplingParams:\n"
        "    max_new_tokens: int = 8\n    verify_len: int = 8\n",
    )
    monkeypatch.setattr(
        e0_probe, "PROBE_TARGETS", ((f"{name}.SamplingParams", name, ("SamplingParams",), "request"),)
    )
    r = e0_probe.probe_q2()
    assert r["answer"] == "YES"
    assert r["request_field"] == "verify_len"
    assert "SamplingParams" in r["callable"]


def test_q2_no_when_only_the_startup_config_has_it(tmp_path, monkeypatch):
    name = _pkg(
        tmp_path,
        monkeypatch,
        "e0fake_cfg",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass ServerArgs:\n    speculative_num_draft_tokens: int = 16\n",
    )
    monkeypatch.setattr(
        e0_probe, "PROBE_TARGETS", ((f"{name}.ServerArgs", name, ("ServerArgs",), "startup"),)
    )
    r = e0_probe.probe_q2()
    assert r["answer"] == "NO"
    assert r["startup_field"] == "speculative_num_draft_tokens"


def test_q2_prefers_the_request_field_when_both_exist(tmp_path, monkeypatch):
    name = _pkg(
        tmp_path,
        monkeypatch,
        "e0fake_both",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass SamplingParams:\n    verify_columns: int = 9\n",
        sub="from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass ServerArgs:\n    num_draft_tokens: int = 16\n",
    )
    monkeypatch.setattr(
        e0_probe,
        "PROBE_TARGETS",
        (
            (f"{name}.ServerArgs", name, ("ServerArgs",), "startup"),
            (f"{name}.SamplingParams", name, ("SamplingParams",), "request"),
        ),
    )
    r = e0_probe.probe_q2()
    assert r["answer"] == "YES"
    assert r["request_field"] == "verify_columns"
    assert "verify_columns" in r["hits"].get(name, {})


def test_q2_grep_finds_the_symbol_outside_the_object_that_declares_it(tmp_path, monkeypatch):
    name = _pkg(
        tmp_path,
        monkeypatch,
        "e0fake_grep",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass SamplingParams:\n    max_new_tokens: int = 8\n",
        sub="# the scheduler would read req.sampling_params.verify_len here\nVERIFY = 'verify_len'\n",
    )
    monkeypatch.setattr(
        e0_probe, "PROBE_TARGETS", ((f"{name}.SamplingParams", name, ("SamplingParams",), "request"),)
    )
    r = e0_probe.probe_q2()
    # Declared nowhere, referenced somewhere: not enough to claim a per-request field, but
    # the symbol must show up in the grep evidence rather than be silently dropped.
    assert r["answer"] in ("NO", "UNKNOWN")
    assert "verify_len" in r["hits"].get(name, {})


# ------------------------------------------------------------------ Q3 gate


def test_q3_is_not_measurable_without_per_request_control():
    r = e0_probe.probe_q3("UNKNOWN", dry_run=True)
    assert r["answer"] == "NOT MEASURABLE"
    assert "two processes" in r["reason"]


def test_q3_dry_run_reports_a_ratio_and_both_timings():
    r = e0_probe.probe_q3("YES", dry_run=True, backend="sglang", batch=32)
    assert isinstance(r["answer"], float)
    assert r["answer"] > 0
    assert r["ms17"] > r["ms9"], "extra verify columns cannot be cheaper in the synthetic model"
    assert r["batch"] == 32


def test_probe_all_reports_rc_three_when_the_reference_path_fails(monkeypatch):
    monkeypatch.setattr(
        e0_probe,
        "probe_q1",
        lambda *_a, **_k: {"answer": "NO", "reason": "synthetic failure", "dry_run": True},
    )
    r = e0_probe.probe_all(dry_run=True)
    assert r["rc"] == 3


@pytest.mark.parametrize("columns", [17, 9])
def test_engine_config_accepts_the_two_probe_widths(columns):
    from serve import EngineConfig

    cfg = EngineConfig(drafter=e0_probe.DRAFTER, backend="sglang", verify_columns=columns)
    assert cfg.submitted == columns - 1
