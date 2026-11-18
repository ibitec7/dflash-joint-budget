"""vlc_tokps multiplies two instruments; the multiplication has to stay arithmetic, not vibes.

Every fixture here is a handful of rows in ``tmp_path`` or an in-memory engine map, so the real
1.6 MB acceptance record is never opened. What is pinned is the mechanics: which way the round-cost
factor points, what ``absent`` means (and that it is not 0 and not an interpolation), that the
composition is a product to 1e-9, that the bootstrap CI reproduces from ``random.Random(0)``, and
that the verdict is read on the paired reading rather than the pooled one.
"""

import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import vlc_tokps


def engine_entry(arm, ms, columns=17, tau=8.0, tok=100.0, mode="default", conc=32, seed=0):
    rec = {
        "arm": arm, "benchmark": "math500", "concurrency": conc, "seed": seed,
        "ms_per_round_p50": ms, "tok_per_s": tok, "tau": tau, "verify_columns": columns,
    }
    return {"rec": rec, "path": Path(f"results/{arm}_math500_c{conc}_s{seed}.json")}


def engine(a0_ms=30.0, a1_ms=15.0):
    """The two engine records compose() prices. A0 is 17 columns, A1 is the 9-column window."""
    return {
        ("default", "A0", "math500", 32, 0): engine_entry("A0", a0_ms, columns=17, tau=13.0),
        ("default", "A1", "math500", 32, 0): engine_entry("A1", a1_ms, columns=9, tau=7.0),
    }


def flat_rows(accepted_16, accepted_9):
    """One round per prompt under each block, so pooled and paired agree by construction."""
    out = []
    for i, (a16, a9) in enumerate(zip(accepted_16, accepted_9, strict=True)):
        out.append({"block": 16, "prompt": i, "ctx_len": 512, "accepted": a16})
        out.append({"block": 9, "prompt": i, "ctx_len": 512, "accepted": a9})
    return out


FULL_TAU = {"b16_k9": 9.0, "b16_k16": 16.0, "b9_k9": 8.0}


def sweep(rows, tau=None):
    """Write a minimal harness JSON to tmp_path and return its path."""
    return {"tau": dict(tau or {}), "rows": list(rows)}


# ------------------------------------------------------------------- (a) direction of the cost


def test_more_expensive_treatment_gives_cost_below_one_and_a_ratio_below_tau():
    """cost = round(control) / round(treatment), so a costlier 9-column round shrinks the ratio.

    A0 (17 columns) is the control at 30 ms/round; the truncated arm's own record says 60 ms/round
    for the 9-column round. The factor must be 0.5, and the composed tok/s ratio must land below
    the tau ratio it multiplies, because the second factor is a penalty rather than a bonus.
    """
    data = sweep(flat_rows([16, 16, 16], [8, 8, 8]))
    a = vlc_tokps.compose(data, engine(a0_ms=30.0, a1_ms=60.0))

    a0 = a["targets"]["A0"]
    assert a0["cost"] == pytest.approx(0.5)
    assert a0["tau_paired"] == pytest.approx(9.0 / 16.0)
    assert a0["composed_paired"] < a0["tau_paired"]
    assert a0["composed_paired"] == pytest.approx(a0["tau_paired"] * 0.5)
    assert a["missing"] == []


def test_cheaper_treatment_pushes_the_cost_factor_above_one():
    """The mirror case, so the sign of the factor is measured and not just asserted one way."""
    data = sweep(flat_rows([16, 16, 16], [8, 8, 8]))
    a = vlc_tokps.compose(data, engine(a0_ms=60.0, a1_ms=30.0))

    a0 = a["targets"]["A0"]
    assert a0["cost"] == pytest.approx(2.0)
    assert a0["composed_paired"] > a0["tau_paired"]


def test_cost_against_the_narrow_arm_is_the_identity_on_one_record():
    """A1 prices 9 columns over 9 columns: 1.0, and the print layer says it is not a measurement."""
    data = sweep(flat_rows([16, 16, 16], [8, 8, 8]))
    a = vlc_tokps.compose(data, engine(a0_ms=30.0, a1_ms=60.0))

    a1 = a["targets"]["A1"]
    assert a1["cost"] == pytest.approx(1.0)
    assert a1["composed_paired"] == pytest.approx(a1["tau_paired"])


# ------------------------------------------------------------------- (b) absent cells


def test_deleted_cell_is_absent_not_zero_and_the_verdict_disappears():
    """Delete b16_k9 (and the rows that could re-derive it): the factor is absent, verdict gone.

    Nothing may stand in for the cell: not 0.0, not the neighbouring k, not an interpolation
    between b12 and b16. The factor prints ``absent`` and any verdict built on it is withdrawn.
    """
    rows = [r for r in flat_rows([16, 16], [8, 8]) if r["block"] != 16]
    cut = sweep(rows, FULL_TAU)
    del cut["tau"]["b16_k9"]

    a = vlc_tokps.compose(cut, engine())
    value, txt, source = a["cells"]["tau_9(16)"]

    assert value is None
    assert txt == vlc_tokps.ABSENT == "absent"
    assert source == "absent"
    assert value != 0.0
    assert any(name.endswith(":b16_k9") for name in a["missing"])
    assert a["targets"]["A0"]["tau_cells"] is None
    assert a["targets"]["A0"]["tau_paired"] is None
    assert a["targets"]["A0"]["composed_paired"] is None
    assert a["verdict"] == "no verdict"
    assert vlc_tokps.fmt(None) == "absent"


def test_the_same_fixture_without_the_deletion_does_carry_a_verdict():
    """The control for the deletion test: the cell is the only thing that changed."""
    a = vlc_tokps.compose(sweep(flat_rows([16, 16], [8, 8]), FULL_TAU), engine())
    assert a["cells"]["tau_9(16)"][0] == pytest.approx(9.0)
    assert a["missing"] == []
    assert a["verdict"] != "no verdict"


def test_beyond_the_drafted_width_is_absent_but_the_correction_token_is_measured():
    """k = b + 1 is in reach (accepted counts the target's token); k = b + 2 is not invented."""
    data = sweep(flat_rows([16, 16], [8, 8]))

    assert vlc_tokps.tau_at(data, 16, 17)[0] == pytest.approx(16.0)
    assert vlc_tokps.tau_at(data, 16, 18) == (None, "absent")
    assert vlc_tokps.tau_at(data, 9, 11) == (None, "absent")


def test_missing_engine_record_is_absent_too_and_withdraws_the_verdict():
    """A cell can be present while the round cost is not; that is also ``no verdict``."""
    only_a0 = {("default", "A0", "math500", 32, 0): engine_entry("A0", 30.0)}
    a = vlc_tokps.compose(sweep(flat_rows([16, 16], [8, 8]), FULL_TAU), only_a0)

    assert a["rounds"]["A1"] == (None, "absent")
    assert a["targets"]["A1"]["cost"] is None
    assert a["targets"]["A1"]["composed_paired"] is None
    assert a["verdict"] == "no verdict"


# ------------------------------------------------------------------- (c) the product


def test_composed_is_tau_times_cost_to_one_e_minus_nine():
    a = vlc_tokps.compose(sweep(flat_rows([16, 14, 11], [9, 7, 5]), FULL_TAU), engine())

    for target in ("A0", "A1"):
        t = a["targets"][target]
        assert math.isclose(t["composed_paired"], t["tau_paired"] * t["cost"], abs_tol=1e-9)
        assert math.isclose(t["composed_cells"], t["tau_cells"] * t["cost"], abs_tol=1e-9)
        assert math.isclose(t["ci_tau"][0] * t["cost"], t["ci"][0], abs_tol=1e-9)
        assert math.isclose(t["ci_tau"][1] * t["cost"], t["ci"][1], abs_tol=1e-9)


# ------------------------------------------------------------------- (d) the bootstrap


def _percentile(values, q):
    pos = (len(values) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _replay(num, den, resamples, seed):
    """The documented procedure, written out independently of the module."""
    rng = random.Random(seed)
    n = len(num)
    draws = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        bottom = sum(den[i] for i in idx)
        if bottom:
            draws.append(sum(num[i] for i in idx) / bottom)
    draws.sort()
    return (_percentile(draws, 2.5), _percentile(draws, 97.5)), len(draws)


def test_bootstrap_ci_reproduces_under_seed_zero():
    """Same fixture, same seed, same interval: the CI is a function of the data, not of the run."""
    num, den = [9.0, 9.0, 1.0, 5.0], [16.0, 16.0, 16.0, 12.0]
    ci, n = vlc_tokps.bootstrap_ratio_ci(num, den, resamples=64, seed=0)
    expected, expected_n = _replay(num, den, 64, 0)

    assert (ci[0], ci[1]) == pytest.approx(expected, abs=1e-12)
    assert n == expected_n == 64
    point = sum(num) / sum(den)
    assert ci[0] <= point <= ci[1]
    assert vlc_tokps.bootstrap_ratio_ci(num, den, resamples=64, seed=0) == (ci, n)


def test_bootstrap_of_identical_arms_is_exactly_one():
    """A known answer: if every prompt has num == den, every resample ratio is 1.0."""
    num = den = [2.0, 4.0, 6.0, 8.0]
    ci, n = vlc_tokps.bootstrap_ratio_ci(num, den, resamples=32, seed=0)

    assert ci == (1.0, 1.0)
    assert n == 32


def test_bootstrap_needs_two_prompts():
    """One prompt cannot be resampled: no interval, and the count says zero draws."""
    assert vlc_tokps.bootstrap_ratio_ci([9.0], [8.0]) == ((None, None), 0)
    assert vlc_tokps.bootstrap_ratio_ci([], []) == ((None, None), 0)


# ------------------------------------------------------------------- (e) paired vs pooled


def uneven_rows():
    """Prompt 0 has three rounds under each block, prompt 1 has one: the weightings diverge."""
    rows = []
    for a16 in (16, 16, 16):
        rows.append({"block": 16, "prompt": 0, "ctx_len": 512, "accepted": a16})
    for a9 in (9, 9, 9):
        rows.append({"block": 9, "prompt": 0, "ctx_len": 512, "accepted": a9})
    rows.append({"block": 16, "prompt": 1, "ctx_len": 512, "accepted": 1})
    rows.append({"block": 9, "prompt": 1, "ctx_len": 512, "accepted": 9})
    return rows


def test_paired_and_pooled_differ_when_round_counts_differ_and_the_verdict_is_paired():
    """Pooled weights rounds, paired weights prompts; the reported verdict rides on the paired one."""
    a = vlc_tokps.compose(sweep(uneven_rows()), engine(a0_ms=30.0, a1_ms=15.0))

    a0, a1 = a["targets"]["A0"], a["targets"]["A1"]
    # pooled: mean of min(accepted, 9) over the four block-16 rounds = (9 + 9 + 9 + 1) / 4,
    # over the pooled min(accepted, 16) of the same rounds = (16 + 16 + 16 + 1) / 4.
    assert a0["tau_cells"] == pytest.approx(7.0 / 12.25)
    # paired: mean of the two per-prompt means = (9 + 1) / 2, over mean(16, 1) / 2 = 8.5, because
    # prompt 1's own block-16 mean is 1.0 and the paired reading does not let prompt 0's three
    # rounds outvote it.
    assert a0["tau_paired"] == pytest.approx(5.0 / 8.5)
    assert a0["tau_paired"] != pytest.approx(a0["tau_cells"])
    assert a0["prompts"] == 2
    assert (a0["wins"], a0["ties"], a0["losses"]) == (0, 1, 1)

    assert a1["tau_cells"] == pytest.approx(7.0 / 9.0)
    assert a1["tau_paired"] == pytest.approx(5.0 / 9.0)

    tail = a["verdict"].split("VLC vs A1: ", 1)[1]
    assert tail.startswith(f"not supported ({a1['composed_paired']:.3f}")
    assert f"{a1['composed_paired']:.3f}" not in ("", "None")
    assert f"{a1['composed_paired']:.3f}" in a["claims"][1]
    assert f"{a0['composed_paired']:.3f} paired" in a["verdict"]


def test_render_prints_absent_for_the_missing_cell_and_no_verdict(tmp_path, capsys):
    """The print path carries the same two words the caller greps for."""
    cut = sweep(flat_rows([16, 16], [8, 8]), FULL_TAU)
    del cut["tau"]["b16_k9"]
    cut["rows"] = [r for r in cut["rows"] if r["block"] != 16]
    path = tmp_path / "accept_mini.json"
    path.write_text(json.dumps(cut))

    a = vlc_tokps.render(vlc_tokps.load(path), engine(), path)
    out = capsys.readouterr().out

    assert a["verdict"] == "no verdict"
    assert "b16_k9" in out
    assert "absent" in out
    assert "verdict: no verdict" in out
    assert "COMPOSED, NOT END-TO-END" in out
