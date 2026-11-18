"""Guards the anchors and the model algebra. Run: uv run pytest -q"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import cost_model as cm


def test_anchors_reproduce_measured_speedups():
    for rho in (0.3, 0.5, 0.7, 0.9):
        assert abs(cm.speedup(1, 17, rho) - cm.S_C1) < 1e-6
        assert abs(cm.speedup(32, 17, rho) - cm.S_C32) < 1e-6


def test_verification_cost_matches_v_at_both_ends():
    assert abs(cm.v_verify(1, 17, 0.5) - cm.V_C1) < 1e-9
    assert abs(cm.v_verify(32, 17, 0.5) - cm.V_C32) < 1e-9


def test_tau_caps_on_truncation():
    assert cm.tau_of(17) == cm.TAU_16
    assert cm.tau_of(5) == 5


def test_truncation_beats_width_reduction():
    """The project's central claim: L=8 keeps tau, block-8 loses it."""
    for rho in (0.5, 0.7, 0.9):
        c = 32
        trunc = cm.speedup(c, 9, rho)
        width = cm.tau_of(9, tau_block=cm.TAU_8) / (cm.ALPHA + cm.v_verify(c, 9, rho))
        assert trunc > width, (rho, trunc, width)


def test_selftest_runs():
    out = subprocess.run(
        [sys.executable, "analysis/cost_model.py", "--selftest"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "selftest ok" in out.stdout
