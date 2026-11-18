"""Guards the arm register and the LibRaSpec marginal-length rule. Run: uv run pytest -q"""

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arms import ARMS, Arm, choose_length, submitted_prefix


def cumprod(seq):
    out, acc = [], 1.0
    for p in seq:
        acc *= p
        out.append(acc)
    return out


def brute_length(confs, c_cost, min_len, max_len):
    """Reference: the largest admissible prefix, with the same clamping as the rule."""
    n = len(confs)
    upper = n if max_len is None else min(max_len, n)
    lower = min(min_len, upper)
    best = lower
    for length in range(lower, upper + 1):
        if length and cumprod(confs)[length - 1] > c_cost:
            best = length
    return best


# --- the register -------------------------------------------------------------------------


def test_registry_matches_the_contract():
    assert set(ARMS) == {"A0", "A1", "A2", "A3", "A4", "A5"}
    assert (ARMS["A0"].draft_width, ARMS["A0"].verify_columns, ARMS["A0"].selector) == (16, 17, None)
    assert (ARMS["A1"].draft_width, ARMS["A1"].verify_columns, ARMS["A1"].selector) == (16, 9, "fixed")
    assert (ARMS["A2"].draft_width, ARMS["A2"].verify_columns, ARMS["A2"].selector) == (16, None, "fixed")
    assert (ARMS["A3"].draft_width, ARMS["A3"].verify_columns, ARMS["A3"].selector) == (8, 9, "fixed")
    assert (ARMS["A4"].draft_width, ARMS["A4"].verify_columns, ARMS["A4"].selector) == (16, None, "marginal")
    assert (ARMS["A5"].draft_width, ARMS["A5"].verify_columns, ARMS["A5"].selector) == (0, 0, None)
    for key, arm in ARMS.items():
        assert arm.name == key


def test_arm_is_frozen():
    with pytest.raises(FrozenInstanceError):
        ARMS["A1"].verify_columns = 9  # type: ignore[misc]
    assert ARMS["A1"] == Arm("A1", 16, 9, "fixed")   # the registry entry survived intact


# --- the marginal rule --------------------------------------------------------------------


def test_monotone_hazard_stops_at_the_first_drop():
    """Optimum sits at the first index where the prefix product drops below the cost."""
    confs = [0.9, 0.8, 0.5, 0.2, 0.1]
    prods = cumprod(confs)  # 0.9, 0.72, 0.36, 0.072, 0.0072
    cost = 0.1
    assert prods[2] > cost > prods[3]
    assert choose_length(confs, c_cost=cost) == 3
    # Interior: strictly between the floor and the draft length.
    assert 1 < choose_length(confs, c_cost=cost) < len(confs)


def test_the_rule_equals_the_first_dropping_index_on_a_long_sequence():
    confs = [0.97 - 0.01 * i for i in range(40)]
    for cost in (1e-6, 0.01, 0.05, 0.2, 0.5):
        prods = cumprod(confs)
        expected = next((i for i, v in enumerate(prods) if not v > cost), len(confs))
        expected = max(1, expected)
        assert choose_length(confs, c_cost=cost) == expected, cost


def test_alpha_scales_the_cost():
    confs = [0.9, 0.8, 0.5, 0.2, 0.1]
    assert choose_length(confs, c_cost=0.1, alpha=1.0) == 3
    assert choose_length(confs, c_cost=0.1, alpha=4.0) == 2   # threshold 0.4
    assert choose_length(confs, c_cost=0.1, alpha=0.5) == 4   # threshold 0.05


def test_free_cost_submits_the_whole_draft():
    assert choose_length([1.0] * 10) == 10
    assert choose_length([0.6] * 10, c_cost=0.0) == 10
    assert choose_length([1.0] * 10, c_cost=0.0, max_len=6) == 6
    assert choose_length([0.5] * 4, c_cost=-1.0) == 4        # negative cost never truncates


def test_clamping():
    assert choose_length([0.01] * 5, c_cost=0.5) == 1         # min_len floor
    assert choose_length([0.01] * 5, c_cost=0.5, min_len=3) == 3
    assert choose_length([0.01] * 5, c_cost=0.5, min_len=0) == 0
    assert choose_length([1.0] * 10, max_len=1) == 1          # max_len ceiling
    assert choose_length([0.9] * 4, max_len=99) == 4          # ceiling above n is ignored
    assert choose_length([0.9] * 4, min_len=6) == 4           # floor above n collapses to n
    assert choose_length([0.9] * 4, min_len=3, max_len=2) == 2


def test_all_confident_is_max_len():
    confs = [1.0] * 12
    assert choose_length(confs, c_cost=0.0) == 12
    assert choose_length(confs, c_cost=0.0, max_len=12) == 12
    assert choose_length(confs, c_cost=0.0, max_len=7) == 7
    assert choose_length(confs, c_cost=1.0) == 1              # 1.0 is not > 1.0


def test_all_zero_is_the_floor():
    assert choose_length([0.0] * 5, c_cost=0.0) == 1
    assert choose_length([0.0] * 5, c_cost=0.0, min_len=0) == 0
    assert choose_length([0.0] * 5, c_cost=0.0, min_len=2) == 2


def test_empty_input_is_the_degenerate_case():
    assert choose_length([]) == 0
    assert choose_length([], c_cost=0.5, min_len=4, max_len=8) == 0
    assert choose_length([]) == choose_length([])
    assert submitted_prefix([], ARMS["A0"]) == 0


def test_against_brute_force_on_a_deterministic_sequence():
    """Interior optima, both ends, and ties -- no randomness, fixed seed-free arithmetic."""
    seq = [(i * 37 % 101) / 100.0 for i in range(1, 60)]
    for c_cost in (0.0, 0.01, 0.1, 0.35, 0.9, 2.0):
        for max_len in (None, 1, 5, 17, 200):
            got = choose_length(seq, c_cost=c_cost, max_len=max_len)
            assert got == brute_length(seq, c_cost, 1, max_len), (c_cost, max_len, got)


def test_is_pure_and_deterministic():
    confs = [0.9, 0.7, 0.4, 0.2]
    snapshot = list(confs)
    first = choose_length(confs, c_cost=0.15)
    assert choose_length(confs, c_cost=0.15) == first
    assert confs == snapshot                                  # input not consumed or mutated
    assert choose_length(tuple(confs), c_cost=0.15) == first   # any Sequence works


# --- arm dispatch -------------------------------------------------------------------------


def test_submitted_prefix_per_arm():
    confs = [0.95] * 16
    assert submitted_prefix(confs, ARMS["A0"]) == 16   # verify the whole draft
    assert submitted_prefix(confs, ARMS["A1"]) == 9    # fixed 9 columns
    assert submitted_prefix(confs, ARMS["A2"]) == 16   # fixed L unresolved -> whole draft
    assert submitted_prefix([0.9] * 8, ARMS["A3"]) == 8
    assert submitted_prefix(confs, ARMS["A5"]) == 0    # AR reference drafts nothing
    assert submitted_prefix([0.9] * 40, ARMS["A4"]) == 16   # capped by draft width
    assert submitted_prefix(confs, ARMS["A4"], c_cost=0.0) == choose_length(
        confs, c_cost=0.0, max_len=16
    )


def test_marginal_arm_truncates_when_confidence_decays():
    confs = [0.9] * 16
    assert submitted_prefix(confs, ARMS["A4"], c_cost=0.6) == 4
    assert submitted_prefix(confs, ARMS["A4"], c_cost=0.6) < submitted_prefix(confs, ARMS["A0"])


def test_logits_and_probabilities_agree():
    probs = [0.99, 0.95, 0.9, 0.8, 0.6, 0.3]
    logits = [math.log(p / (1.0 - p)) for p in probs]
    for cost in (0.0, 0.1, 0.4):
        assert submitted_prefix(logits, ARMS["A4"], c_cost=cost) == submitted_prefix(
            probs, ARMS["A4"], c_cost=cost
        )


def test_per_position_score_rows_are_reduced_to_their_max_probability():
    rows = [[3.0, 0.5, -1.0]] * 5
    assert submitted_prefix(rows, ARMS["A4"]) == 5
    prob_rows = [[0.7, 0.2, 0.1]] * 5
    assert submitted_prefix(prob_rows, ARMS["A4"]) == 5
    assert submitted_prefix([[0.0, 0.0]], ARMS["A4"]) == 1


def test_unknown_selector_is_rejected():
    bogus = Arm("A9", 8, 8, "oracle")
    with pytest.raises(ValueError, match="unknown selector"):
        submitted_prefix([0.9] * 8, bogus)


# --- contract guards ----------------------------------------------------------------------


def test_rule_is_documented_as_prior_art():
    doc = choose_length.__doc__ or ""
    assert "arXiv:2608.08721" in doc
    assert "Theorem 3.4" in doc
    assert "LibraSpec" in doc


def test_module_imports_no_torch():
    code = "import sys, src.arms; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
