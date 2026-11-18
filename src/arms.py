"""Arm register and the per-round verify-length selector (CPU only, no torch).

Two separable knobs are encoded as an :class:`Arm`: ``draft_width`` is how many tokens the
block-diffusion drafter proposes in one pass, ``verify_columns`` is how many columns the target
verifies (``None`` = decided per round). ``selector`` says how the submitted length is picked:
``None`` submits the whole draft, ``"fixed"`` submits a pre-registered length, ``"marginal"``
runs :func:`choose_length` on the drafter's own per-position confidences.

Nothing here measures anything; it only defines what each arm *would* do, so the costing and
decision scripts can share one definition of the arms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ARMS", "Arm", "choose_length", "submitted_prefix"]


@dataclass(frozen=True)
class Arm:
    """One experimental arm: two knobs plus the rule that picks the submitted length."""

    name: str                    # "A0", "A1", ... "A5"
    draft_width: int             # tokens the drafter proposes
    verify_columns: int | None   # columns submitted to the target; None = chosen per round
    selector: str | None         # None (verify all), "fixed", or "marginal"


#: A0 = DFlash as published (draft 16, verify all 17 columns).
#: A1 = fixed truncation to 9 verify columns.
#: A2 = draft 16, fixed length chosen from the L sweep (resolved by the run, not here).
#: A3 = width reduced to 8 instead of truncating.
#: A4 = VLC: draft wide, submit the LibRaSpec-style marginal prefix.
#: A5 = autoregressive reference (no draft, no verify columns).
ARMS: dict[str, Arm] = {
    "A0": Arm("A0", 16, 17, None),
    "A1": Arm("A1", 16, 9, "fixed"),
    "A2": Arm("A2", 16, None, "fixed"),
    "A3": Arm("A3", 8, 9, "fixed"),
    "A4": Arm("A4", 16, None, "marginal"),
    "A5": Arm("A5", 0, 0, None),
}


def choose_length(
    confidences: Sequence[float],
    c_cost: float = 0.0,
    alpha: float = 1.0,
    min_len: int = 1,
    max_len: int | None = None,
) -> int:
    """Submit the longest prefix whose acceptance stays above the scaled column cost.

    Rule: extend the submitted prefix while the marginal expected acceptance gain of the next
    drafted token -- the prefix product ``confidences[0] * ... * confidences[i]``, which is
    non-increasing in ``i`` -- stays strictly above ``alpha * c_cost``, and stop at the first
    token where it does not.

    Prior art: this marginal-length rule is LibraSpec (arXiv:2608.08721), Theorem 3.4, not ours.
    It is reproduced here so the arms are executable, and the report must cite it as such.

    Args:
        confidences: drafter probability for each drafted token, 0-indexed from the verified
            anchor (token 0 is the first drafted token, not the anchor).
        c_cost: per-column verification cost in the same units as the gain (expected tokens).
        alpha: scales ``c_cost`` (``alpha > 1`` is a conservative controller).
        min_len: floor on the answer; speculative decoding submits at least one token.
        max_len: optional ceiling; the answer never exceeds ``len(confidences)``.

    Returns:
        Number of drafted tokens to submit. ``0`` only when nothing was drafted or
        ``min_len <= 0``; the answer is clamped to ``[min_len, min(max_len, n)]``, collapsed to
        the upper bound when ``min_len`` would exceed it.

    Degenerate inputs: empty input returns ``0`` (there is no token to submit, so ``min_len``
    cannot be honoured); all-zero confidences return ``min_len``; all-ones with ``c_cost <= 0``
    return ``max_len`` (or ``n``).
    """
    n = len(confidences)
    if n == 0:
        return 0

    upper = n if max_len is None else min(max(0, int(max_len)), n)
    lower = min(max(0, int(min_len)), upper)
    if upper == 0:
        return lower

    threshold = float(alpha) * float(c_cost)

    # Strict ">" is deliberate: a token whose gain only ties the cost is not worth a column.
    # NaN compares False to everything, so a non-finite confidence stops the prefix as well.
    gain = 1.0
    length = 0
    for i, p in enumerate(confidences):
        if i >= upper:
            break
        gain *= float(p)
        if not gain > threshold:
            break
        length += 1

    return max(lower, min(length, upper))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _row_confidence(row: Sequence[float]) -> float:
    """Confidence of one position: its max probability, from probabilities or from logits."""
    values = [float(v) for v in row]
    if not values:
        return 0.0
    if all(0.0 <= v <= 1.0 for v in values) and abs(math.fsum(values) - 1.0) < 1e-6:
        return max(values)
    top = max(values)
    denom = math.fsum(math.exp(v - top) for v in values)
    return 1.0 / denom if denom > 0.0 else 0.0


def _as_confidences(logits_or_confidences) -> list[float]:
    """Normalise per-position scores into probabilities in [0, 1].

    A 1-D sequence in [0, 1] is already a confidence vector; anything else 1-D is read as
    per-position logits and squashed with a logistic. A nested sequence is read as one score
    vector per position (vocabulary logits or a probability vector) and reduced to its max
    probability.
    """
    rows = list(logits_or_confidences)
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, (int, float)):
        values = [float(v) for v in rows]
        if all(0.0 <= v <= 1.0 for v in values):
            return values
        return [_sigmoid(v) for v in values]
    return [_row_confidence(row) for row in rows]


def submitted_prefix(logits_or_confidences, arm: Arm, c_cost: float = 0.0) -> int:
    """How many drafted tokens this arm submits to the target for one round.

    ``logits_or_confidences`` is the drafter's per-position score for each drafted token (see
    :func:`_as_confidences` for the accepted shapes). The result is always
    ``<= min(len(input), arm.draft_width)`` and is ``0`` for an arm that drafts nothing (A5).

    ``selector="fixed"`` uses ``arm.verify_columns`` as the length; when that is ``None`` (A2,
    whose fixed length comes from the L sweep and is resolved by the run) the whole draft is
    submitted, which is the only length this module can defend without the sweep.
    """
    confs = _as_confidences(logits_or_confidences)
    n = len(confs)
    if n == 0 or arm.draft_width <= 0:
        return 0

    cap = n if arm.verify_columns is None else min(n, int(arm.verify_columns))
    cap = min(cap, int(arm.draft_width))

    if arm.selector is None:
        return cap
    if arm.selector == "fixed":
        return cap
    if arm.selector == "marginal":
        return choose_length(confs, c_cost=c_cost, max_len=cap)

    raise ValueError(f"unknown selector {arm.selector!r} on arm {arm.name!r}")
