"""Record one configured speculative-decoding run as a ``results/*.json`` file.

The record schema is a hard contract: ``scripts/decide.py`` refuses to load a file whose key
set is not exactly its ``REQUIRED`` set, and its verdict is the project's verdict
(``docs/EXPERIMENTS.md`` §4). So this module records, validates against that key set, and
refuses to write anything that would not survive ``decide.load()``.

``SCHEMA_KEYS`` duplicates ``decide.REQUIRED`` instead of importing it: importing
``scripts/decide.py`` mutates ``sys.path`` and pulls in ``analysis/cost_model.py``, which the
serving harness should not need. ``tests/test_metrics.py`` asserts the two sets are identical,
so the duplication cannot drift silently.

Nothing here touches a GPU; a run feeds in per-round samples and end-to-end throughput.

tau: ``tau`` is the mean number of tokens emitted per verification round **including the bonus
token**. A round that accepts none of its draft tokens still emits one token (the bonus), so
``tau >= 1`` always, and a plain mean over ``accepted`` values is the acceptance length. That is
the definition used throughout ``docs/EXPERIMENTS.md`` (§2, H2 compares tau against 7.2 for a
measured 8.01 baseline at 17 columns) and the field ``scripts/decide.py`` thresholds.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path

__all__ = ["SCHEMA_KEYS", "RunRecorder", "summarize", "validate_record"]

#: Exactly the keys ``scripts/decide.py::REQUIRED`` accepts -- no more, no fewer.
#: ``tests/test_metrics.py::test_schema_keys_match_the_decider`` pins this equality.
SCHEMA_KEYS: frozenset[str] = frozenset({
    "arm", "benchmark", "concurrency", "seed", "draft_width", "verify_columns", "prompts",
    "tau", "tok_per_s", "ar_tok_per_s", "ms_per_round_p50", "ms_per_round_p95",
    "exact_match_ar", "graph_recaptures",
})


def _percentile(samples: list[float], q: float) -> float:
    """Linear-interpolated percentile of ``samples``, defined for a single sample.

    Same rule as ``numpy.percentile``'s default: position ``(n - 1) * q``, interpolating between
    the two neighbouring order statistics. Used instead of ``statistics.quantiles`` because that
    one raises for ``n == 1``, and one round is a legitimate (if noisy) run.
    """
    if not samples:
        raise ValueError("no latency samples: percentiles need at least one round")
    xs = sorted(samples)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _as_float(value: float, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _as_positive(value: float, name: str) -> float:
    v = _as_float(value, name)
    if v <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return v


def _as_count(value: int, name: str) -> int:
    """Accept an int-like value (not a string, not a fraction) and return it as an int."""
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if n != value:                                # 1.5, "8", 3.2 all fail here
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if n < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return n


def validate_record(record: Mapping[str, object]) -> None:
    """Raise ``ValueError`` unless ``record`` has exactly the keys the decider accepts.

    A missing key would make ``decide.load()`` refuse the file; a key outside the schema would be
    dropped without a trace, which is how a measurement quietly stops being reported.
    """
    keys = set(record)
    missing = SCHEMA_KEYS - keys
    if missing:
        raise ValueError(f"record is missing required key(s): {', '.join(sorted(missing))}")
    extra = keys - SCHEMA_KEYS
    if extra:
        raise ValueError(
            f"record has key(s) outside the results schema: {', '.join(sorted(extra))}"
        )


class RunRecorder:
    """Accumulate per-round samples for one (arm, benchmark, concurrency, seed) cell.

    ``tau`` is the mean tokens emitted per round **including the bonus token** (see the module
    docstring and ``docs/EXPERIMENTS.md`` §2/§4). ``accepted`` passed to :meth:`add_round` follows
    the same convention: it is the number of tokens the round emitted, so a round that drafts and
    accepts nothing still reports ``accepted=1``.

    ``verify_columns`` is the run's configured column budget; ``None`` for an arm whose length is
    chosen per round (A2, A4) means "not fixed", and per-round values are still validated against
    each round's own ``verified_columns``.
    """

    def __init__(
        self,
        arm: str,
        benchmark: str,
        concurrency: int,
        seed: int,
        draft_width: int,
        verify_columns: int | None,
        prompts: int,
    ) -> None:
        if not isinstance(arm, str) or not arm:
            raise ValueError(f"arm must be a non-empty string, got {arm!r}")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError(f"benchmark must be a non-empty string, got {benchmark!r}")
        self.arm = arm
        self.benchmark = benchmark
        self.concurrency = _as_count(concurrency, "concurrency")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency!r}")
        self.seed = _as_count(seed, "seed")
        self.draft_width = _as_count(draft_width, "draft_width")
        self.verify_columns = (
            None if verify_columns is None else _as_count(verify_columns, "verify_columns")
        )
        self.prompts = _as_count(prompts, "prompts")
        if self.prompts < 1:
            raise ValueError(f"prompts must be >= 1, got {prompts!r}")
        self._latencies_ms: list[float] = []
        self._accepted: list[int] = []
        self._record: dict[str, object] | None = None

    def add_round(self, accepted: int, verified_columns: int, ms: float) -> None:
        """Record one verification round.

        ``accepted`` counts emitted tokens *including the bonus token* (>= 1). ``ms`` is the
        round's wall-clock latency from the serving loop, not a throughput-derived number
        (``docs/EXPERIMENTS.md`` §4 rule 5).
        """
        n_accepted = _as_count(accepted, "accepted")
        n_cols = _as_count(verified_columns, "verified_columns")
        cap = max(n_cols, 1)
        if not 1 <= n_accepted <= cap:
            raise ValueError(
                f"accepted={n_accepted} must be in [1, {cap}] for verified_columns={n_cols}; "
                "accepted counts emitted tokens including the bonus token"
            )
        latency = _as_positive(ms, "ms")
        self._accepted.append(n_accepted)
        self._latencies_ms.append(latency)

    @property
    def rounds(self) -> int:
        return len(self._accepted)

    @property
    def tau(self) -> float:
        """Mean emitted tokens per round, bonus token included (tau >= 1)."""
        if not self._accepted:
            raise ValueError("tau needs at least one round")
        return statistics.fmean(self._accepted)

    def finish(
        self,
        tok_per_s: float,
        ar_tok_per_s: float,
        exact_match_ar: float,
        graph_recaptures: int = 0,
    ) -> dict:
        """Close the run and return its record, validated against ``SCHEMA_KEYS``.

        ``tau`` is the mean tokens per round including the bonus token (module docstring;
        ``docs/EXPERIMENTS.md`` §2). Latency percentiles come from the ``add_round`` samples and
        are defined for a single round.
        """
        if not self._accepted:
            raise ValueError("finish() needs at least one add_round() sample to compute tau")
        match = _as_float(exact_match_ar, "exact_match_ar")
        if not 0.0 <= match <= 1.0:
            raise ValueError(f"exact_match_ar must be a fraction in [0, 1], got {exact_match_ar!r}")
        record: dict[str, object] = {
            "arm": self.arm,
            "benchmark": self.benchmark,
            "concurrency": self.concurrency,
            "seed": self.seed,
            "draft_width": self.draft_width,
            "verify_columns": self.verify_columns,
            "prompts": self.prompts,
            "tau": self.tau,
            "tok_per_s": _as_positive(tok_per_s, "tok_per_s"),
            "ar_tok_per_s": _as_positive(ar_tok_per_s, "ar_tok_per_s"),
            "ms_per_round_p50": _percentile(self._latencies_ms, 0.50),
            "ms_per_round_p95": _percentile(self._latencies_ms, 0.95),
            "exact_match_ar": match,
            "graph_recaptures": _as_count(graph_recaptures, "graph_recaptures"),
        }
        validate_record(record)
        self._record = record
        return dict(record)

    def to_json(self, path: Path) -> None:
        """Write the finished record to ``path``, creating parent directories."""
        if self._record is None:
            raise ValueError("to_json() called before finish(); there is no record to write")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._record, indent=2, sort_keys=True) + "\n")


def summarize(records: list[dict]) -> dict:
    """Summarize the seeds of one cell: mean/median tau, the spread, and the median ms p50.

    Pass one cell (same arm, benchmark and concurrency) -- mixing cells averages different
    mechanisms. Spread is reported because ``docs/EXPERIMENTS.md`` §4 rule 4 asks for it, not
    only the mean.
    """
    if not records:
        raise ValueError("summarize() needs at least one record")
    for i, record in enumerate(records):
        try:
            validate_record(record)
        except ValueError as exc:
            raise ValueError(f"record {i}: {exc}") from exc
    taus = [_as_positive(r["tau"], "tau") for r in records]
    p50s = [_as_positive(r["ms_per_round_p50"], "ms_per_round_p50") for r in records]
    return {
        "n": len(records),
        "tau_mean": statistics.fmean(taus),
        "tau_median": statistics.median(taus),
        "tau_min": min(taus),
        "tau_max": max(taus),
        "ms_per_round_p50_median": statistics.median(p50s),
    }
