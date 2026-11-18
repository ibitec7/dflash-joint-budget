"""The frozen prompt set: the unit of comparison for every arm in the grid.

Usage:
    uv run python scripts/prompts.py --freeze-all
    uv run python scripts/prompts.py --benchmark mtbench --show

Protocol (docs/EXPERIMENTS.md §4, rule 1): prompts are fixed and committed *before* the first
GPU run, because every comparison in this project is within-set. `freeze()` writes the set once;
`load_frozen()` is how the harness reads it afterwards.

Offline rule: the repo must work with no network. `build()` tries the real Hugging Face dataset
first and falls back to the committed fixture on *any* exception, marking every returned dict
with `"source": "hf"` or `"source": "fixture"` so a run can never silently claim provenance it
does not have. The fixture stores the full pool, so a fallback build reproduces the same order
that an online build produces for the same seed.

Prompt text is the benchmark's own wording, unmodified (no chat template, no instruction
wrapper). A template would be a second free parameter and every arm would inherit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BENCHMARKS = ("math500", "mtbench")

# PLAN.md §4: decode is greedy, max 256 new tokens for both benchmarks.
MAX_NEW_TOKENS = 256

# Ordered candidates, best first. `lighteval/MATH-500` is the canonically cited repo and is
# tried first; it has since been removed from the Hub, so the mirrors below carry the same 500
# problems. `lmsys/mt_bench` is likewise not a Hub dataset (the 80 questions live in the
# FastChat repo), so `HuggingFaceH4/mt_bench_prompts` is the live mirror of those 80.
HF_SOURCES: dict[str, tuple[str, ...]] = {
    "math500": ("lighteval/MATH-500", "HuggingFaceH4/MATH-500", "di-zhang-fdu/MATH500"),
    "mtbench": ("lmsys/mt_bench", "HuggingFaceH4/mt_bench_prompts", "lmsys/mt_bench_human_judgments"),
}

HF_SPLITS = {"math500": "test", "mtbench": "train"}

# Below this many usable rows a candidate is treated as the wrong dataset and the next one runs.
MIN_POOL = {"math500": 200, "mtbench": 80}

OFFLINE_ENV = "DFLASH_PROMPTS_OFFLINE"

_POOL_CACHE: dict[str, tuple[list[dict], str]] = {}


# --------------------------------------------------------------------------- helpers

def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _slug(text: str) -> str:
    """MATH-500 unique_ids look like 'test/prealgebra/1991.json'; keep ids path-safe."""
    clean = str(text).strip().removesuffix(".json")
    for ch in "/\\ :[]()":
        clean = clean.replace(ch, "-")
    return clean or "unknown"


def _offline_requested() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _check_benchmark(benchmark: str) -> str:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}")
    return benchmark


def fixture_path(benchmark: str) -> Path:
    return DATA_DIR / f"frozen_{_check_benchmark(benchmark)}.json"


def clear_cache() -> None:
    """Drop the cached pool. Tests and the CLI use this before forcing a source."""
    _POOL_CACHE.clear()


# --------------------------------------------------------------------------- pools

def _row_to_item(benchmark: str, row: dict, index: int) -> dict | None:
    """One dataset row -> {"id", "prompt"}, or None if unusable."""
    try:
        if benchmark == "math500":
            text = row.get("problem")
            uid = row.get("unique_id")
            key = f"math500-{_slug(uid)}" if uid else f"math500-{_short_hash(str(text))}"
        else:
            raw = row.get("prompt")
            text = raw[0] if isinstance(raw, list) and raw else raw
            pid = row.get("prompt_id", row.get("question_id"))
            key = f"mtbench-{pid}" if pid is not None else f"mtbench-{_short_hash(str(text))}"
    except (KeyError, TypeError, AttributeError, IndexError):
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return {"id": str(key), "prompt": text.strip()}


def _load_hf(benchmark: str) -> list[dict]:
    """Fetch the real dataset. Raises on any failure so the caller can fall back."""
    from datasets import load_dataset  # imported lazily: the offline path must not need it

    errors: list[str] = []
    for repo in HF_SOURCES[benchmark]:
        try:
            ds = load_dataset(repo, split=HF_SPLITS[benchmark])
        except Exception as exc:  # noqa: BLE001 - every failure means "try the next source"
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
            continue
        pool, seen = [], set()
        for i, row in enumerate(ds):
            item = _row_to_item(benchmark, row, i)
            if item is None or item["id"] in seen:
                continue
            seen.add(item["id"])
            pool.append(item)
        if len(pool) < MIN_POOL[benchmark]:
            errors.append(f"{repo}: only {len(pool)} usable rows, need {MIN_POOL[benchmark]}")
            continue
        return sorted(pool, key=lambda it: it["id"])
    raise RuntimeError(f"no usable Hugging Face source for {benchmark}: " + "; ".join(errors))


def _load_fixture(benchmark: str) -> list[dict]:
    """The committed pool, in canonical order. Raises if it is missing or malformed."""
    path = fixture_path(benchmark)
    if not path.is_file():
        raise FileNotFoundError(f"no fixture at {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    pool = doc.get("pool") or doc.get("prompts")
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"{path.name}: no usable pool")
    out = []
    for item in pool:
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
            raise ValueError(f"{path.name}: malformed pool entry {item!r}")
        out.append({"id": str(item["id"]), "prompt": str(item["prompt"])})
    return sorted(out, key=lambda it: it["id"])


def _pool(benchmark: str) -> tuple[list[dict], str]:
    """(canonical pool, "hf" | "fixture"). Never raises for a network reason."""
    benchmark = _check_benchmark(benchmark)
    if benchmark in _POOL_CACHE:
        pool, source = _POOL_CACHE[benchmark]
        return list(pool), source

    pool: list[dict] = []
    source = "fixture"
    if not _offline_requested():
        try:
            pool = _load_hf(benchmark)
            source = "hf"
        except Exception:  # noqa: BLE001 - the offline rule: never crash on a network failure
            pool = []
    if not pool:
        pool, source = _load_fixture(benchmark), "fixture"

    _POOL_CACHE[benchmark] = (list(pool), source)
    return list(pool), source


# --------------------------------------------------------------------------- public API

def build(benchmark: str, n: int = 200, seed: int = 0) -> list[dict]:
    """Return [{"id": str, "prompt": str, "max_new_tokens": int}, ...] deterministically.

    Each dict also carries "source": "hf" or "fixture". Ordering is a seeded permutation of the
    canonically-sorted pool, so the same (benchmark, n, seed) yields the same list on any
    machine and in any process -- independent of hash randomisation and of dataset row order.

    When n exceeds the pool (mtbench has 80 curated questions) the pool is repeated, and every
    repeat gets a distinct id ("-r1", "-r2", ...) so per-prompt results never collide.
    """
    benchmark = _check_benchmark(benchmark)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    pool, source = _pool(benchmark)

    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)

    out: list[dict] = []
    rep = 0
    while len(out) < n:
        for idx in order:
            if len(out) >= n:
                break
            item = pool[idx]
            out.append({
                "id": item["id"] if rep == 0 else f"{item['id']}-r{rep}",
                "prompt": item["prompt"],
                "max_new_tokens": MAX_NEW_TOKENS,
                "source": source,
            })
        rep += 1
    return out


def freeze(benchmark: str, n: int, seed: int, out: Path, *, force: bool = False) -> Path:
    """Write the frozen set to `out` and return that path. Byte-stable for a given pool.

    The file holds both the resolved `prompts` (what the harness consumes) and the full `pool`
    in canonical order, so an offline `build()` can reproduce any (n, seed) later. No timestamp
    is recorded: a wall clock would make two freezes of the same set differ byte for byte.

    A set that is already frozen is never silently replaced. Writing the same set again is a
    no-op (the file keeps its original provenance label); writing a *different* set raises
    unless `force=True`, because the protocol's whole point is that this file cannot move.
    """
    benchmark = _check_benchmark(benchmark)
    out = Path(out)
    pool, source = _pool(benchmark)
    items = build(benchmark, n, seed)

    if out.exists() and not force:
        try:
            existing = load_frozen(out)
        except (ValueError, json.JSONDecodeError):
            existing = None
        if existing is not None:
            if [(it["id"], it["prompt"]) for it in existing] == [
                (it["id"], it["prompt"]) for it in items
            ]:
                return out
            raise FileExistsError(
                f"{out} already holds a different frozen prompt set; "
                "pass force=True (--force) only if that is deliberate"
            )

    doc = {
        "benchmark": benchmark,
        "n": n,
        "seed": seed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pool_size": len(pool),
        "repeated": n > len(pool),
        "repeats": -(-n // len(pool)),
        "source": source,
        "generator": "scripts/prompts.py",
        "note": (
            "Frozen before any GPU run (docs/EXPERIMENTS.md §4 rule 1). "
            "Do not regenerate: every reported comparison is within this set."
        ),
        "prompts": items,
        "pool": pool,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def load_frozen(path: Path) -> list[dict]:
    """Read a frozen file written by `freeze()` and return its prompt list verbatim."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    items = doc.get("prompts")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path.name}: no prompts")
    for item in items:
        missing = {"id", "prompt", "max_new_tokens"} - item.keys()
        if missing:
            raise ValueError(f"{path.name}: entry {item.get('id')!r} missing {sorted(missing)}")
    return items


# --------------------------------------------------------------------------- CLI

def _describe(benchmark: str, path: Path) -> str:
    items = load_frozen(path)
    ids = [it["id"] for it in items]
    doc = json.loads(path.read_text(encoding="utf-8"))
    sources = sorted({it.get("source", "?") for it in items})
    return (
        f"{benchmark:<8} {len(items):>3} prompts  source={','.join(sources):<7} "
        f"pool={doc['pool_size']:<3} unique_ids={len(set(ids)):<3} "
        f"repeated={str(doc['repeated']).lower():<5} -> {path}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze / inspect the benchmark prompt sets.")
    ap.add_argument("--freeze-all", action="store_true", help="freeze every benchmark to data/")
    ap.add_argument("--freeze", action="store_true", help="freeze the selected --benchmark")
    ap.add_argument("--force", action="store_true", help="overwrite an existing, different set")
    ap.add_argument("--benchmark", choices=BENCHMARKS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--offline", action="store_true", help="skip Hugging Face, use the fixture")
    ap.add_argument("--show", action="store_true", help="print the first prompts")
    args = ap.parse_args(argv)

    if args.offline:
        os.environ[OFFLINE_ENV] = "1"
        clear_cache()

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR
    if args.freeze_all:
        targets = BENCHMARKS
    elif args.benchmark:
        targets = (args.benchmark,)
    else:
        ap.error("give --freeze-all or --benchmark")

    # Inspecting is read-only: only --freeze/--freeze-all may write. A browse must never
    # rewrite the frozen file, or the provenance label and the set itself could churn.
    for benchmark in targets:
        path = out_dir / f"frozen_{benchmark}.json"
        if args.freeze or args.freeze_all:
            clear_cache()
            path = freeze(benchmark, args.n, args.seed, path, force=args.force)
        if not path.is_file():
            print(f"{benchmark:<8} no frozen set at {path}")
            continue
        print(_describe(benchmark, path))
        if args.show:
            for item in load_frozen(path)[:3]:
                print(f"    {item['id']}  {item['prompt'][:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
