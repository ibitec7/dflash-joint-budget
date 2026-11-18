"""Run the measurement grid: arms x concurrency x benchmark x seed -> results/*.json.

One cell = one Engine, one config, one batch chunk. The engine already aggregates per-round
latency at the source (its serving loop), so records are built from `Engine.stats()` and
validated with `metrics.validate_record`; `RunRecorder` exists for backends that emit
per-round metadata and is exercised by its own tests.

Usage:
    uv run python scripts/run_grid.py --dry-run
    uv run python scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \\
        --arms A0,A1,A3,A5 --benchmarks math500 --concurrency 1,4,8,16,32

A4 (per-round verify-length selection) is refused until E0 confirms the backend exposes a
per-request verify length. Inventing that path before the probe is how a grid fills with
numbers that no backend produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import prompts as promptlib  # pyright: ignore[reportMissingImports]

import metrics  # pyright: ignore[reportMissingImports]
from arms import ARMS  # pyright: ignore[reportMissingImports]
from serve import Engine, EngineConfig  # pyright: ignore[reportMissingImports]

DEFAULT_ARMS = ("A0", "A1", "A3", "A5")
DEFAULT_C = (1, 4, 8, 16, 32)
DEFAULT_BENCH = ("math500", "mtbench")
BENCH_DIR = ROOT / "data"


def cells(arms: tuple[str, ...], conc: tuple[int, ...]) -> list[tuple[str, int]]:
    out = []
    for arm in arms:
        for c in conc:
            if arm == "A2" and c != 32:
                continue          # A2 is the length sweep, only at serving batch
            out.append((arm, c))
    return out


def _config(arm: str, args: argparse.Namespace) -> EngineConfig:
    spec = ARMS[arm]
    if spec.selector == "marginal":
        raise SystemExit(
            f"{arm} needs per-round verify-length selection, which is gated on E0 "
            "(does this backend expose a per-request verify length?). Run E0 first, then "
            "add the selector path to EngineConfig."
        )
    ar = arm == "A5"
    return EngineConfig(
        model=args.model,
        drafter="" if ar else (args.drafter or "dry-run/placeholder"),
        backend=args.backend,
        draft_width=spec.draft_width or 16,
        verify_columns=spec.verify_columns or 17,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )


def _chunks(engine: Engine, texts: list[str], c: int) -> list[str]:
    """Generate over the prompt set in chunks of size c on an already-loaded engine."""
    outs: list[str] = []
    for i in range(0, len(texts), c):
        outs.extend(engine.generate(texts[i:i + c]))
    return outs


def record(arm: str, bench: str, c: int, seed: int, cfg: EngineConfig, st: dict,
           exact_match: float, n_prompts: int, ar_tok: float) -> dict:
    rec = {
        "arm": arm,
        "benchmark": bench,
        "concurrency": c,
        "seed": seed,
        "draft_width": cfg.draft_width,
        "verify_columns": cfg.verify_columns,
        "prompts": n_prompts,
        "tau": st["tau_mean"],
        "tok_per_s": st["tok_per_s"],
        "ar_tok_per_s": ar_tok,
        "ms_per_round_p50": st["ms_per_round_p50"],
        "ms_per_round_p95": st["ms_per_round_p95"],
        "exact_match_ar": exact_match,
        "graph_recaptures": int(st.get("graph_recaptures") or 0),
    }
    metrics.validate_record(rec)
    return rec


def agreement(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no GPU, synthetic numbers")
    ap.add_argument("--backend", default="sglang", choices=("sglang", "vllm", "transformers"))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--drafter", default="", help="HF id of the DFlash drafter (required unless dry-run)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--benchmarks", default=",".join(DEFAULT_BENCH))
    ap.add_argument("--concurrency", default=",".join(map(str, DEFAULT_C)))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--prompts", type=int, default=0, help="cap per benchmark (0 = all frozen)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dump-texts", action="store_true",
                    help="also write <record>.texts.json (needed for cross-run determinism)")
    args = ap.parse_args(argv)

    if args.dry_run:
        pass
    elif not args.drafter:
        ap.error("--drafter is required unless --dry-run")

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    conc = tuple(int(x) for x in args.concurrency.split(","))
    benches = tuple(b.strip() for b in args.benchmarks.split(",") if b.strip())
    seeds = [int(x) for x in args.seeds.split(",")]

    out = Path(args.out)
    if args.dry_run and "dryrun" not in out.name:
        out = out.with_name(out.name + "_dryrun")
        print(f"[dry-run] writing to {out} so synthetic numbers cannot be mistaken for data")
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (out / "SYNTHETIC_DATA_DO_NOT_USE").write_text(
            "Records here come from Engine(dry_run=True). They exercise the plumbing and must "
            "never enter a report. decide.py refuses any directory carrying this marker.\n"
        )

    written = 0
    for bench in benches:
        fx = BENCH_DIR / f"frozen_{bench}.json"
        items = promptlib.load_frozen(fx)
        if args.prompts:
            items = items[: args.prompts]
        texts = [it["prompt"] for it in items]
        print(f"\n{bench}: {len(texts)} prompts from {fx.name}")

        for seed in seeds:
            # One engine per (arm, seed): the verify length is a startup flag, so a new
            # engine is the only way to change it, but concurrency is not.
            ar_cfg = _config("A5", args)
            ar_outs: dict[int, list[str]] = {}
            ar_tok: dict[int, float] = {}
            ar_st: dict[int, dict] = {}
            engine = Engine(ar_cfg, dry_run=args.dry_run)
            try:
                for c in conc:
                    engine.reset_stats()
                    ar_outs[c] = _chunks(engine, texts, c)
                    ar_st[c] = engine.stats()
                    ar_tok[c] = ar_st[c]["tok_per_s"]
            finally:
                engine.close()
            if "A5" in arms:
                for c in conc:
                    rec = record("A5", bench, c, seed, ar_cfg, ar_st[c], 1.0, len(texts), ar_tok[c])
                    (out / f"A5_{bench}_c{c}_s{seed}.json").write_text(json.dumps(rec, indent=2))
                    if args.dump_texts:
                        (out / f"A5_{bench}_c{c}_s{seed}.texts.json").write_text(
                            json.dumps(ar_outs[c]))
                    written += 1
                    print(f"  c={c:<3} A5  tau={rec['tau']:.2f} tok/s={rec['tok_per_s']:.1f}")

            for arm in arms:
                if arm == "A5":
                    continue
                cfg = _config(arm, args)
                engine = Engine(cfg, dry_run=args.dry_run)
                try:
                    for c in conc:
                        engine.reset_stats()
                        outs = _chunks(engine, texts, c)
                        st = engine.stats()
                        rec = record(arm, bench, c, seed, cfg, st,
                                     agreement(outs, ar_outs[c]), len(texts), ar_tok[c])
                        p = out / f"{arm}_{bench}_c{c}_s{seed}.json"
                        if p.exists() and not args.force:
                            raise SystemExit(
                                f"{p} exists; rerun with --force to overwrite measurements")
                        p.write_text(json.dumps(rec, indent=2))
                        if args.dump_texts:
                            (out / f"{arm}_{bench}_c{c}_s{seed}.texts.json").write_text(
                                json.dumps(outs))
                        written += 1
                        print(f"  c={c:<3} {arm} tau={rec['tau']:.2f} "
                              f"tok/s={rec['tok_per_s']:.1f} match={rec['exact_match_ar']:.3f}")
                finally:
                    engine.close()

    print(f"\n{written} records -> {out}")
    if args.dry_run:
        print("SYNTHETIC: plumbing only. Real measurements need --backend and --drafter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
