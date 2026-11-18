"""E0 feasibility probe: three questions that gate the grid, answered with evidence.

E0 runs once on a fresh box and decides whether A4 (per-round verify-length selection) can
exist at all. Every answer here is produced by loading something or introspecting something;
none is inferred from the paper or from a model card. When a probe cannot get evidence it
says UNKNOWN and prints why, because a confident wrong YES costs a whole grid.

Questions and their evidence:

  Q1  Does the DFlash drafter load and emit tokens through the transformers reference path?
      -> builds ``serve.Engine(backend="transformers", drafter="z-lab/Qwen3-8B-DFlash-b16")``,
         generates one short prompt, prints the first 40 chars, tokens/s and tau.
  Q2  Does the serving path expose a PER-REQUEST verify length (a field on the request
      object, not a process-startup flag)?
      -> ``inspect.signature`` on the generate entry points and on the request/config
         classes, ``hasattr`` checks for each candidate field name, plus a bounded grep of
         the installed package for every candidate symbol. The exact callable, its signature,
         the field list and the file:line hits are printed.
  Q3  What is ms(17 verify columns) / ms(9 verify columns) at batch 32?
      -> only measurable if Q2 is YES (otherwise the two arms are two processes with two
         different startups, which is A0-vs-A1, not a within-run column term), so Q3 says
         NOT MEASURABLE and names the blocked premise. With Q2 YES it builds both engines at
         batch 32 and reports the ratio of ``ms_per_round_p50``.

Output is marker lines, so a caller greps instead of tailing:

    E0-Q1: YES|NO <reason>
    E0-Q1.D<n>: <evidence>
    E0-Q2: YES|NO|UNKNOWN <reason>
    E0-Q2.D<n>: <evidence>
    E0-Q3: <ratio>x|NOT MEASURABLE|UNKNOWN <reason>
    E0-DONE rc=<n>

rc: 0 probe completed (any verdict is a result), 3 Q1 failed (no reference path at all).

Usage:
    uv run python scripts/e0_probe.py --dry-run                 # laptop: plumbing only
    source ~/dflash_env.sh && $PY scripts/e0_probe.py --backend sglang --json-out results/e0.json

``--dry-run`` needs no GPU, no weights and no network: Q1/Q3 use ``Engine(dry_run=True)`` and
Q2 reports a synthetic YES, each labelled ``dry-run`` in the line itself so no dry-run number
can be mistaken for a measurement. No dry-run number may enter ``results/``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import os
import sys
import time
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from serve import BackendUnavailable, Engine, EngineConfig

MODEL = "Qwen/Qwen3-8B"
DRAFTER = "z-lab/Qwen3-8B-DFlash-b16"
PROMPT = "How many positive whole-number divisors does 196 have?"
FROZEN = ROOT / "data" / "frozen_math500.json"

# Candidate per-request field names, widest first. None of these is assumed to exist: each is
# looked for on a real object and in the installed source, and a miss is reported as a miss.
VERIFY_FIELD_NAMES = (
    "verify_len",
    "verify_columns",
    "verify_length",
    "num_draft_tokens",
    "num_speculative_tokens",
    "speculative_num_draft_tokens",
)

# (label, module, attribute chain, role). role="request" objects are what a single call may
# vary per request; role="startup" objects are fixed when the engine is constructed.
PROBE_TARGETS = (
    ("sglang.Engine.generate", "sglang", ("Engine", "generate"), "entrypoint"),
    ("sglang SamplingParams", "sglang.srt.sampling.sampling_params", ("SamplingParams",), "request"),
    ("sglang ServerArgs", "sglang.srt.server_args", ("ServerArgs",), "startup"),
    ("vllm.LLM.generate", "vllm", ("LLM", "generate"), "entrypoint"),
    ("vllm SamplingParams", "vllm", ("SamplingParams",), "request"),
    ("vllm SpeculativeConfig", "vllm.config", ("SpeculativeConfig",), "startup"),
)

_SKIP_DIRS = {"__pycache__", "tests", "test", ".git", "third_party", "node_modules"}
_MAX_FILES = 900
_MAX_FILE_BYTES = 2_000_000
_MAX_TOTAL_BYTES = 60_000_000
_MAX_HITS = 4


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def emit(key: str, text: str, out: Any = None) -> None:
    """Print one marker line. Key is ``E0-Q<n>`` for verdicts and ``E0-Q<n>.D<k>`` for detail."""
    stream = out or sys.stdout
    stream.write(f"{key}: {text}\n")
    stream.flush()


# --------------------------------------------------------------------------------------
# Introspection helpers (never raise on a missing attribute)
# --------------------------------------------------------------------------------------


def resolve_chain(module_name: str, chain: tuple[str, ...]) -> tuple[Any, str]:
    """Import ``module_name`` and walk ``chain``. Returns (object|None, status string)."""
    try:
        obj: Any = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - a broken package can raise anything at import
        return None, f"import {module_name} failed: {type(exc).__name__}: {exc}"
    for part in chain:
        if not hasattr(obj, part):
            return None, f"{module_name}.{'.'.join(chain)} missing attribute {part!r}"
        obj = getattr(obj, part)
    return obj, "ok"


def signature_text(obj: Any) -> str | None:
    """``inspect.signature`` as a one-line string, or None when the object has none."""
    if obj is None:
        return None
    try:
        return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', '?')}" + str(
            inspect.signature(obj)
        )
    except (TypeError, ValueError) as exc:
        return f"<no signature: {type(exc).__name__}: {exc}>"


def parameter_names(obj: Any) -> list[str]:
    """Parameter names of a callable, or [] — used for hasattr-style field questions."""
    if obj is None:
        return []
    try:
        return list(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return []


def field_names(obj: Any) -> list[str]:
    """Declared fields of a dataclass / pydantic model / annotated class."""
    if obj is None:
        return []
    if is_dataclass(obj):
        return [f.name for f in dataclass_fields(obj)]
    model_fields = getattr(obj, "model_fields", None)  # pydantic v2
    if isinstance(model_fields, dict):
        return sorted(model_fields)
    annotations = getattr(obj, "__annotations__", None)
    if isinstance(annotations, dict) and annotations:
        return sorted(annotations)
    return parameter_names(obj)


def grep_tree(root: Path, names: tuple[str, ...]) -> dict[str, list[str]]:
    """Bounded grep for each symbol under ``root``; returns {name: ["path:line", ...]}.

    Bounded on purpose: sglang is a large tree and E0 must not spend a GPU hour's wall clock
    reading it. Missing symbols come back as empty lists, which is itself the evidence.
    """
    hits: dict[str, list[str]] = {n: [] for n in names}
    if not root.is_dir():
        return hits
    budget = _MAX_TOTAL_BYTES
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > _MAX_FILE_BYTES or budget <= 0:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            budget -= size
            seen += 1
            if seen > _MAX_FILES:
                return hits
            for symbol in names:
                idx = text.find(symbol)
                if idx < 0 or len(hits[symbol]) >= _MAX_HITS:
                    continue
                line = text.count("\n", 0, idx) + 1
                hits[symbol].append(f"{path}:{line}")
    return hits


def safe_close(engine: Any) -> None:
    """Close an engine without letting a close() failure mask the probe's verdict."""
    if engine is None:
        return
    with contextlib.suppress(Exception):
        engine.close()


def _hit_kind(path_and_line: str) -> str:
    """Classify a grep hit by module name: request-path code vs startup/CLI code."""
    path = path_and_line.split(":")[0].lower()
    if any(k in path for k in ("server_args", "launch", "cli", "/args", "arguments")):
        return "startup"
    if any(k in path for k in ("sampling", "request", "verify", "schedul", "spec")):
        return "request"
    return "other"


# --------------------------------------------------------------------------------------
# Q1 — does the drafter serve at all through the transformers reference path?
# --------------------------------------------------------------------------------------


def probe_q1(
    dry_run: bool, model: str = MODEL, drafter: str = DRAFTER, max_new_tokens: int = 32
) -> dict:
    cfg = EngineConfig(
        model=model,
        drafter=drafter,
        backend="transformers",
        draft_width=16,
        verify_columns=17,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    engine = None
    t0 = time.perf_counter()
    try:
        engine = Engine(cfg, dry_run=dry_run)
        texts = engine.generate([PROMPT])
        stats = engine.stats()
    except BackendUnavailable as exc:
        return {
            "answer": "NO",
            "reason": f"backend unavailable: {str(exc).splitlines()[0]}",
            "detail": str(exc),
            "dry_run": dry_run,
        }
    except Exception as exc:  # noqa: BLE001 - any load/generate failure is a Q1 NO, reported
        return {
            "answer": "NO",
            "reason": f"{type(exc).__name__}: {exc}",
            "detail": f"Engine or generate raised {type(exc).__name__}: {exc}",
            "dry_run": dry_run,
        }
    finally:
        safe_close(engine)

    text = (texts[0] if texts else "") or ""
    tokens = int(stats.get("tokens_generated") or 0)
    answer = "YES" if tokens > 0 and text.strip() else "NO"
    return {
        "answer": answer,
        "reason": "" if answer == "YES" else "engine returned no tokens",
        "dry_run": dry_run,
        "text_head": text[:40],
        "tokens": tokens,
        "tok_per_s": stats.get("tok_per_s"),
        "tau": stats.get("tau_mean"),
        "rounds_source": stats.get("ms_per_round_source"),
        "ms_per_round_p50": stats.get("ms_per_round_p50"),
        "wall_s": round(time.perf_counter() - t0, 3),
        "verify_control": stats.get("verify_control"),
        "synthetic": stats.get("synthetic"),
    }


# --------------------------------------------------------------------------------------
# Q2 — is the verify length per-request, or only per-process?
# --------------------------------------------------------------------------------------


def probe_q2(dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "answer": "YES",
            "dry_run": True,
            "reason": "dry-run: synthetic verdict, nothing imported or introspected",
            "callable": "<dry-run>",
            "signature": "<dry-run>",
            "request_fields": [],
            "fields_found": [],
            "hits": {},
        }

    rows: list[dict] = []
    request_field: str | None = None
    request_label = ""
    startup_field: str | None = None
    startup_label = ""
    roots: dict[str, Path] = {}
    imported_any = False

    for label, module_name, chain, role in PROBE_TARGETS:
        obj, status = resolve_chain(module_name, chain)
        if obj is None:
            rows.append({"label": label, "role": role, "status": status})
            continue
        imported_any = True
        names = field_names(obj) if role != "entrypoint" else parameter_names(obj)
        matched = [n for n in VERIFY_FIELD_NAMES if n in set(names)]
        rows.append(
            {
                "label": label,
                "role": role,
                "status": status,
                "signature": signature_text(obj),
                "n_params": len(names),
                "matched": matched,
                "names": names,
            }
        )
        if matched:
            if role == "request" and request_field is None:
                request_field, request_label = matched[0], label
            elif role == "startup" and startup_field is None:
                startup_field, startup_label = matched[0], label
        # Package root for the grep, taken from the module that actually imported.
        top = module_name.split(".")[0]
        if top not in roots:
            with contextlib.suppress(Exception):  # grep is best-effort evidence
                f = getattr(importlib.import_module(top), "__file__", None)
                if f:
                    roots[top] = Path(f).resolve().parent

    # Does the transformers stack even have the entry point? It does not: spec_generate ships
    # in the checkpoint's remote code, so its signature is only readable after a load (Q1).
    tf, _ = resolve_chain("transformers", ("PreTrainedModel", "spec_generate"))
    has_tf_spec = tf is not None

    hits: dict[str, dict[str, list[str]]] = {}
    for pkg, root in roots.items():
        found = grep_tree(root, VERIFY_FIELD_NAMES)
        hits[pkg] = {k: v for k, v in found.items() if v}

    if not imported_any:
        return {
            "answer": "UNKNOWN",
            "reason": (
                "none of sglang/vllm is importable in this interpreter, so no request object "
                "could be introspected; run E0 inside the served venv"
            ),
            "rows": rows,
            "hits": hits,
            "has_tf_spec_generate": has_tf_spec,
            "fields_found": [],
            "request_fields": [],
            "callable": "<not importable>",
            "signature": "<not importable>",
        }

    if request_field:
        # Found on a per-request object AND referenced by non-CLI source -> static YES.
        symbol_hits = {
            pkg: [h for name in hits.get(pkg, {}) if name == request_field for h in hits[pkg][name]]
            for pkg in hits
        }
        request_path_hits = [
            h
            for hl in symbol_hits.values()
            for h in hl
            if _hit_kind(h) in ("request", "other")
        ]
        answer = "YES"
        reason = (
            f"{request_label} declares {request_field!r}"
            + (f"; {len(request_path_hits)} source hit(s) outside CLI/args code" if request_path_hits
               else "; no non-CLI source hit, so honouring it is unproven until a run shows "
               "spec_verify_ct changing")
        )
    elif startup_field:
        answer = "NO"
        reason = (
            f"the only match is {startup_field!r} on {startup_label}, which is fixed when the "
            "engine is constructed: a per-request length would need two processes"
        )
    else:
        answer = "UNKNOWN"
        reason = (
            "no candidate field exists on any introspected request object, and none appears in "
            "the installed source; the knob may be named something this probe does not know"
        )

    return {
        "answer": answer,
        "reason": reason,
        "rows": rows,
        "hits": hits,
        "request_field": request_field,
        "startup_field": startup_field,
        "fields_found": [r for r in (request_field, startup_field) if r],
        "request_fields": [request_field] if request_field else [],
        "has_tf_spec_generate": has_tf_spec,
        "callable": next((r["signature"] for r in rows if r.get("signature")), "<none>"),
        "signature": next((r["signature"] for r in rows if r.get("signature")), "<none>"),
        "dry_run": False,
    }


# --------------------------------------------------------------------------------------
# Q3 — ms(17 columns) / ms(9 columns) at batch 32
# --------------------------------------------------------------------------------------


def batch_prompts(n: int) -> list[str]:
    """Frozen math500 prompts (deterministic, offline); synthetic only if the file is absent."""
    try:
        doc = json.loads(FROZEN.read_text(encoding="utf-8"))
        items = doc.get("prompts") or doc.get("pool") or []
        prompts = [it["prompt"] for it in items if it.get("prompt")]
    except (OSError, ValueError, KeyError, TypeError):
        prompts = []
    if not prompts:
        prompts = [f"What is {17 * i + 3} times {23 + i}, and why?" for i in range(n)]
    while len(prompts) < n:
        prompts.extend(prompts[: n - len(prompts)])
    return prompts[:n]


def _ms_per_round(
    columns: int, prompts: list[str], backend: str, dry_run: bool, max_new_tokens: int
) -> tuple[float | None, dict, str]:
    """Build one engine at one verify width, return (ms_per_round_p50, stats, status)."""
    cfg = EngineConfig(
        model=MODEL,
        drafter=DRAFTER,
        backend=backend,
        draft_width=16,
        verify_columns=columns,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    engine = None
    try:
        engine = Engine(cfg, dry_run=dry_run)
        engine.generate(prompts)
        stats = engine.stats()
    except BackendUnavailable as exc:
        return None, {}, f"backend unavailable: {str(exc).splitlines()[0]}"
    except Exception as exc:  # noqa: BLE001 - reported as evidence, never a traceback
        return None, {}, f"{type(exc).__name__}: {exc}"
    finally:
        safe_close(engine)
    ms = stats.get("ms_per_round_p50")
    if not ms:
        return None, stats, "backend reported no per-round timing"
    return float(ms), stats, "ok"


def probe_q3(
    q2_answer: str,
    dry_run: bool,
    backend: str = "sglang",
    batch: int = 32,
    max_new_tokens: int = 32,
) -> dict:
    if q2_answer != "YES":
        return {
            "answer": "NOT MEASURABLE",
            "reason": (
                f"Q2 is {q2_answer}: without a per-request verify length the 17- and 9-column "
                "arms are two processes with two different startups, so the ratio would mix the "
                "column term with startup state (that pair is A0 vs A1, not the within-run term "
                "Q3 asks for)"
            ),
            "batch": batch,
            "dry_run": dry_run,
        }

    prompts = batch_prompts(batch)
    ms17, st17, s17 = _ms_per_round(17, prompts, backend, dry_run, max_new_tokens)
    ms9, st9, s9 = _ms_per_round(9, prompts, backend, dry_run, max_new_tokens)
    if ms17 is None or ms9 is None or ms9 <= 0:
        return {
            "answer": "UNKNOWN",
            "reason": f"measurement failed: columns=17 {s17}; columns=9 {s9}",
            "ms17": ms17,
            "ms9": ms9,
            "batch": batch,
            "dry_run": dry_run,
        }
    return {
        "answer": round(ms17 / ms9, 4),
        "reason": "",
        "ms17": round(ms17, 4),
        "ms9": round(ms9, 4),
        "batch": batch,
        "backend": backend,
        "tau17": st17.get("tau_mean"),
        "tau9": st9.get("tau_mean"),
        "tokens17": st17.get("tokens_generated"),
        "tokens9": st9.get("tokens_generated"),
        "dry_run": dry_run,
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def report_q1(r: dict, out: Any = None) -> None:
    tag = " (dry-run synthetic)" if r.get("dry_run") else ""
    emit("E0-Q1", f"{r['answer']}{tag}", out)
    if r.get("reason"):
        emit("E0-Q1.R", r["reason"], out)
    if r.get("dry_run"):
        emit("E0-Q1.D0", "dry-run: Engine(dry_run=True), no weights were loaded", out)
    if r.get("text_head") is not None:
        emit("E0-Q1.D1", f"first-40-chars={r['text_head']!r}", out)
        emit(
            "E0-Q1.D2",
            f"tokens={r['tokens']} tok/s={r.get('tok_per_s')} tau={r.get('tau')} "
            f"ms/round_p50={r.get('ms_per_round_p50')} wall_s={r.get('wall_s')}",
            out,
        )
        emit(
            "E0-Q1.D3",
            f"backend=transformers model={MODEL} drafter={DRAFTER} "
            f"rounds_source={r.get('rounds_source')} verify_control={r.get('verify_control')}",
            out,
        )


def report_q2(r: dict, out: Any = None) -> None:
    tag = " (dry-run synthetic)" if r.get("dry_run") else ""
    emit("E0-Q2", f"{r['answer']}{tag}", out)
    if r.get("reason"):
        emit("E0-Q2.R", r["reason"], out)
    if r.get("dry_run"):
        emit("E0-Q2.D0", "dry-run: nothing imported; the verdict is synthetic", out)
        return
    n = 1
    for row in r.get("rows", []):
        if "signature" in row:
            emit(
                f"E0-Q2.D{n}",
                f"{row['label']} [{row['role']}] sig={row['signature']} "
                f"matched={row['matched'] or 'none'}",
                out,
            )
        else:
            emit(f"E0-Q2.D{n}", f"{row['label']} [{row['role']}] unavailable: {row['status']}", out)
        n += 1
    for pkg, found in r.get("hits", {}).items():
        for symbol, places in found.items():
            emit(
                f"E0-Q2.D{n}",
                f"grep {pkg}: {symbol!r} at {places[0]} "
                f"(kind={_hit_kind(places[0])}, hits={len(places)})",
                out,
            )
            n += 1
    if not r.get("hits"):
        emit(f"E0-Q2.D{n}", "grep: no candidate symbol found in any installed package", out)
        n += 1
    emit(
        f"E0-Q2.D{n}",
        f"transformers.PreTrainedModel.spec_generate present="
        f"{r.get('has_tf_spec_generate')} (DFlash entry point lives in the checkpoint's remote "
        "code, readable only after a load)",
        out,
    )


def report_q3(r: dict, out: Any = None) -> None:
    tag = " (dry-run synthetic)" if r.get("dry_run") else ""
    if isinstance(r.get("answer"), float):
        emit("E0-Q3", f"{r['answer']:.4f}x{tag} at batch={r['batch']}", out)
        emit(
            "E0-Q3.D1",
            f"ms_per_round_p50: 17cols={r['ms17']} 9cols={r['ms9']} "
            f"tau17={r.get('tau17')} tau9={r.get('tau9')} backend={r.get('backend')}",
            out,
        )
    else:
        emit("E0-Q3", f"{r['answer']}{tag}", out)
        if r.get("reason"):
            emit("E0-Q3.R", r["reason"], out)
        if r.get("ms17") is not None:
            emit("E0-Q3.D1", f"partial: ms17={r['ms17']} ms9={r['ms9']} batch={r['batch']}", out)


def probe_all(dry_run: bool, backend: str = "sglang", batch: int = 32) -> dict:
    """Run Q1..Q3 and return the raw result dict. Importable and GPU-free with dry_run=True."""
    q1 = probe_q1(dry_run)
    q2 = probe_q2(dry_run)
    q3 = probe_q3(q2["answer"], dry_run, backend=backend, batch=batch)
    rc = 0 if q1["answer"] == "YES" else 3
    return {"dry_run": dry_run, "q1": q1, "q2": q2, "q3": q3, "rc": rc}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E0: can this box serve DFlash, and is L per-request?")
    ap.add_argument("--dry-run", action="store_true", help="no GPU/weights/network; synthetic")
    ap.add_argument("--backend", default="sglang", choices=("sglang", "vllm"))
    ap.add_argument("--batch", type=int, default=32, help="batch for Q3")
    ap.add_argument("--json-out", help="also write the raw probe result here")
    args = ap.parse_args(argv)

    result = probe_all(args.dry_run, backend=args.backend, batch=args.batch)
    report_q1(result["q1"])
    report_q2(result["q2"])
    report_q3(result["q3"])
    emit("E0-DONE", f"rc={result['rc']}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str))
    return int(result["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
