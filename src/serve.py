"""Backend-agnostic engine adapter for the DFlash draft-width / verify-length grid.

One config per process, one process per measured cell. The adapter knows three backends
(``sglang``, ``vllm``, ``transformers``); every backend import is lazy and every import
failure is re-raised as :class:`BackendUnavailable` naming the install that is missing,
because on the laptop there is no GPU and no serving stack, and on the rented box there is
no second chance to guess which wheel was wrong.

Measurement rule (binding; it mirrors ``docs/EXPERIMENTS.md`` §4):

  * **One config per process.** ``EngineConfig`` is frozen and the process exits after its
    cell. Reusing a warm engine across arms carries allocator state, CUDA graphs captured for
    a different verify width, and a KV pool sized for another batch into the next number.
  * **GPU idle before timing.** No resident compute apps, ``nvidia-smi`` utilisation < 5%
    for the 10 s before the run. Check it, do not assume it: a Jupyter kernel holding 6 GB is
    invisible in the command you just typed.
  * **Warmup rounds discarded.** Each `generate` call in a batch is preceded by
    ``warmup_calls`` discarded calls; their timings never enter ``stats()``.
  * **CUDA-graph re-captures counted.** A re-capture is a multi-hundred-ms stall in the first
    round after the decode batch size changes. ``stats()["graph_recaptures"]`` reports it and
    ``stats()["graph_recaptures_source"]`` says whether it was observed in the server log, in
    the backend's own counters, or is only a batch-size proxy — an unobserved zero is not a
    measured zero.
  * ``ms_per_round`` is measured from this loop (wall time / speculative rounds), never
    derived from tokens/s. p50 and p95 are across requests when the backend reports
    per-request verify counts, and across calls otherwise; the field
    ``ms_per_round_source`` says which.

``dry_run=True`` needs no GPU, no weights and no network: it returns deterministic
completions derived from a hash of the prompt and a synthetic per-round timing model, so the
whole 51-cell grid can be wired up and smoke-tested on a laptop. The synthetic model is
*shaped* to the repo's published anchors (5.1x at c=1, 2.8x at c=32, tau 8.01 / 5.09) so that
plumbing errors show up as absurd ratios rather than as plausible ones. It is not evidence:
every dry-run stat carries ``"synthetic": true`` and no dry-run number may enter
``results/*.json``.

Usage:
    uv run python src/serve.py --dry-run --prompt "How many divisors does 196 have?"
    uv run python src/serve.py --dry-run --prompts-file prompts/math500.txt \\
        --draft-width 16 --verify-columns 9
    uv run python src/serve.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \\
        --verify-columns 17 --prompts-file prompts/math500.txt      # A0 on the box
    uv run python src/serve.py --backend sglang --prompts-file prompts/math500.txt  # A5 AR

Importing this module must not import sglang, vllm, torch or transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

__all__ = ["BACKENDS", "BackendUnavailable", "Engine", "EngineConfig"]

BACKENDS = ("sglang", "vllm", "transformers")

# Printed when a backend cannot be imported. Each entry names the exact install and why it
# cannot be done on the laptop, because "No module named 'sglang'" has cost us a session.
_INSTALL_HINTS: dict[str, str] = {
    "sglang": (
        "sglang is not importable. Install it on the GPU host (it pins its own torch/CUDA, so "
        "never in the laptop venv):\n"
        "  uv pip install 'sglang[all]'\n"
        "DFlash needs a build that has the DFLASH algorithm; if `--speculative-algorithm "
        "DFLASH` is rejected at startup, use the pinned build from the model card:\n"
        "  uv pip install 'git+https://github.com/sgl-project/sglang.git@refs/pull/20547/"
        "head#subdirectory=python'"
    ),
    "vllm": (
        "vllm is not importable. Install it on the GPU host:\n"
        "  uv pip install vllm\n"
        "For the dflash speculative method, the released nightly wheel is the safe build:\n"
        "  uv pip install -U vllm --torch-backend=auto "
        "--extra-index-url https://wheels.vllm.ai/nightly"
    ),
    "transformers": (
        "transformers/torch are not importable in this interpreter. Install the reference "
        "stack (the DFlash `spec_generate` path needs trust_remote_code and a recent torch):\n"
        "  uv pip install 'transformers==4.57.3' 'torch>=2.9' accelerate safetensors"
    ),
}


class BackendUnavailable(RuntimeError):
    """A serving backend's dependency is missing, with the install command attached."""


@dataclass(frozen=True)
class EngineConfig:
    """One measured cell. Frozen: a config is not mutated between runs, it is re-instantiated.

    ``draft_width`` is how many tokens the drafter emits per round (DFlash block size).
    ``verify_columns`` is how many target-forward columns carry those tokens, i.e. submitted
    tokens + the bonus column, so ``verify_columns - 1 <= draft_width``. Leaving ``drafter``
    empty serves the target autoregressively: that is the A5 speedup denominator, measured
    through the same adapter so both sides pay the same client overhead.
    """

    model: str = "Qwen/Qwen3-8B"
    drafter: str = ""
    backend: str = "sglang"
    draft_width: int = 16
    verify_columns: int = 17
    max_new_tokens: int = 256
    temperature: float = 0.0
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.draft_width < 1:
            raise ValueError(f"draft_width must be >= 1, got {self.draft_width}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        # Strict even when `drafter` is empty: an AR cell reuses the spec cell's columns so
        # only one field differs between them, and a typo there must fail here, not at 40 GB.
        if not 2 <= self.verify_columns <= self.draft_width + 1:
            raise ValueError(
                f"verify_columns={self.verify_columns} cannot be served by draft_width="
                f"{self.draft_width}: submitted tokens are verify_columns-1, which must lie in "
                f"[1, {self.draft_width}] (17 columns = 16 drafted + 1 bonus)."
            )

    @property
    def submitted(self) -> int:
        """Draft tokens actually handed to the verifier this round."""
        return self.verify_columns - 1


# --------------------------------------------------------------------------------------
# Dry-run: deterministic, GPU-free, obviously synthetic
# --------------------------------------------------------------------------------------

_VOCAB = ["therefore", "the", "divisor", "count", "follows", "from", "factoring", "into", "primes", "and", "counting", "exponent", "increments", "while", "the", "verifier", "accepts", "the", "longest", "matching", "prefix", "of", "the", "drafted", "block", "so", "round", "cost", "stays", "flat", "in", "accepted", "length", "and", "only", "the", "column", "count", "moves", "the", "target", "forward", "pass", "which", "is", "what", "the", "sweep", "is", "trying", "to", "isolate", "in", "the", "first", "place"]

# Synthetic cost model. Calibrated to the repo's anchors, NOT measured: 17 columns at c=1
# must land near 5.1x AR and at c=32 near 2.8x, otherwise the plumbing is lying to us.
_AR_STEP_MS_C1 = 8.6
_DRAFTER_MS_C1 = 1.1
_RHO = 0.50  # share of the verify pass that is column-independent
_TAU_16 = 8.01
_TAU_8 = 5.09


def _prompt_seed(prompt: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256((salt + prompt).encode()).digest()[:8], "big")


def _synth_tau(cfg: EngineConfig, rng: random.Random) -> float:
    rng.random()  # keep the stream stable across configs
    if not cfg.drafter:
        return 1.0
    full = _TAU_16 if cfg.draft_width >= 16 else _TAU_8
    cap = float(cfg.verify_columns - 1)
    return max(1.0, min(full, cap))


def _synth_round_ms(cfg: EngineConfig, batch: int, rng: random.Random) -> float:
    log_b = math.log2(max(1, batch))
    step = _AR_STEP_MS_C1 * (1.0 + 0.04 * (batch - 1))
    if not cfg.drafter:
        return step * rng.lognormvariate(0.0, 0.05)
    drafter = _DRAFTER_MS_C1 * (1.0 + 0.02 * (batch - 1))
    # Verify cost = weight-read floor + column-dependent share that grows with batch,
    # because at c=32 the pass is compute-bound on the columns it pads to.
    col = cfg.verify_columns / 17.0
    verify = step * (_RHO + (1.0 - _RHO) * col * (1.0 + 0.35 * log_b)) * 1.44
    return (drafter + verify + 0.15) * rng.lognormvariate(0.0, 0.05)


def _synth_text(prompt: str, cfg: EngineConfig) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    n = max(4, min(48, cfg.max_new_tokens // 8))
    words = [_VOCAB[int(h[i * 2 : i * 2 + 2], 16) % len(_VOCAB)] for i in range(n)]
    return f"[dry-run {h[:8]}] " + " ".join(words)


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


class _SGLangBackend:
    """In-process SGLang engine with the DFLASH speculative algorithm.

    In-process rather than HTTP: the grid needs no server lifecycle, no client retry, and no
    HTTP overhead inside a 9 ms round. ``--speculative-num-draft-tokens`` is the one knob the
    grid turns, and SGLang documents it as the DFlash verify block size.
    """

    name = "sglang"
    verify_control = "startup_flag"

    def __init__(self, cfg: EngineConfig) -> None:
        try:
            import sglang  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise BackendUnavailable(_INSTALL_HINTS["sglang"]) from exc
        kwargs: dict[str, Any] = {
            "model_path": cfg.model,
            "dtype": "bfloat16",
            "tp_size": 1,
            "mem_fraction_static": 0.75,
            "trust_remote_code": True,
        }
        if cfg.deterministic:
            # Batch-invariant kernels: without this, two identical AR runs at c=32 agree on
            # only 42% of prompts, so no exact-match gate means anything.
            kwargs["enable_deterministic_inference"] = True
        if cfg.drafter:
            kwargs.update(
                speculative_algorithm="DFLASH",
                speculative_draft_model_path=cfg.drafter,
                # SGLang counts draft tokens verified; the bonus column is not one of them.
                speculative_num_draft_tokens=cfg.submitted,
            )
        try:
            self._engine = sglang.Engine(**kwargs)
        except TypeError as exc:  # older builds: no DFLASH kwargs at all
            raise BackendUnavailable(
                f"{_INSTALL_HINTS['sglang']}\nAlso: this SGLang build rejected the DFLASH "
                f"arguments ({exc}). Check `--speculative-algorithm` choices with "
                "`python -m sglang.launch_server --help`."
            ) from exc
        self._cfg = cfg

    def generate(self, prompts: list[str]) -> tuple[list[str], list[dict]]:
        params = {"temperature": self._cfg.temperature, "max_new_tokens": self._cfg.max_new_tokens}
        outs = self._engine.generate(prompts, sampling_params=params)
        texts, metas = [], []
        for out in outs:
            # sglang's offline Engine returns plain dicts (text/output_ids/meta_info);
            # some builds return objects. Handle both rather than losing the text.
            if isinstance(out, dict):
                text = out.get("text") or ""
                meta = dict(out.get("meta_info") or {})
            else:
                text = getattr(out, "text", "") or ""
                meta = dict(getattr(out, "meta_info", {}) or {})
            rounds = meta.get("spec_verify_ct")
            accepted = meta.get("spec_accepted_tokens")
            if accepted is None and rounds and meta.get("spec_accept_length") is not None:
                # spec_accept_length is emitted tokens per round, bonus column included.
                accepted = float(meta["spec_accept_length"]) * float(rounds)
            texts.append(text)
            metas.append(
                {
                    "completion_tokens": meta.get("completion_tokens"),
                    "rounds": rounds,
                    "accepted": accepted,
                    "latency_s": meta.get("e2e_latency"),
                    "raw": meta,
                }
            )
        return texts, metas

    def close(self) -> None:
        for attr in ("shutdown", "close"):
            fn = getattr(self._engine, attr, None)
            if callable(fn):
                fn()
                return


class _VLLMBackend:
    """vLLM offline engine with ``speculative_config={method: dflash}``."""

    name = "vllm"
    verify_control = "startup_config"

    def __init__(self, cfg: EngineConfig) -> None:
        try:
            from vllm import LLM  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise BackendUnavailable(_INSTALL_HINTS["vllm"]) from exc
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.75,
            "trust_remote_code": True,
            "max_num_batched_tokens": 32768,
        }
        if cfg.drafter:
            kwargs["speculative_config"] = {
                "method": "dflash",
                "model": cfg.drafter,
                "num_speculative_tokens": cfg.submitted,
            }
        self._llm = LLM(**kwargs)
        self._cfg = cfg

    def generate(self, prompts: list[str]) -> tuple[list[str], list[dict]]:
        from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

        sp = SamplingParams(
            temperature=self._cfg.temperature, max_tokens=self._cfg.max_new_tokens
        )
        outs = self._llm.generate(prompts, sp, use_tqdm=False)
        texts, metas = [], []
        for out in outs:
            comp = out.outputs[0]
            metrics = getattr(out, "metrics", None)
            texts.append(comp.text or "")
            metas.append(
                {
                    "completion_tokens": len(comp.token_ids),
                    "rounds": _first_attr(metrics, ("num_spec_rounds", "spec_verify_ct")),
                    "accepted": _first_attr(
                        metrics, ("num_accepted_tokens", "spec_accepted_tokens")
                    ),
                    "latency_s": _first_attr(metrics, ("finished_time", "e2e_latency")),
                }
            )
        return texts, metas

    def close(self) -> None:
        del self._llm


class _TransformersBackend:
    """Reference path: the drafter's own ``spec_generate``, batch of one, no scheduler.

    This is the fallback when neither serving stack supports the per-round knob, and it is
    c=1 only by construction: ``spec_generate`` is a single-sequence method. Verify length is
    passed only if the released checkpoint's signature exposes it; otherwise
    ``verify_control`` says ``not_exposed`` and the arm is reported as untruncated rather than
    silently measured at full width.
    """

    name = "transformers"
    verify_control = "not_exposed"

    def __init__(self, cfg: EngineConfig) -> None:
        if not cfg.drafter:
            raise BackendUnavailable(
                "the transformers reference path needs --drafter: it runs the DFlash "
                "checkpoint's spec_generate against the target, it is not a plain AR engine. "
                "Use --backend sglang or --backend vllm for the A5 AR denominator."
            )
        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModel,
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise BackendUnavailable(_INSTALL_HINTS["transformers"]) from exc
        # Otherwise a laptop run silently starts pulling ~16 GB of weights before failing.
        vram = (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        )
        if vram < 20 * 2**30:
            raise BackendUnavailable(
                "the transformers reference path holds the bf16 8B target plus the drafter and "
                f"needs >= 20 GiB of VRAM; this device reports {vram / 2**30:.1f} GiB. Run it on "
                "the rented A100, and use --dry-run on the laptop."
            )
        self._torch: Any = torch
        self._draft: Any = AutoModel.from_pretrained(
            cfg.drafter, trust_remote_code=True, dtype="auto", device_map="cuda:0"
        ).eval()
        self._target: Any = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype="auto", device_map="cuda:0"
        ).eval()
        self._tok = AutoTokenizer.from_pretrained(cfg.model)
        self._cfg = cfg
        sig = _signature_names(getattr(self._draft, "spec_generate", None))
        self._len_kwarg = next(
            (n for n in ("num_draft_tokens", "verify_length", "spec_len", "block_size") if n in sig),
            None,
        )
        self.verify_control = "applied" if self._len_kwarg else "not_exposed"

    def generate(self, prompts: list[str]) -> tuple[list[str], list[dict]]:
        if len(prompts) != 1:
            raise ValueError(
                "the transformers reference path is c=1 only: spec_generate takes one sequence. "
                f"Got {len(prompts)} prompts; use sglang or vllm for c>1."
            )
        text = self._tok.apply_chat_template(
            [{"role": "user", "content": prompts[0]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self._tok([text], return_tensors="pt").to(self._draft.device)
        kwargs: dict[str, Any] = {
            "input_ids": ids["input_ids"],
            "max_new_tokens": self._cfg.max_new_tokens,
            "temperature": self._cfg.temperature,
            "target": self._target,
            "stop_token_ids": [self._tok.eos_token_id],
        }
        if self._len_kwarg:
            kwargs[self._len_kwarg] = self._cfg.submitted
        out = self._draft.spec_generate(**kwargs)
        new_ids = out[0][ids["input_ids"].shape[1] :]
        return [self._tok.decode(new_ids, skip_special_tokens=True)], [
            {"completion_tokens": int(new_ids.numel())}
        ]

    def close(self) -> None:
        self._draft = None
        self._target = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _signature_names(fn: Any) -> set[str]:
    if fn is None:
        return set()
    import inspect

    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


_BACKEND_CLASSES = {
    "sglang": _SGLangBackend,
    "vllm": _VLLMBackend,
    "transformers": _TransformersBackend,
}


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


class Engine:
    """Timing shell around a backend. All bookkeeping lives here, so stats are comparable.

    ``warmup_calls`` discarded batches per ``generate`` call: the first round after a batch
    size change pays graph capture, allocator growth and cache-cold weights, and averaging
    that into p50 is how a 5.1x becomes a 4.2x.
    """

    def __init__(self, cfg: EngineConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = bool(dry_run)
        self.warmup_calls = 1
        self._backend: Any = None
        self._closed = False
        self._samples: list[float] = []
        self._rounds = 0
        self._accepted = 0.0
        self._tokens = 0
        self._calls = 0
        self._prompts = 0
        self._wall = 0.0
        self._batch_sizes: set[int] = set()
        self._warmed: set[int] = set()
        self._recaptures = 0
        self._recapture_source = "none"
        self._round_source = "unknown"
        self._notes: list[str] = []
        self._log_path = os.environ.get("DFLASH_SERVE_LOG", "")
        if not self.dry_run:
            self._backend = _BACKEND_CLASSES[cfg.backend](cfg)
            self._round_source = "backend_meta" if cfg.drafter else "completion_tokens"
            if self._log_path:
                self._recapture_source = "log"
        else:
            self._round_source = "synthetic"
            self._recapture_source = "synthetic_batch_sizes"

    # -- public ----------------------------------------------------------------

    def generate(self, prompts: list[str]) -> list[str]:
        if self._closed:
            raise RuntimeError("engine is closed; construct a new Engine for the next cell")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("generate() takes a non-empty list of prompts")
        for p in prompts:
            if not isinstance(p, str) or not p.strip():
                raise ValueError("every prompt must be a non-empty string")

        if self.dry_run:
            texts, metas, dt = self._dry_generate(prompts)
        else:
            self._warmup(prompts)
            t0 = time.perf_counter()
            texts, metas = self._backend.generate(prompts)
            dt = time.perf_counter() - t0
            if len(texts) != len(prompts):
                raise RuntimeError(f"backend returned {len(texts)} texts for {len(prompts)} prompts")

        self._calls += 1
        self._prompts += len(prompts)
        self._wall += dt
        self._account(len(prompts), dt, metas)
        return texts

    def reset_stats(self) -> None:
        """Drop accumulated counters so the next cell's stats describe only that cell.

        One engine serves every concurrency for an arm (the verify length is a startup
        flag, concurrency is not), so without this the c=8 record would include c=1 and
        c=4 in its tau and tok/s. Warmup state is kept on purpose.
        """
        self._samples = []
        self._rounds = 0
        self._accepted = 0.0
        self._tokens = 0
        self._calls = 0
        self._prompts = 0
        self._wall = 0.0
        self._batch_sizes = set()

    def stats(self) -> dict:
        if self._log_path and not self.dry_run:
            observed = _read_log_recaptures(self._log_path)
            if observed is not None:
                self._recaptures = observed
                self._recapture_source = "log"
        p50 = p95 = 0.0
        if self._samples:
            ordered = sorted(self._samples)
            p50 = statistics.median(ordered)
            p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        tokens = self._tokens
        return {
            "dry_run": self.dry_run,
            "synthetic": self.dry_run,
            "backend": self.cfg.backend,
            "model": self.cfg.model,
            "drafter": self.cfg.drafter,
            "draft_width": self.cfg.draft_width,
            "verify_columns": self.cfg.verify_columns,
            "submitted_tokens": self.cfg.submitted if self.cfg.drafter else 0,
            "temperature": self.cfg.temperature,
            "calls": self._calls,
            "prompts": self._prompts,
            "tokens_generated": tokens,
            "rounds": self._rounds,
            "accepted": round(self._accepted, 3),
            "tau_mean": round((self._accepted + self._rounds) / self._rounds, 4)
            if self._rounds
            else 0.0,
            "ms_per_round_p50": round(p50, 4),
            "ms_per_round_p95": round(p95, 4),
            "ms_per_round_samples": len(self._samples),
            "ms_per_round_source": self._round_source,
            "tok_per_s": round(tokens / self._wall, 2) if self._wall > 0 else 0.0,
            "wall_s": round(self._wall, 4),
            "graph_recaptures": self._recaptures,
            "graph_recaptures_source": self._recapture_source,
            "verify_control": getattr(self._backend, "verify_control", "startup_flag")
            if not self.dry_run
            else "synthetic",
            "notes": list(self._notes),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------------

    def _warmup(self, prompts: list[str]) -> None:
        """One discarded batch per distinct batch size, so the first timed batch is not paying
        graph capture, allocator growth and cold weights. Repeated for the same batch size it
        would just double the cell's GPU time and change nothing."""
        if self.warmup_calls <= 0 or len(prompts) in self._warmed:
            return
        self._warmed.add(len(prompts))
        for _ in range(self.warmup_calls):
            try:
                self._backend.generate(prompts)
            except Exception as exc:  # noqa: BLE001 - warmup is best-effort and never fatal
                self._notes.append(f"warmup call failed and was skipped: {exc!r}")
                return

    def _dry_generate(self, prompts: list[str]) -> tuple[list[str], list[dict], float]:
        texts = [_synth_text(p, self.cfg) for p in prompts]
        metas: list[dict] = []
        total_ms = 0.0
        for p in prompts:
            rng = random.Random(_prompt_seed(p, f"{self.cfg.draft_width}/{self.cfg.verify_columns}"))
            tau = _synth_tau(self.cfg, rng)
            rounds = max(1, math.ceil(self.cfg.max_new_tokens / tau))
            per_round = _synth_round_ms(self.cfg, len(prompts), rng)
            accepted = 0.0
            for _ in range(rounds):
                self._samples.append(per_round * rng.lognormvariate(0.0, 0.06))
                if self.cfg.drafter:
                    accepted += max(0.0, min(tau, float(self.cfg.verify_columns)) - 1.0)
            # One batched decode: every request advances one round per scheduler step, so
            # the batch costs max(rounds) * per-round time, not the sum over requests.
            total_ms = max(total_ms, per_round * rounds)
            metas.append(
                {
                    "completion_tokens": self.cfg.max_new_tokens,
                    "rounds": rounds,
                    "accepted": accepted if self.cfg.drafter else None,
                }
            )
        self._note_batch(len(prompts))
        return texts, metas, total_ms / 1000.0

    def _account(self, batch: int, dt: float, metas: list[dict]) -> None:
        tokens = sum(int(m.get("completion_tokens") or 0) for m in metas)
        rounds = sum(int(m.get("rounds") or 0) for m in metas)
        accepted = sum(float(m.get("accepted") or 0.0) for m in metas)
        self._note_batch(batch)

        if rounds <= 0:
            # No round counter: estimate from tokens and nominal acceptance, and say so.
            tau_nom = (
                max(1.0, float(min(self.cfg.verify_columns, self.cfg.draft_width + 1)))
                if self.cfg.drafter
                else 1.0
            )
            rounds = max(1, math.ceil(tokens / tau_nom)) if tokens else 0
            accepted = max(0.0, float(tokens) - rounds) if self.cfg.drafter else 0.0
            self._round_source = "estimated_from_tokens"

        self._tokens += tokens
        self._rounds += rounds
        self._accepted += accepted

        if rounds > 0 and not self.dry_run:
            per_req = [
                float(m["latency_s"]) * 1000.0 / float(m["rounds"])
                for m in metas
                if m.get("latency_s") and m.get("rounds")
            ]
            if per_req:
                self._samples.extend(per_req)
                self._round_source = "per_request"
            else:
                self._samples.append(dt * 1000.0 / rounds)
                if self._round_source == "backend_meta":
                    self._round_source = "per_call"

    def _note_batch(self, batch: int) -> None:
        if batch in self._batch_sizes:
            return
        if self._batch_sizes:
            self._recaptures += 1  # a new decode batch size costs one graph capture
        self._batch_sizes.add(batch)


def _read_log_recaptures(path: str) -> int | None:
    """Count CUDA-graph capture lines in a captured server log, if one was pointed at."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    return sum(1 for line in text.splitlines() if "capture" in line.lower() and "graph" in line.lower())


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "How many positive whole-number divisors does 196 have?",
    "Solve for x: 3x + 7 = 2x - 5.",
]


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text().splitlines()
        prompts.extend(ln.strip() for ln in lines if ln.strip())
    return prompts or list(_DEFAULT_PROMPTS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve one DFlash config and report round stats.")
    ap.add_argument("--dry-run", action="store_true", help="no GPU, no weights, synthetic stats")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--drafter", default="", help="HF id of the DFlash drafter; empty = AR")
    ap.add_argument("--backend", default="sglang", choices=BACKENDS)
    ap.add_argument("--draft-width", type=int, default=16)
    ap.add_argument("--verify-columns", type=int, default=17)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", action="append", help="repeatable")
    ap.add_argument("--prompts-file", help="one prompt per line")
    ap.add_argument("--json-out", help="also write {'completions','stats'} here")
    ap.add_argument("--show-completions", action="store_true")
    args = ap.parse_args(argv)

    cfg = EngineConfig(
        model=args.model,
        drafter=args.drafter,
        backend=args.backend,
        draft_width=args.draft_width,
        verify_columns=args.verify_columns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    prompts = _load_prompts(args)
    try:
        with Engine(cfg, dry_run=args.dry_run) as eng:
            texts = eng.generate(prompts)
            stats = eng.stats()
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.show_completions or args.dry_run:
        for i, t in enumerate(texts):
            print(f"[{i}] {t}")
    print(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"config": asdict(cfg), "completions": texts, "stats": stats}, indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
