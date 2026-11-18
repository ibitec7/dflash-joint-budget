# Serving runbook — one A100, one config per process

`src/serve.py` is the only entry point and `EngineConfig` the only description of a cell. Every
cell is a fresh process: CUDA graphs, KV pool and allocator state are sized for the arm you
launched, and a warm engine reused across arms leaks one arm's shapes into the next. Target
`Qwen/Qwen3-8B` bf16, drafter `z-lab/Qwen3-8B-DFlash-b16` (or its b8 sibling), greedy, 256 tokens.

## Rules that make a number a measurement
- `nvidia-smi` before every cell: no compute apps, <5% util for 10 s. A Jupyter kernel holding
  6 GB does not appear in the command you just typed.
- Warmup batches are discarded (once per batch size) and never enter `stats()`, which reports
  `tau_mean`, `ms_per_round_p50/p95`, `tok_per_s` from this loop, and `graph_recaptures` with
  its `graph_recaptures_source`: an unobserved zero says so.
- Spec output must match the AR run token-for-token on >=99% of prompts (H6). A spec cell
  without its AR cell on the same prompts is not a comparison.

## 1. Install (on the box; never in the laptop venv)
```bash
uv venv .venv-gpu && source .venv-gpu/bin/activate && uv pip install -U pip 'sglang[all]'
# pin only if DFLASH is unknown at startup:
# uv pip install 'git+https://github.com/sgl-project/sglang.git@refs/pull/20547/head#subdirectory=python'
huggingface-cli login          # Qwen3-8B is gated; the drafter is not
huggingface-cli download Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16 && pip freeze \
  | grep -E 'sglang|vllm|torch|transformers'   # into the session log
```
5.1x is a property of a build, not of the paper: pinned build + A100 80 GB is the reference.

## 2. A0 anchor: 5.1x at c=1 within 10% (same adapter both sides, client overhead cancels)
```bash
uv run python src/serve.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --draft-width 16 --verify-columns 17 --prompts-file prompts/math500.txt --json-out results/a0.json
uv run python src/serve.py --backend sglang \
  --draft-width 16 --verify-columns 17 --prompts-file prompts/math500.txt --json-out results/a5.json
uv run python -c "import json;a=json.load(open('results/a0.json'))['stats']['tok_per_s'];b=json.load(open('results/a5.json'))['stats']['tok_per_s'];print(f'{a/b:.2f}x pass>=4.59')"
```
The same config as server flags, for an interactive check with identical semantics:
```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 \
  --speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-num-draft-tokens 16 --mem-fraction-static 0.75 --trust-remote-code
```
Don't copy the card's `--attention-backend fa3` (Hopper+): on A100 leave the default and record
it, because acceptance differs by attention backend. If A0 < 4.59x, R5 fired: our A0 replaces the
paper's as the baseline and every ratio is recomputed against ours, not against 5.1x.

## 3. Verify length: one value per process, one knob
`verify_columns` counts submitted tokens **plus the bonus column**, so
`--speculative-num-draft-tokens = verify_columns - 1` (17 -> 16, 9 -> 8). Draft width is the
checkpoint's block size, not a flag you can lower: A3 serves the b8 checkpoint.

| arm | width | columns | flags |
|---|---|---|---|
| A0 | 16 | 17 | `--draft-width 16 --verify-columns 17` |
| A1 | 16 | 9 | `--draft-width 16 --verify-columns 9` |
| A2 | 16 | 7/9/11/13/17 | one process per column count |
| A3 | 8 | 9 | `--draft-width 8 --verify-columns 9`, b8 drafter |
| A5 | — | — | omit `--drafter`: AR denominator |

**Per request** is what A4 needs and what no released API promises. Probe it on day 1:
```bash
curl -s localhost:30000/generate -H 'content-type: application/json' -d '{"text":"2+2=",
"sampling_params":{"max_new_tokens":8,"temperature":0,"speculative_num_draft_tokens":8}}' | head -c 600
```
- `spec_verify_ct` falls for that request -> per-request control exists; pass it per call in
  `_SGLangBackend.generate` and A4 runs as designed.
- Unchanged or rejected -> A4 cannot run on SGLang: patch `SamplingParams` and the scheduler to
  read the field, or run A4 on Transformers at c=1 and label it `c=1 only`. Whichever you do,
  `stats()["verify_control"]` (`startup_flag` / `applied` / `not_exposed`) reports whether the
  length was actually applied, so an arm whose L never varied cannot pass as adaptive.

## 4. Fallback ladder — what each step loses
| backend | c>1 | per-request L | what is lost |
|---|---|---|---|
| sglang, DFLASH | yes | only if the probe passes | nothing: the reference |
| vllm, `speculative_config={"method":"dflash"}` | yes | no | DFlash block path and overlap scheduler are SGLang-tuned; `num_speculative_tokens` may be offset by one from SGLang's block size, so calibrate at startup and record it |
| transformers, `--backend transformers` | **no, c=1 only** | only if `spec_generate` exposes it | every c>1 cell (H1/H3/H4 at c=32), batched ms/round, throughput above batch 1; A4 untestable |

Order is fixed: sglang -> vllm -> transformers, and each drop is reported, not hidden. vllm keeps the grid minus per-round adaptivity; transformers keeps only the c=1 anchors (A0's 5.1x, A5). R4 firing: the report ships C1-C3 plus a c=1 column, grid NOT MEASURED. Never mix backends in one table.

## 5. DFlash install on a Brev A100
`scripts/box_bootstrap.sh` does §1 for you, idempotently, as whatever user the instance gave you
(`ubuntu`, `shadeform`, …) — it reads `whoami`/`$HOME`, refuses to run as root, and builds a 3.12
venv because the 22.04 image's `python3` is 3.10 and `from enum import StrEnum` fails at import.
```bash
ssh <instance>                                  # then, in a NON-login shell:
git clone <repo> ~/dflash-joint-budget && cd ~/dflash-joint-budget
bash scripts/box_bootstrap.sh                   # prints only MARK-STEP / MARK-DONE lines
bash scripts/box_bootstrap.sh --dry-run         # same markers, mutates nothing
source ~/dflash_env.sh                          # REPO, PY, PYTHONPATH, HF_HOME, CUDA_HOME
```
Read it as markers, not logs: `MARK-STEP <name> ok|FAIL <detail>` and a final `MARK-DONE rc=<n>`;
rc 0 means every step passed, 2 means it refused to run. Command output is in
`~/dflash_bootstrap.log`. The torch wheel comes from the `nvcc --version` on the box
(12.6 -> `https://download.pytorch.org/whl/cu126`; `/whl/cu12` is not a valid index, it 404s), and
`CUDA_HOME`/`PATH` are exported in the script *and* in `dflash_env.sh`, because non-login remote
shells never source `/etc/profile.d`. The script never downloads weights and never installs
sglang: it prints the exact `uv pip install --python "$PY" 'sglang[all]'` line, and runs it only
with `--with-sglang`, since that wheel pins its own torch/CUDA and is a separate decision.

Then pull the weights deliberately and run E0 before any grid cell:
```bash
$PY -m huggingface_hub.cli download Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16   # Qwen3-8B is gated
$PY scripts/e0_probe.py --backend sglang --json-out results/e0.json
```
E0 answers three questions with evidence and prints `E0-Q1:`/`E0-Q2:`/`E0-Q3:` lines: does the
drafter serve tokens at all, is the verify length per-request or only per-process (it introspects
`SamplingParams`/`Engine.generate` and greps the installed package, never guesses a field name),
and what ms(17 cols)/ms(9 cols) costs at batch 32. `E0-Q3: NOT MEASURABLE` means A4 is dead on
this backend — see §3. Verify the plumbing on the laptop first with
`uv run python scripts/e0_probe.py --dry-run`; every dry-run line is labelled `dry-run`.
