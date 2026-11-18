# RUNBOOK — fresh clone to a measured verdict

Brev A100 80 GB, `ubuntu@<ip>`, driver 565.57.01, CUDA 12.6, repo `~/dflash-joint-budget`, venv `~/.venv312`. Run everything **on the box**, from `~/dflash-joint-budget`, after `source ~/dflash_env.sh` (sets `$PY`, `$REPO`, `PYTHONPATH`, `HF_HOME`, `CUDA_HOME`). Only value you supply: `BOX_HOST` in §4.

Three rules; breaking one voids the run:
1. `results/` receives **only** `scripts/run_grid.py` records. E0 JSON → `logs/`, the E1 probe → `results_e1/`: a stray `e0.json` in `results/` makes `decide.py` die on a missing-key `ValueError` before any verdict prints (verified).
2. A `*_dryrun` directory is synthetic by construction (`SYNTHETIC_DATA_DO_NOT_USE` marker) and `decide.py` refuses it, exit 3. Never copy a record into one, or out of one.
3. One process per cell, `nvidia-smi` quiet before every timed step. A warm engine or a leftover Jupyter kernel changes the number you are about to publish.

## S0 — Preflight (no GPU time until every line prints what it should)
```bash
cd ~/dflash-joint-budget && source ~/dflash_env.sh && mkdir -p logs
"$PY" -V 2>&1 | sed 's/^/P1-PY /'                        # expect Python 3.12.x
git rev-parse HEAD | sed 's/^/P2-HEAD /'                  # the commit the numbers belong to
git status --porcelain | head -3 | sed 's/^/P3-DIRTY /'   # expect no output: clean tree
pgrep -af 'sglang|vllm|python' | grep -v pgrep && echo "P4-PROCS OCCUPIED" || echo "P4-PROCS none"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader  # expect empty
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 1 -c 5   # <5 %, <1 GiB
"$HOME/.venv312/bin/hf" auth whoami 2>&1 | sed 's/^/P5-TOKEN /'          # expect user=...
"$PY" -c "import sglang,torch;print('P6-SGLANG',sglang.__version__,'cuda',torch.cuda.is_available())"
"$HOME/.venv312/bin/hf" download Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16 2>&1 | tail -3
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B                     # ~16 GB
```
`$PY -m huggingface_hub.cli` is not runnable (package, no `__main__`); the `hf` entry point above is what the venv has. Fail: `P4-PROCS OCCUPIED` or any compute app → kill that PID, re-check; util ≥5% → idle 60 s and retake the 5 samples, never start a cell on a busy card; `P5-TOKEN` `Not logged in` → `"$HOME/.venv312/bin/hf" auth login` with a token that accepted the Qwen3-8B gate (a 401/403 download is the gate, not the network); `P6-SGLANG` ImportError → `~/.local/bin/uv pip install --python "$PY" 'sglang[all]'`, re-source the env file, and if startup then rejects `--speculative-algorithm DFLASH` install the pinned build from `src/serve.py`'s hint, logging that build id to `logs/session.md` (5.1× is a property of a build); Python ≠3.12 → env file not sourced, else `rm -rf ~/.venv312 && bash scripts/box_bootstrap.sh`.

## S1 — E0 feasibility gate (three questions, one stop rule)
```bash
cd ~/dflash-joint-budget && source ~/dflash_env.sh
"$PY" scripts/e0_probe.py --backend sglang --json-out logs/e0.json 2>&1 | tee logs/e0.log | grep -E '^E0-(Q1|Q2|Q3|DONE)'; echo "E0-RC=${PIPESTATUS[0]}"
```
Q1: does the drafter emit tokens at all (loads the transformers `spec_generate` path, needs ≥20 GiB)? Q2: is verify length per-request or only a startup flag? Q3: ms(17)/ms(9) at batch 32 — `NOT MEASURABLE` unless Q2 is YES.

- **Stop rule, Q1 = NO:** `E0-Q1: NO` with `E0-DONE rc=3`. Do not run E1 or E2. Note it in `logs/gaps.txt`, pull `logs/`, stop the instance: the report ships C1–C3 plus the cost model, empirical section NOT MEASURED.
- Q2 = NO is **expected** on SGLang: the length is the startup flag `--speculative-num-draft-tokens` (alias `--speculative-dflash-block-size`), never per-request, so A0 (17 columns) and A1 (9 columns) are two engine configurations — exactly what E1/E2 launch. No action.
- Q3 `NOT MEASURABLE` (also expected, follows from Q2): A4 is dead on this backend, H7 is **N/A, not failed**. Log and continue; `run_grid.py` refuses A4 with exit 1 and a message naming E0, so no A4 cell can appear by accident.

## S2 — E1 shape probe (R1 gate: is verification column-sensitive at all?)
```bash
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --arms A0,A1,A5 --benchmarks math500 --concurrency 32 --prompts 32 --out results_e1
"$PY" - <<'PY'
import json
a = json.load(open('results_e1/A0_math500_c32_s0.json')); b = json.load(open('results_e1/A1_math500_c32_s0.json'))
r = b['ms_per_round_p50'] / a['ms_per_round_p50']
print(f"E1-RATIO {r:.3f} gate<=0.85 {'PASS' if r <= 0.85 else 'FAIL'} | tau A0={a['tau']:.2f} A1={b['tau']:.2f} | match A1={b['exact_match_ar']:.3f}")
PY
```
Marker `E1-RATIO ... PASS`. `FAIL` (ratio ≥0.85, in practice ≈1.0) → R1 fired: the target pads to 17 columns regardless, so the lever does not exist here — stop, report the negative result, the grid would buy nothing. `match A1 < 0.99` → H6 already broken by truncation; fix before spending grid time. No `E1-RATIO` line → no record written: read the `tok/s` lines the grid printed; a `SYNTHETIC_DATA_DO_NOT_USE` file in `results_e1/` means `--dry-run` slipped in (delete the directory, do not edit it).

32 prompts: a gate, not data. These records stay in `results_e1/` and must not be merged into `results/`, where `decide.py` would average them into the same cell as the 200-prompt grid.

## S3 — E2 main grid
A3 (width 8) needs a b8 drafter, so gate it first:
```bash
curl -s -o /dev/null -w 'A3-DRAFTER http=%{http_code}\n' https://huggingface.co/api/models/z-lab/Qwen3-8B-DFlash-b8
```
`http=200` → run 3.2. `http=401`/`404` → **no b8 DFlash checkpoint exists for Qwen3-8B** (the only 8B repo is `-b16`): A3 unmeasurable, H4 MISSING, and `decide.py` returns INCOMPLETE however good the other cells are. Serving b16 with 9 columns is A1, not A3 — never relabel it. Log `GAP-A3`, skip 3.2/3.4, read §4 on INCOMPLETE.
```bash
# 3.1 main arms + AR denominator, both benchmarks, seed 0 (A5 records come from this run)
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 --arms A0,A1,A5 --benchmarks math500,mtbench --concurrency 1,4,8,16,32 --seeds 0 --out results
# 3.2 A3, only if the gate printed http=200
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b8 --arms A3 --benchmarks math500,mtbench --concurrency 1,4,8,16,32 --seeds 0 --out results
# 3.3 three seeds on the headline cells (new filenames, so no --force needed)
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 --arms A0,A1,A5 --benchmarks math500 --concurrency 32 --seeds 1,2 --out results
# 3.4 A3 seeds, only with A3 present
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b8 --arms A3 --benchmarks math500 --concurrency 32 --seeds 1,2 --out results
```
Markers: one line per cell, `c=32  A1 tau=7.xx tok/s=xxx.x match=0.99x`. Count: `ls results/*.json | wc -l` → **36** without A3 (30 from 3.1 + 6 from 3.3), **48** with it (3.2's 10 + 3.4's 2).

Fail: `<path> exists; rerun with --force` → that cell is measured already; change `--seeds`, never `--force` a published cell. CUDA OOM at c=32 → something else holds memory, re-check S0, re-run that cell alone. `BackendUnavailable` → S0 P6. `match < 0.99` on any A1/A3 cell → H6 fail: stop, a completed run would be INVALID.

## S4 — Pull the results back and take the verdict
```bash
# on the box
"$PY" scripts/decide.py --results results --benchmark math500 | tee logs/decide.log; echo "DECIDE-RC=${PIPESTATUS[0]}"
```
```bash
# on the laptop — fill in BOX_HOST (the only placeholder in this runbook)
BOX_HOST=${BOX_HOST:-ubuntu@FILL-IN-PUBLIC-IP}
mkdir -p ~/uni/dflash-joint-budget/results_brev ~/uni/dflash-joint-budget/logs_brev
rsync -avz "$BOX_HOST":~/dflash-joint-budget/results/ ~/uni/dflash-joint-budget/results_brev/
rsync -avz "$BOX_HOST":~/dflash-joint-budget/logs/    ~/uni/dflash-joint-budget/logs_brev/
cd ~/uni/dflash-joint-budget && uv run python scripts/decide.py --results results_brev --benchmark math500; echo "rc=$?"
uv run python scripts/analyze.py --results results_brev --out figures/gain_vs_concurrency.png
```
Markers: a `verdict:` line (`POSITIVE`, `POSITIVE, WEAKER CLAIM`, `MECHANISM ONLY`, `NEGATIVE RESULT`, `INVALID RUN`, `INCOMPLETE`), one `H1`…`H7` `PASS|FAIL|MISSING` line each, then `rho:` and `w_tail:` model-vs-measurement lines. Exit code first: 0 verdict, 1 INCOMPLETE, 2 INVALID RUN, 3 refused a synthetic directory.

`INCOMPLETE` with `H4 MISSING` is the expected shape when §S3 blocked A3: report the cell as missing, never as a failed hypothesis, and never move a threshold to make it disappear. `INVALID RUN` (H6 <0.99) → withdraw the results section, report the bug. Exit 3 → `--results` pointed at a `*_dryrun` directory.

## S5 — Teardown, and what to leave running
```bash
pgrep -af 'sglang|vllm' || echo "TEARDOWN-NOPROCS"          # expect no server, no notebook
nvidia-smi --query-gpu=memory.used --format=csv,noheader     # expect <1 GiB after the last cell
ls -l ~/dflash-joint-budget/results/*.json | wc -l           # matches the S3 count
```
Nothing has to keep running: every cell is a batch process that exits, so there is no server to leave up and no daemon to babysit. Pull §S4 first, then keep `results/`, `results_e1/`, `logs/`, `~/dflash_env.sh`, `~/dflash_bootstrap.log`. The `~/.cache/huggingface` weights (~18 GB) are disposable, so delete nothing if the instance is reused this session. Stop the instance from the Brev console — an idle A100 still bills $1.98/h.

## Cost at $1.98/h
| step | wall clock | cost |
|---|---|---|
| S0 preflight + 18 GB weight pull | 0.4 h | $0.79 |
| S1 E0 | 0.4 h | $0.79 |
| S2 E1 (32 prompts, c=32) | 0.4 h | $0.79 |
| S3 E2 grid + seeds | 1.8 h | $3.56 |
| S4 pull-back + verdict | 0.2 h | $0.40 |
| contingency (one failed session) | 1.5 h | $2.97 |
| **total** | **4.7 h** | **$9.31** |

~2 min per cell at c ≤16, with a c=32 cell at 256 new tokens the long pole. `docs/PLAN.md`'s 10 h envelope includes a second session; this sequence is one.

## Gaps this runbook does not paper over
- **A4 is unimplementable as coded:** `run_grid.py` refuses it by selector whatever E0 reports, and the adapter fixes the length at `sglang.Engine(...)` construction. Enabling it is a change to `EngineConfig`/`src/serve.py`, not an operator step.
- **A2's sweep is unreachable** from `run_grid.py`: `ARMS["A2"].verify_columns` is `None`, which falls back to 17, so an A2 run silently reproduces A0 (verified). The sweep needs one process per column count via `src/serve.py --verify-columns`, whose JSON is not a `decide.py` record — keep it outside `results/`.
- **A3 needs a checkpoint the Hub does not publish for this target** (§S3 gate). `e0_probe.py --backend` accepts only `sglang|vllm`; Q1 always runs the transformers path internally. Any flag not shown above is unverified here: check `--help` before using it.
