# Results

We re-measure DFlash (arXiv 2602.06036, ICML 2026) at serving batch on one A100, reproduce its own block-width ablation to within 7%, and specify the single unmeasured cell that decides whether verify-window truncation is a knob separate from draft-block width. Every number carries its source; the one cell that does not exist yet is marked `TO BE MEASURED` with the key it will come from.

## 1. Setup

| field | value |
|---|---|
| GPU | 1x A100 80GB PCIe, driver 580.178.04, CUDA 13.0 |
| target / drafter | `Qwen/Qwen3-8B` / `z-lab/Qwen3-8B-DFlash-b16` |
| backend | SGLang 0.5.23, in-process `Engine`, `speculative_algorithm=DFLASH` |
| decoding | greedy (T=0), `max_new_tokens=256` |
| benchmark | frozen MATH-500, first 32 prompts; replication at 64 |
| concurrency | c in {1, 4, 8, 16, 32}; SGLang clamps `max_running_requests` to 48 |
| arms | A0 = draft 16 / 17 columns; A1, A3 = draft 8 / 9 columns; A5 = plain AR |
| kernel modes | default; `enable_deterministic_inference` (batch-invariant) |

Sources: all rows from RESULTS.md (Configuration; Correction 2026-11-19); prompt count from `data/frozen_math500.json: n` (200 frozen prompts, seed 0).

Prompts are frozen before the first GPU run and read by every arm through `scripts/prompts.py::load_frozen`, so no arm can be handed a different input. The acceptance instrument regenerates its prefixes by greedy AR at batch 1 (`experiments/standalone/run_accept.py:36`), so the wide and narrow arms see byte-identical context and identical target hidden states.

Both kernel modes appear because they answer different questions and cannot be mixed. Default kernels are the production path but not reproducible: two identical AR runs agree on 27/64 prompts (0.422), so a default-kernel quality number measures backend noise (RESULTS.md F6). Batch-invariant kernels give 1.000 agreement on 64/64 and make losslessness checkable, at 2.5x the absolute throughput cost and 7% of the paired ratio (1.318 deterministic vs 1.409 default, RESULTS.md F11).

## 2. The premise: the published gain does not survive serving batch

DFlash reports 2.8x at c=32 on a B200 (ABLATION-PROTOCOL.md §1, T3 p.7). Here, at c=32, the published block-16 configuration is at parity with plain autoregressive decoding:

| pass | AR tok/s | A0 tok/s | A0 / AR | A0 ms/round | source |
|---|---|---|---|---|---|
| c=1, default | 88.91 | 301.44 | **3.39x** | 15.41 | (results/A0_math500_c1_s0.json: tok_per_s) |
| c=32, default, s0 | 2451.1 | 2245.11 | **0.916x** | 50.74 | (results/A0_math500_c32_s0.json: tok_per_s) |
| c=32, default, s1 | 2460.8 | 2234.5 | **0.908x** | — | (RESULTS.md F12) |
| c=32, default, s2 | 2461.1 | 2461.8 | **1.002x** | — | (RESULTS.md F12) |
| c=32, deterministic | 993.58 | 1674.29 | **1.685x** | 73.04 | (results_det/A0_math500_c32_s0.json: tok_per_s) |

The collapse is not a drafter problem. A0's round time grows 3.29x from c=1 to c=32 (15.41 to 50.74 ms) while the 9-column arm's grows 2.01x (15.05 to 30.31 ms): the verify pass filling the batch, not slower drafting (results/A0_math500_c1_s0.json, results/A1_math500_c32_s0.json: ms_per_round_p50).

The reportable quantity is the paired ratio, not the absolute. Across two default-kernel seeds the absolutes move ~10% (A0 2234.5 to 2461.8 tok/s; A1 3149.0 to 3467.9) while A1/A0 stays width-shaped: 1.498 (s0), 1.409 (s1), 1.409 (s2), 1.364 at 64 prompts, 1.318 deterministic, so quote the range 1.32-1.50 rather than the pair that agrees (RESULTS.md F12). AR is not a stable denominator either: A0 lands on both sides of parity with it (0.908x s1, 1.002x s2). So 1.685x is quotable only with "deterministic kernels" attached, and 0.908x only with "default kernels, seed 1".

## 3. What we can already claim: the block-width effect at c=32

Halving the block at inference time — the remedy DFlash nominates in §5.5.4 — recovers most of the loss at serving batch on this hardware:

| arm (draft / verify columns) | tau | tok/s | vs AR | vs A0 | source |
|---|---|---|---|---|---|
| A0 (16 / 17) | 5.861 | 1674.29 | 1.685x | 1.000 | (results_det/A0_math500_c32_s0.json) |
| A1 (8 / 9) | 5.001 | 2206.70 | 2.221x | **1.318x** | (results_det/A1_math500_c32_s0.json) |
| A3 (8 / 9) | 5.001 | 2228.34 | 2.243x | 1.331x | (results_det/A3_math500_c32_s0.json) |

A1 and A3 are one server configuration measured twice (RESULTS.md, Correction 2026-11-19), so the 1.331x and 1.318x rows are a repeat 1.0% apart, not a second finding. The width effect is tau 5.861 to 5.001, a 14.7% drop, paid in acceptance and repaid in round time (73.04 to 41.37 ms at c=32; ms_per_round_p50 in the two deterministic records above).

This is DFlash's Table 8 diagonal run at batch 32 instead of single stream, and it lands on the paper's operating point rather than on a misconfigured arm:

| cell | paper tau (H200, batch 1) | our tau (A100, deterministic, c=1) | delta |
|---|---|---|---|
| b16 drafter at b16 (= A0) | 6.33 | 5.861 | -7.4% |
| b16 drafter at b8 (= A1/A3) | 5.09 | 5.001 | -1.8% |

Sources: paper cells from ABLATION-PROTOCOL.md §1 (T8, p.8); our cells from results_det/A0_math500_c1_s0.json and results_det/A1_math500_c1_s0.json (tau); deltas from RESULTS.md F10. Different GPU class, concurrency and kernel mode, so this is not a bit-level reproduction and is not quoted as one. It shows the paper's width effect (-19.6%) is visible here (-14.7%), and that is the effect the verify-window cell has to beat.

None of this licenses a statement about a verify window: no arm ever drafted wider than it verified. Under the pre-registered rules the verdict is MECHANISM ONLY — H1 0.566 PASS, H3 1.318 PASS, H6 1.0000 PASS, H2 tau 5.00 against a 7.2 bar FAIL, H4 0.990 FAIL and re-scored INVALID-BY-CONFOUND, H5 0.798 at c=1 FAIL, H7 N/A (no per-request verify length) (RESULTS.md, Verdict; EXPERIMENTS.md §2).

## 4. The cell that decides the proposal

Design: **VLC(16, 9)** — draft a 16-token block, verify only the first 9 columns — against **D8** — the same checkpoint at inference block 8, verifying all 9 — at c=32, 64 frozen prompts, one kernel mode, one backend.

Only drafter width differs. Verify columns are 9 in both arms; the tau ceiling is 9 in both arms, since `tau_k(b) = E[min(a, k)]` and k is shared; prompts and AR prefixes are byte-identical; the checkpoint is shared, so width is not confounded with capacity (ABLATION-PROTOCOL.md §3.4 i). It is not a relabelling of the paper's b16→b8 row because block attention is bidirectional: position *i* of a 16-block sees 16 positions, position *i* of an 8-block sees 8, so the two arms put **different tokens in the same 9 slots** (EXPERIMENTS.md §2b).

The cell cannot be expressed on SGLang 0.5.23: `speculative_num_draft_tokens` sets the drafter's block and the verified length at once, and different values are a hard `ValueError` (RESULTS.md F5). It is measured instead by `experiments/standalone/run_accept.py`, gated on reproducing the checkpoint's own `spec_generate` token for token; until that gate prints `REFERENCE_OK`, no acceptance number counts.

| quantity | where it will be read from |
|---|---|
| instrument gate | `TO BE MEASURED` — `REFERENCE_OK` from `run_accept.py --check-ref` |
| `tau_9(16)` | `TO BE MEASURED` — `accept.json` `tau["b16_k9"]` |
| `tau_9(8)` | `TO BE MEASURED` — `accept.json` rows at `block=8`, `mean(min(accepted, 9))` |
| tau ratio, paired on (prompt, ctx_len) + 95% CI | `TO BE MEASURED` — `vlc.py` output [2] |
| per-prompt wins / ties / losses | `TO BE MEASURED` — `vlc.py` output [2] |
| cost factor `round(8,9)/round(16,9)` | `TO BE MEASURED` — `vlc.py` output [3] |
| predicted tok/s ratio | `TO BE MEASURED` — `vlc.py` output [3] |
| verdict | `TO BE MEASURED` — `vlc.py` output [4] |

Nothing here is estimated or carried over from the block-k arms; those sit inside 5% of each other and that is a warning about the prior, not a substitute for the measurement (NOVELTY.md §3).

Bars fixed before the run: PASS needs `tau_9(16) >= 1.05 * tau_9(8)` and `tok/s(16,9) >= 1.10 * tok/s(8,9)` at c=32 (EXPERIMENTS.md §2b; ABLATION-PROTOCOL.md §3.4). **Pre-committed collapse condition:** a tok/s ratio in [0.95, 1.05] collapses the contribution to a replication of the published observation that shortening verification helps at batch (AdaFlash, LibraSpec) plus a negative result — on a bidirectional-block drafter the block-size remedy DFlash nominates and the truncation remedy proposed here buy the same thing. The report says that in those words and the title loses "better knob" (NOVELTY.md §3).

## 5. Threats to validity

1. **The arm labels were wrong, and the correction is part of the record.** `src/serve.py` passed only `speculative_num_draft_tokens = verify_columns - 1`, so every arm launched with a block equal to its submitted length; A1's declared "draft 16 / verify 9" executed as draft 8 / verify 9 (RESULTS.md, Correction 2026-11-19). Every pre-correction sentence attributing an effect to a verify window is retracted, and A1-vs-A3 is a 1.0% noise floor, not a durability result.
2. **Kernel mode is not portable.** Absolute throughput moves 2.48x by design, A0's position against AR flips sign (1.685x deterministic, 0.908x default), and tau differs for one configuration (5.861 deterministic, 5.510 default) because batch-invariant kernels change argmax ties (RESULTS.md F11). Mixing modes in one table is forbidden; the instrument times a transformers loop, not the engine.
3. **Seed-to-seed movement.** tau is not stable across seeds for one configuration (A0: 5.861 deterministic, 5.510 s1, 5.730 s2) and absolutes move ~10% with it (RESULTS.md F12). A 5% tau effect is within what these seeds do.
4. **One benchmark, not two.** Every record in `results/` and `results_det/` is MATH-500. A frozen MT-Bench set exists (`data/frozen_mtbench.json`, 200 prompts from an 80-prompt pool, repeated 3x) but no arm ran on it, so nothing here generalises to open-ended chat.
5. **Prompt count.** 32 prompts at most concurrencies, 64 at c=32, against a 200-prompt pool (`data/frozen_math500.json: n`). Our 5.5–5.7 tau at 17 columns sits against the paper's 8.01 (RESULTS.md F7), a prompt-subset, output-length and GPU gap; no cross-paper tau comparison is made.
6. **Single GPU class.** One A100 80GB PCIe against the paper's B200 (T3) and H200 (T8). The collapse and the recovery are properties of a compute-bound pass, so their magnitudes are hardware-dependent even where the sign is not.
7. **The treatment cell needs an instrument whose equivalence gate could fail.** `run_accept.py` raises `SystemExit("REFERENCE_MISMATCH")` on any token divergence from `spec_generate`, and times draft and verify in a standalone transformers loop rather than inside SGLang. If the gate fails the cell is unmeasured, not negative; if it passes, the cost terms still carry a runtime change relative to the engine ratios in §2 and §3 (ABLATION-PROTOCOL.md §3.4).

## 6. Reproduction

On the box, from `~/dflash-joint-budget`, after `source ~/dflash_env.sh`:

```bash
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --arms A0,A1,A5 --benchmarks math500,mtbench --concurrency 1,4,8,16,32 --seeds 0 --out results
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --arms A3 --benchmarks math500,mtbench --concurrency 1,4,8,16,32 --seeds 0 --out results
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --arms A0,A1,A5 --benchmarks math500 --concurrency 32 --seeds 1,2 --out results
"$PY" scripts/run_grid.py --backend sglang --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --arms A0,A1,A5 --deterministic --benchmarks math500 --concurrency 1,4,8,16,32 --seeds 0 --out results_det
"$PY" scripts/decide.py --results results_det --benchmark math500 --adaptive-na
```

The acceptance instrument runs standalone on the GPU box, then analysis runs anywhere:

```bash
python experiments/standalone/run_accept.py --prompts 32 --blocks 16,12,9,8 --out accept.json
uv run python scripts/vlc.py experiments/standalone/accept.json
uv run python scripts/vlc.py --selftest
```

`run_accept.py` prints `REFERENCE_OK` before measuring and aborts with `REFERENCE_MISMATCH` otherwise; `vlc.py` prints `absent` for any cell the file cannot supply and `no verdict` rather than interpolating past a block's drafted width.
