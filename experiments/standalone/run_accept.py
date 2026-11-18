"""Acceptance of a wide draft truncated to a narrow verify window (the VLC cell).

For each frozen context, draft blocks of several widths from the same checkpoint, verify each
proposal with the target, and record the per-position agreement. One verify pass per block width
gives tau_k(b) = E[min(a, k)] for every k, so "draft 16 / verify 9" is compared against
"draft 9 / verify 9" on identical contexts instead of on two server configurations.

Writes rows {prompt, ctx_len, block, accepted, agree} plus timings, so cost(b,k) is measurable
rather than assumed.

Cross-instrument check: on these same frozen prompts the engine reports tau 5.86 at block 16
(RESULTS.md, deterministic mode). A harness tau_16 far from that means the instrument is wrong,
not that the hypothesis failed.

Run on the box: python run_accept.py --prompts 32 --blocks 16,12,9,8 --out accept.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.prompts import load_frozen

DRAFT_ID = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_ID = "Qwen/Qwen3-8B"


def frozen_contexts(tok, target, prompts: list[str], ar_len: int) -> list[list[int]]:
    """Greedy AR continuation per prompt, batch 1 so the tokens are a fixed shared prefix."""
    out = []
    for p in prompts:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True), return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            gen = target.generate(ids, max_new_tokens=ar_len, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        out.append(gen[0].tolist())
    return out


def target_hidden(target, ids: torch.Tensor, layer_ids: list[int], cache=None):
    """Concatenated target hidden states at the drafter's read-out layers, last position."""
    o = target(ids, past_key_values=cache, use_cache=cache is not None, output_hidden_states=True,
               logits_to_keep=1)
    th = torch.cat([o.hidden_states[i + 1] for i in layer_ids], dim=-1)
    return th, o


def draft_block(draft, embed, anchor: int, block: int, mask_id: int, th, ctx_len: int):
    """One parallel draft pass: anchor token plus (block-1) mask tokens, bidirectional."""
    blk = torch.full((1, block), mask_id, dtype=torch.long, device=th.device)
    blk[0, 0] = anchor
    with torch.no_grad():
        h = draft(target_hidden=th, noise_embedding=embed(blk),
                  position_ids=torch.arange(ctx_len, ctx_len + block,
                                            device=th.device).unsqueeze(0),
                  past_key_values=DynamicCache(), use_cache=True, is_causal=False)
    return blk, h


def verify(target, props: torch.Tensor, ctx_cache, ctx_len: int, lm=None):
    """Target forward over the proposals only; agreement of draft i is checked at column i-1."""
    with torch.no_grad():
        o = target(props, position_ids=torch.arange(ctx_len, ctx_len + props.shape[1],
                                                    device=props.device).unsqueeze(0),
                   past_key_values=ctx_cache, use_cache=False)
    post = torch.argmax(o.logits, dim=-1)
    return post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--ar-len", type=int, default=128, help="frozen AR prefix length it generates")
    ap.add_argument("--blocks", default="16,12,9,8")
    ap.add_argument("--out", default="accept.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    target = AutoModelForCausalLM.from_pretrained(TARGET_ID, dtype=torch.bfloat16).cuda().eval()
    draft = AutoModel.from_pretrained(DRAFT_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16).cuda().eval()
    lid, mask = list(draft.target_layer_ids), int(draft.mask_token_id)
    embed = target.model.embed_tokens
    print(f"CONFIG block_size={draft.block_size} mask={mask} layers={lid}")

    prompts = [it["prompt"] for it in
               load_frozen(Path(__file__).resolve().parents[2] / "data" / "frozen_math500.json")
               [:a.prompts]]
    print(f"PROMPTS {len(prompts)} from frozen_math500 (same set the engine runs)")
    ctxs = frozen_contexts(tok, target, prompts, a.ar_len)
    print(f"CONTEXTS {len(ctxs)} generated, first len={len(ctxs[0])}")

    blocks = [int(b) for b in a.blocks.split(",")]
    rows, timings = [], []
    for pi, ids in enumerate(ctxs):
        full = torch.tensor([ids], device="cuda")
        ctx_len = full.shape[1]
        cache = DynamicCache()
        th, o = target_hidden(target, full[:, :-1], lid, cache)
        anchor = int(torch.argmax(o.logits[0, -1]))

        for b in blocks:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _blk, h = draft_block(draft, embed, anchor, b, mask, th, ctx_len - 1)
            torch.cuda.synchronize(); t_draft = time.perf_counter() - t0
            with torch.no_grad():
                props = torch.argmax(target.lm_head(h[:, 1:, :]), dim=-1)  # b-1 proposals
            torch.cuda.synchronize(); t0 = time.perf_counter()
            post = verify(target, props, cache, ctx_len - 1)
            torch.cuda.synchronize(); t_verify = time.perf_counter() - t0
            agree = (props[0] == post[0]).tolist()
            acc = 0
            for okv in agree:
                if not okv:
                    break
                acc += 1
            rows.append({"prompt": pi, "ctx_len": ctx_len, "block": b, "accepted": acc,
                         "agree": agree})
            timings.append({"prompt": pi, "block": b, "draft_s": t_draft, "verify_s": t_verify,
                            "columns": b})
        if pi % 8 == 7:
            print(f"  prompt {pi + 1}/{len(ctxs)} done")

    table = {}
    for b in blocks:
        sub = [r for r in rows if r["block"] == b]
        for k in range(1, b + 1):
            table[f"b{b}_k{k}"] = statistics.fmean(min(r["accepted"], k) for r in sub)
    with open(a.out, "w") as fh:
        json.dump({"rows": rows, "tau": table, "timings": timings,
                   "config": {"block_size": draft.block_size, "layers": lid, "mask": mask,
                              "prompts": a.prompts, "ar_len": a.ar_len}}, fh, indent=1)
    print("TAU " + " ".join(f"b{b}:k9={table[f'b{b}_k9']:.2f}" for b in blocks if b >= 9))
    print(f"WROTE {a.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
