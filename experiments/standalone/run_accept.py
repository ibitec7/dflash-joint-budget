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


def encode(tok, prompt: str, mode: str) -> torch.Tensor:
    """Tokenise a prompt the way a given backend would.

    The sglang backend hands raw prompt strings to Engine.generate, so `raw` reproduces what the
    engine saw; `chat` applies the tokenizer template with a generation prompt, which is what the
    single-engine backend does. The two conventions put the model in different regimes (Qwen3's
    template opens a thinking block), and a drafter trained without them accepts differently, so
    the convention is a variable, not a detail.
    """
    text = prompt if mode == "raw" else tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.cuda()


def frozen_contexts(tok, target, prompts: list[str], ar_len: int, mode: str) -> list[list[int]]:
    """Greedy AR continuation per prompt, batch 1 so the tokens are a fixed shared prefix."""
    out = []
    for p in prompts:
        ids = encode(tok, p, mode)
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
    """One parallel draft pass: anchor token plus (block-1) mask tokens, bidirectional.

    position_ids must span the whole cache row range (ctx_len + block), not just the block:
    the rotary table is indexed by the cached K rows and length is enforced by broadcast only,
    so a block-length slice either raises or - worse - runs silently wrong.
    """
    blk = torch.full((1, block), mask_id, dtype=torch.long, device=th.device)
    blk[0, 0] = anchor
    with torch.no_grad():
        h = draft(target_hidden=th, noise_embedding=embed(blk),
                  position_ids=torch.arange(ctx_len + block, device=th.device).unsqueeze(0),
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


def reference_loop(draft, target, embed, lid, mask, blk: int, ids: torch.Tensor, max_new: int):
    """Greedy speculative loop, written to mirror the checkpoint's own spec_generate.

    Returns the committed token ids after the prefix and the per-round committed counts. This
    exists to be diffed against spec_generate: if the two disagree, the instrument is wrong and
    no acceptance number from it may be reported.
    """
    ctx = ids.shape[1]
    out = torch.full((1, ctx + max_new + blk), mask, dtype=torch.long, device=ids.device)
    pos = torch.arange(out.shape[1], device=ids.device).unsqueeze(0)
    pkv_t, pkv_d = DynamicCache(), DynamicCache()
    o = target(ids, position_ids=pos[:, :ctx], past_key_values=pkv_t, use_cache=True,
               logits_to_keep=1, output_hidden_states=True)
    out[:, :ctx] = ids
    out[:, ctx] = torch.argmax(o.logits[0, -1])
    th = torch.cat([o.hidden_states[i + 1] for i in lid], dim=-1)
    start, rounds, end = ctx, [], ctx + max_new
    while start < end:
        block = out[:, start:start + blk].clone()
        h = draft(target_hidden=th, noise_embedding=embed(block),
                  position_ids=pos[:, pkv_d.get_seq_length(): start + blk],
                  past_key_values=pkv_d, use_cache=True, is_causal=False)
        prop = torch.argmax(target.lm_head(h[:, -blk + 1:, :]), dim=-1)
        pkv_d.crop(start)
        block[:, 1:] = prop
        vo = target(block, position_ids=pos[:, start:start + blk], past_key_values=pkv_t,
                    use_cache=True, output_hidden_states=True)
        post = torch.argmax(vo.logits, dim=-1)
        acc = int((block[:, 1:] == post[:, :-1]).flatten().cumprod(0).sum())
        out[:, start:start + acc + 1] = block[:, :acc + 1]
        out[:, start + acc + 1] = post[0, acc]
        rounds.append(acc + 1)
        start += acc + 1
        pkv_t.crop(start)
        th = torch.cat([vo.hidden_states[i + 1] for i in lid], dim=-1)[:, :acc + 1, :]
    # A round commits up to `blk` tokens, so the loop can overshoot `end` by up to blk-1. The
    # checkpoint's spec_generate stops exactly at the budget (measured: ctx+max_new, not one
    # more), so clamp the sequence and subtract the overshoot from the final round's count.
    stop = min(start, end)
    if start > end and rounds:
        rounds[-1] -= start - end
    return out[:, :stop].clone(), rounds


def check_reference(draft, target, embed, lid, mask, tok, prompts, max_new: int, blk: int,
                    mode: str = "raw", engine_tau: float = 0.0):
    """Diff my loop against the checkpoint's own spec_generate; no tolerance, token for token.

    Two ways this can pass while measuring nothing, both now refused:
    - a mismatched token stream (REFERENCE_MISMATCH), and
    - a reference that accepts no drafts at all (REFERENCE_DEGENERATE): when tau is 1 every round,
      both loops are plain greedy decoding and agree whatever the drafter does. A gate that only
      compares tokens therefore validates an instrument that cannot accept.
    """
    taus = []
    for i, p in enumerate(prompts):
        ids = encode(tok, p, mode)
        with torch.no_grad():
            ref = draft.spec_generate(target=target, input_ids=ids, max_new_tokens=max_new,
                                      stop_token_ids=None, temperature=0.0)
        mine, rounds = reference_loop(draft, target, embed, lid, mask, blk, ids, max_new)
        same = ref.shape == mine.shape and bool((ref == mine).all())
        print(f"[REF {i}] identical={same} ref={tuple(ref.shape)} mine={tuple(mine.shape)} "
              f"tau={sum(rounds) / len(rounds):.2f} rounds={len(rounds)}")
        if not same:
            print(f"  REF  {ref[0, ids.shape[1]:].tolist()[:16]}")
            print(f"  MINE {mine[0, ids.shape[1]:].tolist()[:16]}")
            raise SystemExit("REFERENCE_MISMATCH: instrument not validated, refusing to measure")
        taus.append(sum(rounds) / len(rounds))
    mean_tau = sum(taus) / len(taus)
    floor = 0.6 * engine_tau if engine_tau else 1.5
    print(f"REFERENCE_TAU mean={mean_tau:.2f} per prompt={[round(t, 2) for t in taus]} "
          f"floor={floor:.2f}" + (f" engine={engine_tau:.2f}" if engine_tau else ""))
    if mean_tau < floor:
        print("  the reference accepts too few drafts per round: at that level both loops are plain "
              "greedy decoding and agree whether or not the drafter works, so agreement proves nothing")
        raise SystemExit("REFERENCE_DEGENERATE: prompt convention or wiring is wrong, "
                         "refusing to measure")
    print(f"REFERENCE_OK mode={mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="math500")
    ap.add_argument("--prompt-mode", default="raw", choices=("raw", "chat"))
    ap.add_argument("--check-new", type=int, default=256,
                    help="tokens the reference check decodes; shorter windows understate tau")
    ap.add_argument("--engine-tau", type=float, default=0.0,
                    help="tau the engine reports for this benchmark, for the cross-instrument gate")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--ar-len", type=int, default=128, help="frozen AR prefix length it generates")
    ap.add_argument("--blocks", default="16,12,9,8")
    ap.add_argument("--check-ref", type=int, default=2,
                    help="prompts to diff against spec_generate before measuring; 0 disables")
    ap.add_argument("--max-new", type=int, default=48)
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
               load_frozen(Path(__file__).resolve().parents[2] / "data" / f"frozen_{a.bench}.json")
               [:a.prompts]]
    print(f"PROMPTS {len(prompts)} from frozen_{a.bench} (same set the engine runs)")
    ctxs = frozen_contexts(tok, target, prompts, a.ar_len, a.prompt_mode)
    print(f"CONTEXTS {len(ctxs)} generated, first len={len(ctxs[0])}")

    if a.check_ref:
        check_reference(draft, target, embed, lid, mask, tok, prompts[:a.check_ref],
                        a.check_new, int(draft.block_size), mode=a.prompt_mode,
                        engine_tau=a.engine_tau)

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
            matched = 0
            for okv in agree:
                if not okv:
                    break
                matched += 1
            # committed tokens per round, the same quantity the engine reports as
            # spec_accept_length: matched drafts plus the target's own correction token.
            rows.append({"prompt": pi, "ctx_len": ctx_len, "block": b,
                         "accepted": matched + 1, "drafts_matched": matched, "agree": agree})
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
