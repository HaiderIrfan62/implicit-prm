"""Best-of-N evaluation of the implicit PRM against baselines on a test set.

Scores every candidate once, DUMPS the per-candidate scores to
<out>.scores.json, then aggregates into a best-of-N table (see aggregate.py for
the methods). Because the scores are dumped, all aggregation variants can be
recomputed instantly with analyze_scores.py -- no need to re-run the model.

Usage:
    python src/evaluate_bon.py --adapter outputs/prm_ce --data data/test.jsonl \
        --beta 0.05 --out outputs/eval_ce.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from aggregate import aggregate, format_table  # noqa: E402
from reward import implicit_prm_scores, step_boundaries  # noqa: E402
from utils import get_device, load_model_and_tokenizer  # noqa: E402


@torch.no_grad()
def per_token_logratio(model, tok, prompt, response, device, max_len, drop_ref):
    full = tok(prompt + response, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    prompt_len = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    ids, attn = full["input_ids"], full["attention_mask"]
    if ids.shape[1] <= prompt_len:
        return None, None

    def resp_logprobs():
        logits = model(input_ids=ids, attention_mask=attn).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = ids[0, prompt_len:]
        return lp[prompt_len - 1 : -1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    lp_theta = resp_logprobs()
    if drop_ref:
        logratio = lp_theta
    else:
        with model.disable_adapter():
            lp_ref = resp_logprobs()
        logratio = lp_theta - lp_ref
    resp_ids = ids[0, prompt_len:]
    return logratio, resp_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--delimiter", default="\n\n")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--drop-ref", action="store_true", help="drop pi_ref at inference (Sec 5.5)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = get_device()
    base, tok = load_model_and_tokenizer(args.model, dtype=args.dtype, device=device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # group candidates by problem
    by_problem = defaultdict(list)
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            by_problem[r["problem_id"]].append(r)

    # score every candidate
    scored = {}
    for pid, cands in tqdm(by_problem.items(), desc="scoring"):
        rows = []
        for c in cands:
            logratio, resp_ids = per_token_logratio(
                model, tok, c["prompt"], c["response"], device, args.max_len, args.drop_ref)
            if logratio is None:
                reward, min_step, n_steps = -1e9, -1e9, 0
            else:
                bounds = step_boundaries(resp_ids, tok, args.delimiter)
                s = implicit_prm_scores(logratio, bounds, args.beta)
                reward, min_step, n_steps = s.reward, s.min_step_reward, s.n_steps
            rows.append({
                "label": c["label"], "pred": c["pred"], "gold": c["gold"],
                "reward": reward, "min_step": min_step, "n_steps": n_steps,
            })
        scored[pid] = rows

    K = min(len(v) for v in scored.values())
    Ns = [n for n in [1, 2, 4, 8, 16, 32] if n <= K]

    # dump per-candidate scores so aggregations can be recomputed offline
    meta = {"n_problems": len(scored), "K": K, "beta": args.beta,
            "drop_ref": args.drop_ref, "adapter": args.adapter, "Ns": Ns}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scores_path = args.out.replace(".json", "") + ".scores.json"
    with open(scores_path, "w") as f:
        json.dump({"meta": meta, "scored": scored}, f)

    table = aggregate(scored, Ns)
    summary = {**meta, "accuracy_by_N": table}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(format_table(table))
    print(f"\nsaved table -> {args.out}\nsaved scores -> {scores_path}")


if __name__ == "__main__":
    main()
