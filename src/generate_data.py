"""Generate ORM training/eval data: sample N rollouts per problem and label each
by final-answer correctness (response-level labels ONLY).

Supports two datasets:
    --dataset gsm8k   openai/gsm8k (numeric answers, "The answer is X")
    --dataset math    a Hendrycks MATH source (\\boxed{} answers); default
                      lighteval/MATH config 'all'. Override with --hf-path/--hf-config.

Output JSONL rows: {problem_id, question, prompt, response, pred, gold, label}
label = 1 if the sampled solution's final answer matches gold, else 0.

Usage:
    python src/generate_data.py --dataset gsm8k --split train --n-problems 300 --k 8 --out data/train.jsonl
    python src/generate_data.py --dataset math  --split train --n-problems 300 --k 8 --out data/math_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

# MATH subject configs in EleutherAI/hendrycks_math (used for the train pool)
_MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry",
                  "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

sys.path.insert(0, os.path.dirname(__file__))
from utils import (  # noqa: E402
    GSM8K_SYSTEM,
    MATH_SYSTEM,
    answers_match,
    build_prompt,
    extract_boxed,
    extract_pred_answer,
    gold_answer,
    load_model_and_tokenizer,
    normalize_math_answer,
)


def load_problems(args):
    """Return (list of {question, gold}, system_prompt, pred_fn)."""
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=args.split)
        ds = ds.select(range(min(args.n_problems, len(ds))))
        probs = [{"question": ex["question"], "gold": gold_answer(ex["answer"])} for ex in ds]
        return probs, GSM8K_SYSTEM, extract_pred_answer

    # MATH. Test = the canonical MATH-500 benchmark; train = a shuffled mix of
    # all 7 Hendrycks-MATH subjects. Override with --hf-path/--hf-config.
    if args.hf_path:
        ds = load_dataset(args.hf_path, args.hf_config, split=args.split)
    elif args.split == "test":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    else:
        parts = [load_dataset("EleutherAI/hendrycks_math", c, split="train")
                 for c in _MATH_SUBJECTS]
        ds = concatenate_datasets(parts).shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n_problems, len(ds))))

    def q_of(ex):
        return ex.get("problem") or ex.get("question")

    def gold_of(ex):
        if ex.get("answer"):  # MATH-500 provides the extracted answer directly
            return normalize_math_answer(str(ex["answer"]))
        return normalize_math_answer(extract_boxed(ex.get("solution", "") or ""))

    probs = [{"question": q_of(ex), "gold": gold_of(ex)} for ex in ds]

    def math_pred(text):
        b = extract_boxed(text)
        return normalize_math_answer(b) if b is not None else extract_pred_answer(text)

    return probs, MATH_SYSTEM, math_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"])
    ap.add_argument("--hf-path", default="", help="override HF dataset path for --dataset math")
    ap.add_argument("--hf-config", default=None, help="override HF dataset config for --dataset math")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-problems", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8, help="rollouts per problem")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tok = load_model_and_tokenizer(args.model, dtype=args.dtype)
    device = next(model.parameters()).device

    problems, system, pred_fn = load_problems(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written = 0
    n_correct = 0
    with open(args.out, "w") as f:
        for pid, ex in enumerate(tqdm(problems, desc=f"gen {args.dataset}/{args.split}")):
            gold = ex["gold"]
            prompt = build_prompt(tok, ex["question"], system=system)
            enc = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    num_return_sequences=args.k,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            for seq in gen:
                resp_ids = seq[prompt_len:]
                response = tok.decode(resp_ids, skip_special_tokens=True).strip()
                pred = pred_fn(response)
                label = int(answers_match(pred, gold))
                n_correct += label
                f.write(json.dumps({
                    "problem_id": pid,
                    "question": ex["question"],
                    "prompt": prompt,
                    "response": response,
                    "pred": pred,
                    "gold": gold,
                    "label": label,
                }) + "\n")
                n_written += 1

    print(f"wrote {n_written} rows to {args.out} | correct rate = {n_correct/max(n_written,1):.3f}")


if __name__ == "__main__":
    main()
