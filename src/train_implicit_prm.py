"""Train the implicit PRM = an ORM whose reward is beta*log(pi_theta/pi_ref),
using the cross-entropy (CE) loss on response-level labels only.

Memory trick: pi_theta and pi_ref share one base model. pi_theta = base + LoRA
adapter (trainable); pi_ref = same base with the adapter DISABLED (frozen). So
we hold only one copy of the 1.5B weights + a tiny adapter.

CE loss (paper Eq. 5), with logit = r = beta*(sum_t logpi_theta - sum_t logpi_ref):
    L = BCEWithLogits(r, label)     # label in {0,1} = final-answer correctness

Usage:
    python src/train_implicit_prm.py --data data/train.jsonl --out outputs/prm_ce \
        --beta 0.05 --epochs 1 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_device, load_model_and_tokenizer  # noqa: E402


def response_logprob_sum(model, input_ids, attn, prompt_len, requires_grad):
    """Sum of per-token log pi(y_t|..) over the response tokens."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=attn).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt = input_ids[0, prompt_len:]
        src = logprobs[prompt_len - 1 : -1, :]
        return src.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="cap rows for smoke tests")
    args = ap.parse_args()

    device = get_device()
    base, tok = load_model_and_tokenizer(args.model, dtype=args.dtype, device=device)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.train()
    model.print_trainable_parameters()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        opt.zero_grad()
        for i, row in enumerate(tqdm(rows, desc=f"train epoch {epoch}")):
            full = tok(row["prompt"] + row["response"], return_tensors="pt",
                       truncation=True, max_length=args.max_len).to(device)
            prompt_len = tok(row["prompt"], return_tensors="pt")["input_ids"].shape[1]
            if full["input_ids"].shape[1] <= prompt_len:
                continue  # empty/truncated response

            # pi_theta (adapter ON, with grad)
            lp_theta = response_logprob_sum(model, full["input_ids"], full["attention_mask"],
                                            prompt_len, requires_grad=True)
            # pi_ref (adapter OFF, no grad)
            with model.disable_adapter():
                lp_ref = response_logprob_sum(model, full["input_ids"], full["attention_mask"],
                                              prompt_len, requires_grad=False)

            reward = args.beta * (lp_theta - lp_ref)          # scalar logit
            label = torch.tensor(float(row["label"]), device=device)
            loss = F.binary_cross_entropy_with_logits(reward, label) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    tqdm.write(f"  step {step} | avg loss {running/(i+1):.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
