"""RQ2: build a '1 response per problem' training set from the full rollouts.

The paper's signature CE claim is that cross-entropy still trains a useful
implicit PRM even with a single response per instruction (extreme scarcity /
class imbalance) -- the setting DPO cannot handle (no preference pairs). This
keeps one randomly chosen rollout per problem_id.

Usage:
    python src/subsample.py data/train.jsonl data/train_1pp.jsonl --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_problem = defaultdict(list)
    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            by_problem[r["problem_id"]].append(r)

    rng = random.Random(args.seed)
    kept = [rng.choice(cands) for cands in by_problem.values()]
    pos = sum(r["label"] for r in kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"{args.inp}: {len(by_problem)} problems -> {args.out}: {len(kept)} rows "
          f"(1/problem) | positives {pos} ({pos/len(kept):.1%})")


if __name__ == "__main__":
    main()
