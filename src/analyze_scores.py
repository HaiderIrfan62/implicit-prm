"""Recompute best-of-N tables from dumped per-candidate scores -- instantly, no
model. Use this to iterate on aggregation (methods, weighted-vote temperature)
after a single evaluate_bon.py run.

Usage:
    python src/analyze_scores.py outputs/eval_ce.scores.json
    python src/analyze_scores.py outputs/eval_ce.scores.json --temp 0.5 --out outputs/eval_ce.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aggregate import aggregate, format_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="path to <eval>.scores.json")
    ap.add_argument("--temp", type=float, default=1.0, help="weighted-vote softmax temperature")
    ap.add_argument("--out", default=None, help="optional: also write a summary json")
    args = ap.parse_args()

    blob = json.load(open(args.scores))
    meta, scored = blob["meta"], blob["scored"]
    Ns = meta["Ns"]

    table = aggregate(scored, Ns, temp=args.temp)
    print(f"n_problems={meta['n_problems']} K={meta['K']} beta={meta['beta']} "
          f"drop_ref={meta['drop_ref']} temp={args.temp}\n")
    print(format_table(table))

    if args.out:
        summary = {**meta, "temp": args.temp, "accuracy_by_N": table}
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
