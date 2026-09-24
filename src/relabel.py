"""Re-extract pred + re-compute label on already-generated rollouts, using the
current (fixed) extract_pred_answer. Rewrites the jsonl in place. This avoids
re-generating rollouts when only the answer parser changed.

Usage:
    python src/relabel.py data/train.jsonl data/test.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import answers_match, extract_pred_answer  # noqa: E402


def relabel(path: str):
    rows = [json.loads(l) for l in open(path)]
    old_correct = sum(r.get("label", 0) for r in rows)
    for r in rows:
        pred = extract_pred_answer(r["response"])
        r["pred"] = pred
        r["label"] = int(answers_match(pred, r["gold"]))
    new_correct = sum(r["label"] for r in rows)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    n = len(rows)
    print(f"{path}: {n} rows | correct {old_correct} ({old_correct/n:.1%}) "
          f"-> {new_correct} ({new_correct/n:.1%})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        relabel(p)
