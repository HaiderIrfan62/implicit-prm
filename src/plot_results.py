"""Plot best-of-N accuracy curves from one or more evaluate_bon.py outputs.

Usage:
    python src/plot_results.py outputs/eval_ce.json --out outputs/bon_curve.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt

METHODS = {
    "pass@1": ("pass@1", "--", "grey"),
    "majority": ("majority vote (SC)", "-", "tab:orange"),
    "orm_bon": ("ORM best-of-N", "-", "tab:red"),
    "prm_bon_min": ("implicit PRM best-of-N (min)", "-", "tab:blue"),
    "prm_bon_mean": ("implicit PRM best-of-N (mean)", "-", "tab:cyan"),
    "wmaj_prm": ("PRM weighted vote", "-", "tab:green"),
    "oracle": ("oracle@N", ":", "black"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="outputs/bon_curve.png")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of method keys to plot (default: all known)")
    ap.add_argument("--title", default="Best-of-N — implicit PRM vs baselines")
    args = ap.parse_args()
    methods = {k: v for k, v in METHODS.items()
               if args.methods is None or k in args.methods}

    fig, ax = plt.subplots(figsize=(7, 5))
    for path in args.files:
        with open(path) as f:
            data = json.load(f)
        table = data["accuracy_by_N"]
        Ns = sorted(int(n) for n in table)
        tag = "" if len(args.files) == 1 else f" [{path.split('/')[-1].replace('.json','')}]"
        for key, (label, ls, color) in methods.items():
            ys = [table[str(n)].get(key) for n in Ns]
            if any(y is None for y in ys):
                continue
            ax.plot(Ns, [y * 100 for y in ys], ls, color=color, marker="o",
                    label=label + tag, alpha=0.9)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (candidates in best-of-N)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
