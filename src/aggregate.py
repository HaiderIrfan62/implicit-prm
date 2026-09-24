"""Selection methods for best-of-N, computed from per-candidate scores.

Kept separate from scoring so we can iterate on aggregation instantly (via
analyze_scores.py) without re-running the model.

Each candidate score record has:
    label   : 1/0 final-answer correctness
    pred     : extracted answer string
    gold     : gold answer string
    reward   : sequence-level implicit reward = sum of scaled per-step log-ratios
               (this IS the ORM reward; also = sum of step rewards)
    min_step : min per-step reward   (the paper's weakest-link PRM score)
    n_steps  : number of steps       (mean_step = reward / n_steps)

Methods:
    pass@1        first candidate, no selection
    oracle        any of the N correct (upper bound)
    majority      plain self-consistency (most common answer)
    orm_bon       argmax sequence reward
    prm_bon_min   argmax min-step reward           <- paper's implicit PRM
    prm_bon_mean  argmax mean-step reward          (length-normalized)
    wmaj_orm      weighted self-consistency, weight from sequence reward
    wmaj_prm      weighted self-consistency, weight from min-step reward

Weighted voting uses a *scale-robust* weight: within each problem's N
candidates we z-score the chosen reward, then softmax. This is invariant to the
reward's shift and scale, so it behaves sensibly whether or not pi_ref is
dropped (whose reward magnitudes differ by orders of magnitude). Summing raw
(negative) rewards -- the earlier bug -- made more votes hurt; this fixes it.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List

METHODS = ["pass@1", "oracle", "majority", "orm_bon",
           "prm_bon_min", "prm_bon_mean", "wmaj_orm", "wmaj_prm"]


def _mean_step(r):
    n = r.get("n_steps") or 0
    return r["reward"] / n if n else r["reward"]


def _softmax_weights(scores: List[float], temp: float = 1.0) -> List[float]:
    """z-score then softmax -> shift/scale-invariant vote weights that sum to 1."""
    if len(scores) == 1:
        return [1.0]
    m = statistics.mean(scores)
    sd = statistics.pstdev(scores) or 1.0
    z = [(s - m) / sd / temp for s in scores]
    mx = max(z)
    ex = [math.exp(v - mx) for v in z]
    tot = sum(ex) or 1.0
    return [e / tot for e in ex]


def _weighted_vote(sub, key, gold, temp=1.0):
    valid = [r for r in sub if r["pred"] is not None]
    if not valid:
        return 0
    weights = _softmax_weights([r[key] for r in valid], temp)
    tally = defaultdict(float)
    for r, w in zip(valid, weights):
        tally[r["pred"]] += w
    return int(max(tally, key=tally.get) == gold)


def aggregate(scored: Dict[str, List[dict]], Ns: List[int], temp: float = 1.0):
    """scored: {problem_id: [candidate score records]}. Returns {N: {method: acc}}."""
    results = {n: defaultdict(float) for n in Ns}
    n_problems = len(scored)
    for rows in scored.values():
        gold = rows[0]["gold"]
        for N in Ns:
            sub = rows[:N]
            res = results[N]
            res["pass@1"] += sub[0]["label"]
            res["oracle"] += int(any(r["label"] for r in sub))
            votes = Counter(r["pred"] for r in sub if r["pred"] is not None)
            if votes:
                res["majority"] += int(votes.most_common(1)[0][0] == gold)
            res["orm_bon"] += max(sub, key=lambda r: r["reward"])["label"]
            res["prm_bon_min"] += max(sub, key=lambda r: r["min_step"])["label"]
            res["prm_bon_mean"] += max(sub, key=_mean_step)["label"]
            res["wmaj_orm"] += _weighted_vote(sub, "reward", gold, temp)
            res["wmaj_prm"] += _weighted_vote(sub, "min_step", gold, temp)
    return {N: {k: round(v / n_problems, 4) for k, v in results[N].items()} for N in Ns}


def format_table(table) -> str:
    Ns = sorted(table, key=int)
    w = 14
    hdr = "N".ljust(5) + "".join(m.ljust(w) for m in METHODS)
    lines = [hdr, "-" * len(hdr)]
    for N in Ns:
        row = str(N).ljust(5) + "".join(
            f"{table[N].get(m, 0)*100:.1f}%".ljust(w) for m in METHODS)
        lines.append(row)
    return "\n".join(lines)
