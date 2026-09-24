#!/usr/bin/env bash
# Full pipeline: GSM8K and MATH, generate -> train -> evaluate -> plot.
#
# Generation steps are guarded by [ -f ... ], so if the cached data/*.jsonl is
# present (as shipped) they are skipped and the run goes straight to training
# and evaluation. Delete a data file to regenerate it (slow: see the README).
# On macOS, wrap the whole run to prevent sleep:  caffeinate -i ./run_all.sh
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
BETA=${BETA:-0.05}
N_TRAIN=${N_TRAIN:-300}       # problems for training rollouts
N_TEST=${N_TEST:-100}         # problems for evaluation
K=${K:-8}                     # rollouts per problem
MATH_TOKENS=${MATH_TOKENS:-768}

########################################  GSM8K  ########################################
echo ">>> GSM8K: generate rollouts"
[ -f data/train.jsonl ] || $PY src/generate_data.py --model "$MODEL" --dataset gsm8k \
  --split train --n-problems "$N_TRAIN" --k "$K" --out data/train.jsonl
[ -f data/test.jsonl ] || $PY src/generate_data.py --model "$MODEL" --dataset gsm8k \
  --split test --n-problems "$N_TEST" --k "$K" --out data/test.jsonl
[ -f data/train_1pp.jsonl ] || $PY src/subsample.py data/train.jsonl data/train_1pp.jsonl --seed 0

echo ">>> GSM8K: train (8 responses/problem, and 1 response/problem)"
$PY src/train_implicit_prm.py --model "$MODEL" --data data/train.jsonl     --out outputs/prm_ce     --beta "$BETA" --epochs 1
$PY src/train_implicit_prm.py --model "$MODEL" --data data/train_1pp.jsonl --out outputs/prm_ce_1pp --beta "$BETA" --epochs 1

echo ">>> GSM8K: evaluate best-of-N (with and without the reference model)"
$PY src/evaluate_bon.py --model "$MODEL" --adapter outputs/prm_ce     --data data/test.jsonl --beta "$BETA"            --out outputs/eval_ce.json
$PY src/evaluate_bon.py --model "$MODEL" --adapter outputs/prm_ce     --data data/test.jsonl --beta "$BETA" --drop-ref --out outputs/eval_ce_noref.json
$PY src/evaluate_bon.py --model "$MODEL" --adapter outputs/prm_ce_1pp --data data/test.jsonl --beta "$BETA"            --out outputs/eval_ce_1pp.json
$PY src/evaluate_bon.py --model "$MODEL" --adapter outputs/prm_ce_1pp --data data/test.jsonl --beta "$BETA" --drop-ref --out outputs/eval_ce_1pp_noref.json

########################################  MATH  ########################################
echo ">>> MATH: generate rollouts"
[ -f data/math_train.jsonl ] || $PY src/generate_data.py --model "$MODEL" --dataset math \
  --split train --n-problems "$N_TRAIN" --k "$K" --max-new-tokens "$MATH_TOKENS" --out data/math_train.jsonl
[ -f data/math_test.jsonl ] || $PY src/generate_data.py --model "$MODEL" --dataset math \
  --split test --n-problems "$N_TEST" --k "$K" --max-new-tokens "$MATH_TOKENS" --out data/math_test.jsonl

echo ">>> MATH: train"
$PY src/train_implicit_prm.py --model "$MODEL" --data data/math_train.jsonl --out outputs/prm_math_ce --beta "$BETA" --epochs 1

echo ">>> MATH: evaluate best-of-N (with and without the reference model)"
$PY src/evaluate_bon.py --model "$MODEL" --adapter outputs/prm_math_ce --data data/math_test.jsonl --beta "$BETA"            --out outputs/eval_math_ce.json
$PY src/evaluate_bon.py --model "$MODEL" --adapter outputs/prm_math_ce --data data/math_test.jsonl --beta "$BETA" --drop-ref --out outputs/eval_math_ce_noref.json

########################################  plots  ########################################
echo ">>> plots"
$PY src/plot_results.py outputs/eval_ce.json --out outputs/bon_curve.png
$PY src/plot_results.py outputs/eval_ce.json outputs/eval_ce_noref.json \
  --methods pass@1 majority prm_bon_min wmaj_prm oracle \
  --title "Best-of-N on GSM8K: reference vs no-reference" --out outputs/bon_ref_vs_noref.png
$PY src/plot_results.py outputs/eval_ce_noref.json outputs/eval_ce_1pp_noref.json \
  --methods pass@1 majority prm_bon_min wmaj_prm oracle \
  --title "Best-of-N on GSM8K: 8 vs 1 response per problem" --out outputs/bon_8pp_vs_1pp.png
$PY src/plot_results.py outputs/eval_math_ce.json \
  --methods pass@1 majority orm_bon prm_bon_min wmaj_orm wmaj_prm oracle \
  --title "Best-of-N on MATH: implicit reward vs baselines" --out outputs/bon_math.png

echo "DONE. Results in outputs/eval_*.json; figures in outputs/*.png"
