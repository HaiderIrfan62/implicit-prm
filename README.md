# Implicit PRM: free process rewards without process labels

A compact implementation of the implicit process reward model from Yuan et al.,
ICML 2025 (arXiv:2412.01981), sized to run on a laptop. It trains an outcome
reward model on final-answer labels and reads a per-step process reward straight
out of it, then uses that for best-of-N reranking on GSM8K and MATH. The reported
runs used an Apple M4 Pro (24 GB) on the PyTorch MPS backend, but it also runs on
CUDA or CPU.

## The idea

Train an outcome reward model (ORM) on cheap final-answer labels, with the reward
written as a log-likelihood ratio:

```
r_theta(y) = beta * ( log pi_theta(y|x) - log pi_ref(y|x) )
```

A process reward model (PRM) comes out of this for free. The cumulative
per-token log-ratio is the running Q-value (Prop. 3.1 in the paper), and the
difference across a step is that step's reward, so you get per-step scores with
no step labels and no MCTS. Training uses the cross-entropy loss on
response-level labels; evaluation is best-of-N reranking.

| Component | Here | Paper |
|---|---|---|
| Base model (pi_theta, pi_ref) | Qwen2.5-1.5B-Instruct | Llama-3.1-8B-Instruct |
| Fine-tuning | LoRA on pi_theta; pi_ref is the adapter disabled | full fine-tune |
| Loss | cross-entropy on response-level labels | DPO/CE/KTO/NCA |
| Tasks | GSM8K and MATH | MATH |
| Training data | 300 problems x 8 rollouts, final-answer labels only | 33k x 8 |
| Eval | best-of-N; min-step, mean-step, sequence, and weighted-vote scoring | same |

## Results

On GSM8K the implicit PRM reranks above the ORM and pass@1, but it does not beat
majority voting: self-consistency is very strong on easy math, where answers
concentrate. On MATH, where answers are diverse and self-consistency is weaker,
reward-weighted voting comes out ahead (56 vs 52 at N=8). The verifier earns its
keep where majority voting is weak.

Two smaller results. Cross-entropy trains a usable PRM even with a single response
per problem (300 examples match 2400), which preference losses like DPO cannot do
without pairs. And the reference model helps or hurts depending on the dataset:
dropping it improves GSM8K (57 to 68 at N=8) and is roughly neutral on MATH.

Accuracy (%) at N=8:

| GSM8K (pass@1 55, oracle 87) | majority | orm | prm (min) | prm (mean) | wvote (orm) | wvote (prm) |
|---|---|---|---|---|---|---|
| 8/problem, with ref | 76 | 67 | 57 | 61 | 75 | 70 |
| 8/problem, no ref   | 76 | 62 | 68 | 65 | 71 | 76 |
| 1/problem, no ref   | 76 | 62 | 66 | 63 | 70 | 74 |

| MATH (pass@1 37.8, oracle 68.4) | majority | orm | prm (min) | prm (mean) | wvote (orm) | wvote (prm) |
|---|---|---|---|---|---|---|
| with ref | 52.0 | 48.0 | 48.0 | 46.9 | 56.1 | 55.1 |
| no ref   | 52.0 | 46.9 | 42.9 | 48.0 | 54.1 | 53.1 |

A longer write-up is provided as a separate PDF with the submission.

## Setup

Python 3.11 to 3.13 (developed on 3.13.1).

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"
```

The `unauthenticated requests to the HF Hub` warning is harmless. To lift the
download rate limit, log in once with `.venv/bin/hf auth login` (a free "Read"
token from https://huggingface.co/settings/tokens).

## Usage

The examples use `PY=.venv/bin/python`.

Cached rollouts, trained adapters, and per-candidate score dumps ship with the
repo, so you can read off the results without running the model.
`analyze_scores.py` rebuilds a best-of-N table from a score dump:

```bash
$PY src/analyze_scores.py outputs/eval_math_ce.scores.json
$PY src/analyze_scores.py outputs/eval_math_ce.scores.json --temp 1.5   # weighted-vote temperature
```

### Running the whole pipeline from scratch

The steps run in order: generate, then train, then evaluate, then plot.
Generation is the slow part, so on the reference laptop GSM8K took roughly 4 to 5
hours and MATH around 16. `caffeinate -i` keeps a Mac awake. The `correct rate` printed after generation should land
near 0.5 for GSM8K and 0.3 for MATH.

GSM8K:

```bash
# generate rollouts, then build the 1-response-per-problem set
caffeinate -i $PY src/generate_data.py --dataset gsm8k --split train --n-problems 300 --k 8 --out data/train.jsonl
caffeinate -i $PY src/generate_data.py --dataset gsm8k --split test  --n-problems 100 --k 8 --out data/test.jsonl
$PY src/subsample.py data/train.jsonl data/train_1pp.jsonl --seed 0

# train the reward model with LoRA
$PY src/train_implicit_prm.py --data data/train.jsonl     --out outputs/prm_ce     --beta 0.05 --epochs 1
$PY src/train_implicit_prm.py --data data/train_1pp.jsonl --out outputs/prm_ce_1pp --beta 0.05 --epochs 1

# evaluate, with and without the reference model
$PY src/evaluate_bon.py --adapter outputs/prm_ce --data data/test.jsonl --beta 0.05 --out outputs/eval_ce.json
$PY src/evaluate_bon.py --adapter outputs/prm_ce --data data/test.jsonl --beta 0.05 --drop-ref --out outputs/eval_ce_noref.json

# plot
$PY src/plot_results.py outputs/eval_ce.json --out outputs/bon_curve.png
```

`./run_all.sh` runs the entire thing end to end (both GSM8K and MATH: generate,
train, evaluate, plot). It skips generation for any dataset whose `data/*.jsonl`
is already present, so with the shipped rollouts it goes straight to training and
evaluation. Set sizes with the `N_TRAIN` and `N_TEST` environment variables.

MATH (training data mixes the 7 Hendrycks-MATH subjects; the test set is MATH-500):

```bash
caffeinate -i $PY src/generate_data.py --dataset math --split train --n-problems 300 --k 8 --max-new-tokens 768 --out data/math_train.jsonl
caffeinate -i $PY src/generate_data.py --dataset math --split test  --n-problems 100 --k 8 --max-new-tokens 768 --out data/math_test.jsonl
$PY src/train_implicit_prm.py --data data/math_train.jsonl --out outputs/prm_math_ce --beta 0.05 --epochs 1
$PY src/evaluate_bon.py --adapter outputs/prm_math_ce --data data/math_test.jsonl --beta 0.05 --out outputs/eval_math_ce.json
$PY src/evaluate_bon.py --adapter outputs/prm_math_ce --data data/math_test.jsonl --beta 0.05 --drop-ref --out outputs/eval_math_ce_noref.json
```

To skip the slow generation, keep the shipped `data/*.jsonl` and start from the
train step. Generation samples at temperature 0.8 and MPS is not
bit-deterministic, so a from-scratch run produces new rollouts and numbers a few
points off the cached ones.

## Layout

```
src/                      code (below)
data/                     cached rollouts
  train.jsonl test.jsonl        GSM8K, 300/100 problems x 8
  train_1pp.jsonl               GSM8K, 1 response/problem
  math_train.jsonl math_test.jsonl   MATH, 300/98 problems x 8
outputs/
  prm_ce/ prm_ce_1pp/ prm_math_ce/   trained LoRA adapters
  eval_*.json                   result tables
  eval_*.scores.json            per-candidate score dumps
  *.png                         figures
run_all.sh                one-command full pipeline (GSM8K and MATH)
```

| file (`src/`) | role |
|---|---|
| `reward.py` | log-ratio reward, cumulative Q-value, per-step reward, step segmentation |
| `utils.py` | model and device loading, GSM8K and MATH answer extraction, prompts |
| `generate_data.py` | sample K rollouts/problem, label by final-answer correctness (`--dataset gsm8k` or `math`) |
| `train_implicit_prm.py` | cross-entropy training with LoRA; pi_ref is the adapter disabled |
| `evaluate_bon.py` | score every candidate, dump scores, build the best-of-N table |
| `aggregate.py` | the selection methods, including the scale-robust weighted vote |
| `analyze_scores.py` | rebuild tables from dumped scores, with a `--temp` sweep |
| `subsample.py` | build the 1-response-per-problem set |
| `relabel.py` | re-extract answers and re-label cached rollouts after a parser change |
| `plot_results.py` | best-of-N curves (`--methods`, `--title`) |
