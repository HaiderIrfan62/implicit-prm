"""
Core of the Implicit PRM method (Yuan et al., ICML 2025, arXiv:2412.01981).

The outcome reward is parameterized as a log-likelihood ratio:

    r_theta(y) = beta * ( log pi_theta(y | x) - log pi_ref(y | x) )

Because probabilities multiply over tokens and log turns that into a sum, the
*cumulative* log-ratio over the first t tokens is exactly the running Q-value
(Prop. 3.1). The per-step process reward is then the difference of cumulative
Q-values across a step (an advantage) -- obtained with NO step labels.

This module contains the math only; training/eval live elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@torch.no_grad()
def token_logprobs(model, input_ids, attention_mask, prompt_len: int):
    """Per-token log pi(y_t | x, y_<t) for the RESPONSE tokens of one sequence.

    input_ids / attention_mask: shape (1, L), the full prompt+response.
    prompt_len: number of prompt tokens; response occupies [prompt_len:L].
    Returns a 1-D tensor of length (L - prompt_len).
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0].float()                      # (L, V)
    logprobs = torch.log_softmax(logits, dim=-1)        # (L, V)
    # token at position i is predicted by logits at position i-1
    tgt = input_ids[0, prompt_len:]                     # (R,)
    src = logprobs[prompt_len - 1 : -1, :]              # (R, V)
    return src.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (R,)


def token_logprobs_grad(model, input_ids, attention_mask, prompt_len: int):
    """Same as token_logprobs but keeps the graph (for training pi_theta)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    tgt = input_ids[0, prompt_len:]
    src = logprobs[prompt_len - 1 : -1, :]
    return src.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


@dataclass
class ImplicitPRMScores:
    """Everything the implicit PRM produces for a single response."""
    reward: float                 # sequence-level ORM reward r_theta(y)
    q_values: List[float]         # cumulative Q after each STEP (length = n_steps)
    step_rewards: List[float]     # per-step process reward = advantage (length = n_steps)
    min_step_reward: float        # weakest-link score used for best-of-N
    n_steps: int


def step_boundaries(response_token_ids, tokenizer, delimiter: str = "\n\n") -> List[int]:
    """Return response-relative token indices where each step ENDS (exclusive).

    Steps are delimited by `delimiter` in the decoded text. We decode
    incrementally and cut whenever a new delimiter boundary is crossed. The
    final index always closes the last step.
    """
    ids = response_token_ids.tolist()
    boundaries: List[int] = []
    seen = ""
    n_delims_emitted = 0
    for i in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:i], skip_special_tokens=True)
        # count completed delimiter occurrences in the text so far
        n_delims = text.count(delimiter)
        if n_delims > n_delims_emitted:
            boundaries.append(i)
            n_delims_emitted = n_delims
        seen = text
    if not boundaries or boundaries[-1] != len(ids):
        boundaries.append(len(ids))
    return boundaries


def implicit_prm_scores(
    per_token_logratio: torch.Tensor,
    boundaries: List[int],
    beta: float,
) -> ImplicitPRMScores:
    """Turn per-token log-ratios into sequence reward, cumulative Q, step rewards.

    per_token_logratio: (R,) = logpi_theta(y_t|..) - logpi_ref(y_t|..) per response token.
    boundaries: token indices (response-relative) where each step ends.
    """
    scaled = beta * per_token_logratio                  # per-token reward contribution
    cum = torch.cumsum(scaled, dim=0)                   # (R,) running Q per token
    seq_reward = float(cum[-1].item()) if len(cum) else 0.0

    q_values: List[float] = []
    step_rewards: List[float] = []
    prev_q = 0.0
    for b in boundaries:
        q_t = float(cum[b - 1].item())                  # cumulative Q at end of this step
        q_values.append(q_t)
        step_rewards.append(q_t - prev_q)               # advantage = this step's process reward
        prev_q = q_t

    min_step = min(step_rewards) if step_rewards else seq_reward
    return ImplicitPRMScores(
        reward=seq_reward,
        q_values=q_values,
        step_rewards=step_rewards,
        min_step_reward=min_step,
        n_steps=len(step_rewards),
    )


@torch.no_grad()
def score_response(
    policy_model,
    ref_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    beta: float,
    delimiter: str = "\n\n",
    drop_ref: bool = False,
) -> ImplicitPRMScores:
    """Full inference-time scoring of one (prompt, response) pair.

    If drop_ref=True, the reference term is dropped (constant reference), which
    the paper shows is often fine at inference and halves cost (Sec. 5.5).
    """
    device = next(policy_model.parameters()).device
    full = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
    attn = torch.ones_like(full)
    prompt_len = prompt_ids.shape[0]

    lp_theta = token_logprobs(policy_model, full, attn, prompt_len)
    if drop_ref:
        logratio = lp_theta
    else:
        lp_ref = token_logprobs(ref_model, full, attn, prompt_len)
        logratio = lp_theta - lp_ref

    boundaries = step_boundaries(response_ids, tokenizer, delimiter)
    return implicit_prm_scores(logratio, boundaries, beta)
