"""Shared helpers: device, model loading, GSM8K answer extraction, prompting."""
from __future__ import annotations

import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str, dtype: str = "float32", device: Optional[str] = None):
    device = device or get_device()
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tok


# ---- GSM8K helpers -------------------------------------------------------

_ANS_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")


def gold_answer(gsm8k_answer_field: str) -> Optional[str]:
    """Extract the gold final answer from a GSM8K `answer` field (after ####)."""
    m = _ANS_RE.search(gsm8k_answer_field)
    return _normalize_number(m.group(1)) if m else None


def extract_pred_answer(text: str) -> Optional[str]:
    """Extract a model's final numeric answer.

    Priority: \\boxed{X}  ->  the LAST 'answer is/: X'  ->  '#### X'  ->  last
    number overall. We take the LAST 'answer is' match (not the first) and do
    NOT treat a bare '=' as an answer cue: the model writes the final answer at
    the end ("The answer is 18"), while '=' and the first 'answer is' occur
    inside intermediate steps. Matching those was labeling correct solutions
    as wrong.
    """
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        n = _NUM_RE.search(boxed.group(1))
        if n:
            return _normalize_number(n.group(0))
    phrases = list(re.finditer(r"answer\s*(?:is|:)?\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE))
    if phrases:
        return _normalize_number(phrases[-1].group(1))
    hashed = _ANS_RE.search(text)
    if hashed:
        return _normalize_number(hashed.group(1))
    nums = _NUM_RE.findall(text)
    return _normalize_number(nums[-1]) if nums else None


def _normalize_number(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    return pred == gold


GSM8K_SYSTEM = (
    "You are a careful math tutor. Solve the problem step by step. "
    "Put each reasoning step on its own line separated by a blank line. "
    "End with a final line: 'The answer is <number>'."
)

MATH_SYSTEM = (
    "You are a careful mathematician. Solve the problem step by step. "
    "Put each reasoning step on its own line separated by a blank line. "
    "End with the final answer in \\boxed{}."
)


def build_prompt(tokenizer, question: str, system: str = GSM8K_SYSTEM) -> str:
    """Chat-templated prompt string for a question."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---- MATH helpers --------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST \\boxed{...} in text (brace-balanced)."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize_math_answer(s: Optional[str]) -> Optional[str]:
    """Light normalization for comparing MATH answers as strings.

    Not a full symbolic equivalence check (a known limitation), but strips the
    common latex noise (\\left, spaces, \\dfrac vs \\frac, $, units text, ...) so
    that superficially-different-but-equal answers usually match.
    """
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("\\;", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\%", ""), ("%", ""), ("^{\\circ}", ""), ("^\\circ", ""),
                 ("\\$", ""), (" ", "")]:
        s = s.replace(a, b)
    s = s.strip("$").rstrip(".").strip()
    # a bare integer like "007" or "+7" -> canonical number form
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s
