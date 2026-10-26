"""Official SQuAD 2.0 normalisation, Exact Match and F1 (same logic as the official script)."""
import collections
import re
import string


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list:
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    g, p = _tokens(gold), _tokens(pred)
    if len(g) == 0 or len(p) == 0:
        # If either is "no answer", F1 is 1 only when both are.
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def score_item(gold_answers: list, pred):
    """gold_answers == [] means unanswerable. pred None means 'no answer'.
    Returns (EM, F1), taking the best over all gold answers."""
    golds = gold_answers if gold_answers else [""]
    text = pred or ""
    return (
        max(compute_exact(g, text) for g in golds),
        max(compute_f1(g, text) for g in golds),
    )
