"""Turn raw model logits into (answer, start, score).

Pure numpy / python: no torch, no transformers. That is on purpose, so it can be
unit-tested quickly and read in one sitting.

How the no-answer decision works (standard SQuAD 2.0 recipe):
  * null_score  = start_logit[CLS] + end_logit[CLS]   ("the answer is nothing")
  * span_score  = best start_logit[i] + end_logit[j]  over valid context tokens
  * diff        = null_score - span_score
  * answer is null  <=>  diff > threshold
A long context is split in several windows. We keep the best span over ALL windows
and the LOWEST null_score over windows (a window that sees the answer lowers it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

NEG_INF = -1e9


@dataclass
class WindowOutput:
    start_logits: np.ndarray                     # shape (seq_len,)
    end_logits: np.ndarray                       # shape (seq_len,)
    offsets: Sequence[Optional[tuple]]           # (char_start, char_end) per token; None if token is NOT context


@dataclass
class Analysis:
    text: Optional[str]      # best non-null span (verbatim from context) or None
    start: Optional[int]     # char offset of that span in the context
    span_score: float
    null_score: float

    @property
    def diff(self) -> float:
        return self.null_score - self.span_score


def analyze_windows(context: str, windows: Sequence[WindowOutput],
                    n_best: int = 20, max_answer_tokens: int = 30) -> Analysis:
    null_score = math.inf
    best_score, best_char = NEG_INF, None

    for w in windows:
        sl, el = np.asarray(w.start_logits), np.asarray(w.end_logits)
        null_score = min(null_score, float(sl[0] + el[0]))   # index 0 is [CLS] / <s>

        starts = np.argsort(sl)[-n_best:][::-1]
        ends = np.argsort(el)[-n_best:][::-1]
        for s in starts:
            if w.offsets[s] is None:
                continue
            for e in ends:
                if w.offsets[e] is None or e < s or (e - s + 1) > max_answer_tokens:
                    continue
                score = float(sl[s] + el[e])
                if score > best_score:
                    best_score = score
                    best_char = (w.offsets[s][0], w.offsets[e][1])

    if best_char is None:
        return Analysis(None, None, NEG_INF, null_score)

    c0, c1 = best_char
    raw = context[c0:c1]
    text = raw.strip()
    if not text:
        return Analysis(None, None, NEG_INF, null_score)
    start = c0 + (len(raw) - len(raw.lstrip()))          # keep start in sync after strip()
    assert context[start:start + len(text)] == text
    return Analysis(text, start, best_score, null_score)


def _sigmoid(x: float) -> float:
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def decide(a: Analysis, threshold: float):
    """Apply the no-answer threshold. Returns (answer, start, score in [0, 1])."""
    p_answer = _sigmoid(threshold - a.diff)
    if a.text is None or a.diff > threshold:
        return None, None, 1.0 - p_answer
    return a.text, a.start, p_answer
