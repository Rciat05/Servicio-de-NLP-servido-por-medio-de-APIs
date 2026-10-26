import re

import numpy as np

from qa.postprocess import WindowOutput, analyze_windows, decide

CONTEXT = "The Lempa River is the longest river in El Salvador, flowing about 422 km to the sea."


def make_window(context, peak_start_word=None, peak_end_word=None, null=0.0, prefix=3, peak=5.0):
    """Fake window: [CLS], `prefix` question tokens, then one token per whitespace word."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", context)]
    n = 1 + prefix + len(words)
    sl, el = np.full(n, -5.0), np.full(n, -5.0)
    sl[0] = el[0] = null / 2
    offsets = [None] * (1 + prefix) + words
    if peak_start_word is not None:
        sl[1 + prefix + peak_start_word] = peak
        el[1 + prefix + peak_end_word] = peak
    return WindowOutput(sl, el, offsets)


def test_finds_span_and_exact_offsets():
    words = CONTEXT.split()
    i = words.index("about")
    w = make_window(CONTEXT, i, i + 2)                  # "about 422 km"
    a = analyze_windows(CONTEXT, [w])
    assert a.text == "about 422 km"
    assert CONTEXT[a.start:a.start + len(a.text)] == a.text


def test_null_when_cls_wins():
    w = make_window(CONTEXT, 2, 2, null=40.0)           # CLS far stronger than any span
    ans, start, score = decide(analyze_windows(CONTEXT, [w]), threshold=0.0)
    assert ans is None and start is None and 0 <= score <= 1


def test_threshold_moves_the_decision():
    w = make_window(CONTEXT, 2, 2, null=12.0)           # span score 10 vs null 12 -> diff = 2
    a = analyze_windows(CONTEXT, [w])
    assert decide(a, threshold=0.0)[0] is None          # diff 2 > 0 -> null
    assert decide(a, threshold=5.0)[0] is not None      # diff 2 <= 5 -> answer


def test_multiple_windows_best_span_and_min_null():
    weak = make_window(CONTEXT, 0, 0, null=30.0)
    strong = make_window(CONTEXT, 5, 5, null=2.0, peak=8.0)
    a = analyze_windows(CONTEXT, [weak, strong])
    assert a.null_score == 2.0                          # lowest null over windows
    assert a.text == "longest"                          # word #5, from the strong window


def test_strip_keeps_start_consistent():
    ctx = "abc   hello world"
    sl, el = np.full(4, -5.0), np.full(4, -5.0)
    sl[1] = el[1] = 5.0
    w = WindowOutput(sl, el, [None, (3, 11), (11, 17), None])   # offset includes leading spaces
    a = analyze_windows(ctx, [w])
    assert a.text == "hello" and a.start == 6 and ctx[a.start:a.start + 5] == "hello"


def test_no_valid_span_gives_null():
    sl = np.zeros(3)
    a = analyze_windows("x", [WindowOutput(sl, sl.copy(), [None, None, None])])
    assert decide(a, 0.0)[0] is None
