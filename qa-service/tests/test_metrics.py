from qa.metrics import normalize_answer, score_item


def test_normalize():
    assert normalize_answer("The  Lempa, River!") == "lempa river"


def test_exact_and_f1():
    assert score_item(["about 422 km"], "About 422 km.") == (1, 1.0)
    em, f1 = score_item(["16 attention heads"], "16")
    assert em == 0 and 0 < f1 < 1


def test_best_over_multiple_golds():
    assert score_item(["16", "16 attention heads"], "16 attention heads")[0] == 1


def test_unanswerable():
    assert score_item([], None) == (1, 1.0)          # correct null
    assert score_item([], "something") == (0, 0.0)   # answered when it should not
    assert score_item(["x"], None) == (0, 0.0)       # wrong null
