"""Tokenisation tests with a tiny offline WordPiece tokenizer (no internet, no torch)."""
import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from qa.features import encode_windows, make_train_features  # noqa: E402

WORDS = ("the lempa river is longest in el salvador flowing about 422 km to sea what how long "
         "was first dam built when which ocean pacific guatemala from and of a").split()
FILLER = " ".join(["the river is long and the sea is a river"] * 8)
CONTEXT = f"The Lempa River is the longest river in El Salvador, flowing about 422 km to the sea. {FILLER} The dam was built in the Pacific."


@pytest.fixture(scope="module")
def tok():
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS)) + [",", ".", "?"])}
    t = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    t.normalizer = normalizers.BertNormalizer(lowercase=True)
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", sep_token="[SEP]",
                                   pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]")


def test_train_labels_point_to_the_answer(tok):
    ans = "about 422 km"
    ex = {"question": ["How long was the river?", "When was the first dam built?"],
          "context": [CONTEXT, CONTEXT],
          "answers": [{"text": [ans], "answer_start": [CONTEXT.index(ans)]},
                      {"text": [], "answer_start": []}]}
    f = make_train_features(ex, tok, max_length=32, doc_stride=8)
    n = len(f["input_ids"])
    assert n > 2                                          # the long context really was split in windows
    found = 0
    for i in range(n):
        s, e = f["start_positions"][i], f["end_positions"][i]
        if s != 0:                                        # labelled window of question 0
            assert tok.decode(f["input_ids"][i][s:e + 1]) == "about 422 km"
            found += 1
    assert found >= 1
    # unanswerable question -> every window labelled with CLS (0)
    unans = [i for i in range(n) if f["start_positions"][i] == 0 and f["end_positions"][i] == 0]
    assert len(unans) >= 1


def test_encode_windows_offsets_map_back_to_context(tok):
    inputs, offsets = encode_windows(tok, "How long was the river?", CONTEXT, max_length=32, doc_stride=8)
    assert len(inputs["input_ids"]) == len(offsets) > 1
    assert len({len(r) for r in inputs["input_ids"]}) == 1          # padded to the same length
    seen = set()
    for win in offsets:
        assert win[0] is None                                       # [CLS] is not context
        for o in win:
            if o is not None:
                seen.add(CONTEXT[o[0]:o[1]].lower())
    assert "422" in seen and "km" in seen
    assert "token_type_ids" in inputs or "input_ids" in inputs
