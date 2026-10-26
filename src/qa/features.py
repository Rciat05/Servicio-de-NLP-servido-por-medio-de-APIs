"""Tokenisation helpers shared by training (labels) and serving (windows).

They need a *fast* tokenizer because we rely on `offset_mapping`
(token -> character positions in the original context).
"""
from __future__ import annotations


def make_train_features(examples, tokenizer, max_length: int, doc_stride: int):
    """SQuAD 2.0 preprocessing for training (same logic as HF's run_qa example).

    A long context becomes several windows (features). For every window we label
    the answer's first/last token. If the answer is not inside the window, or the
    question is unanswerable, the label is the [CLS] position (index 0 = "no answer").
    """
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)          # 0=question, 1=context, None=special/pad
        answers = examples["answers"][sample_map[i]]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        tok_start = 0
        while sequence_ids[tok_start] != 1:
            tok_start += 1
        tok_end = len(input_ids) - 1
        while sequence_ids[tok_end] != 1:
            tok_end -= 1

        if not (offsets[tok_start][0] <= start_char and offsets[tok_end][1] >= end_char):
            # the answer is not fully inside this window
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while tok_start < len(offsets) and offsets[tok_start][0] <= start_char:
                tok_start += 1
            start_positions.append(tok_start - 1)
            while offsets[tok_end][1] >= end_char:
                tok_end -= 1
            end_positions.append(tok_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def encode_windows(tokenizer, question: str, context: str, max_length: int, doc_stride: int):
    """Inference-time encoding of ONE (question, context) pair.

    Returns:
      model_inputs: dict of lists, one row per window (padded only to the longest window,
                    so short contexts stay cheap on CPU)
      offsets:      per window, per token: (char_start, char_end) or None if not a context token
    """
    enc = tokenizer(
        question.lstrip(),
        context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )
    offsets = []
    for i, offs in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        offsets.append([tuple(o) if seq_ids[k] == 1 else None for k, o in enumerate(offs)])
    model_inputs = {k: enc[k] for k in tokenizer.model_input_names if k in enc}
    return model_inputs, offsets
