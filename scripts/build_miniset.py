"""Build data/team_miniset.json from hand-written items.

Usage (from the repository root):
    python scripts/build_miniset.py

What it does
  1. Holds every item as plain Python (context, question, answer).
  2. Computes the 0-based character offset `start` of each answer.
  3. Verifies: context[start:start+len(answer)] == answer, ids are unique,
     the context length is 300-5000 characters (the contract's maximum is 5000),
     impossible items have no answer.
  4. Writes data/team_miniset.json (same format as data/public_sample.json)
     and data/team_miniset_meta.json (item category, used only for analysis).

The items were written from the course readings (Attention Is All You Need,
LoRA, "How transferable are features in deep neural networks?") and were NOT
checked against any model output.
"""
import json
import sys
from collections import Counter
from pathlib import Path

OUT_DIR = Path("data")

# --------------------------------------------------------------------------
# Contexts (paraphrased from the readings). Several questions can share one.
# --------------------------------------------------------------------------
AT_ARCH = (
    "The Transformer encoder is built from a stack of six identical layers. "
    "Each layer contains two sub-layers: a multi-head self-attention mechanism and a "
    "position-wise fully connected feed-forward network. A residual connection surrounds "
    "each sub-layer and is followed by layer normalization. To make these residual "
    "connections possible, every sub-layer and the embedding layers produce outputs of "
    "dimension 512. The decoder is also a stack of six identical layers, but it inserts a "
    "third sub-layer that performs attention over the output of the encoder stack."
)

AT_HEADS = (
    "Instead of computing one attention function over vectors of full dimension, the "
    "Transformer projects the queries, keys and values several times with different learned "
    "linear projections. In the base model h is 8, and each head works with keys and values "
    "of dimension 64, which is the model dimension divided by the number of heads. Because "
    "each head is smaller, the total computational cost is similar to that of single-head "
    "attention with full dimensionality. The outputs of all heads are concatenated and "
    "projected once more to produce the final values."
)

AT_TRAIN = (
    "The authors trained their models on one machine with eight NVIDIA P100 GPUs. A training "
    "step of the base model took about 0.4 seconds, and the base models were trained for "
    "100,000 steps, roughly twelve hours. The big models needed about 1.0 second per step and "
    "ran for 300,000 steps. The optimizer was Adam, with a learning rate that grows linearly "
    "during the first 4000 steps and then decays in proportion to the inverse square root of "
    "the step number."
)

AT_DATA = (
    "The English-German experiments used the standard WMT 2014 training set of about 4.5 "
    "million sentence pairs, encoded with byte-pair encoding and a shared source-target "
    "vocabulary of about 37000 tokens. The English-French experiments used the much larger "
    "WMT 2014 set of 36 million sentences, split into a 32000 word-piece vocabulary. Each "
    "training batch held roughly 25000 source tokens and 25000 target tokens."
)

AT_BLEU = (
    "On English-to-German translation the big Transformer reached 28.4 BLEU, more than 2.0 "
    "BLEU above the best earlier results, including ensembles. On English-to-French, the big "
    "model scored 41.0 BLEU as a single model. The base model also beat all previously "
    "published models at a fraction of the training cost. For the base model the authors "
    "averaged the last 5 checkpoints, while for the big models they averaged the last 20."
)

LO_METHOD = (
    "LoRA keeps the pre-trained weight matrix W0 frozen and represents its update as the "
    "product of two small matrices, B and A, so that the adapted weight is W0 plus BA. If W0 "
    "has dimensions d by k, then B has dimensions d by r and A has dimensions r by k, where "
    "the rank r is much smaller than d and k. Matrix A is initialized with a random Gaussian, "
    "while B starts at zero, so the update is exactly nothing at the beginning of training. "
    "Only A and B receive gradients."
)

LO_DEPLOY = (
    "Because the update BA can be merged into the frozen weights, LoRA adds no inference "
    "latency, unlike adapter layers. On GPT-3 175B, training with LoRA cut the VRAM "
    "consumption from 1.2 TB to 350 GB. With a rank of 4 and only the query and value "
    "projection matrices adapted, the checkpoint shrank from 350 GB to 35 MB, roughly 10,000 "
    "times smaller. A single pre-trained model can then serve many tasks, since switching "
    "tasks only requires swapping the small LoRA matrices."
)

LO_EXPER = (
    "The authors evaluated LoRA on RoBERTa, DeBERTa and GPT-2 before scaling up to GPT-3 175B. "
    "RoBERTa base has 125 million parameters and RoBERTa large has 355 million, while DeBERTa "
    "XXL has 1.5 billion parameters. For the language understanding experiments they used the "
    "GLUE benchmark, and in most experiments LoRA was applied only to the query and value "
    "matrices of the attention layers."
)

LO_MOTIV = (
    "Full fine-tuning updates every parameter, so each downstream task requires its own copy "
    "of the model. For GPT-3 this means 175 billion trainable parameters per task. Earlier "
    "workarounds, such as adapter layers, add extra depth and therefore increase inference "
    "latency, while prompt-based methods consume part of the available sequence length. LoRA "
    "was designed to avoid both of these problems."
)

YO_SETUP = (
    "The authors split the 1000 ImageNet classes into two random halves of 500 classes each, "
    "called task A and task B, with about 645,000 examples in each half. They trained one "
    "eight-layer network on each half, called baseA and baseB. They then copied the first n "
    "layers of one network into a new network and either froze them or let them keep "
    "training, which tests how general the features of those layers are."
)

YO_BNB = (
    "In the BnB treatment, the first n layers of a network are copied and frozen, the upper "
    "layers are reinitialized, and the network is retrained on the same task. Performance at "
    "layers 1 and 2 matched the baseline, but layers 3 to 6 became worse, especially layers 4 "
    "and 5. The authors attribute this drop to fragile co-adapted features, where neighboring "
    "layers depend on each other in ways that are hard to relearn."
)

YO_BOOST = (
    "A surprising result is that initializing a network with transferred features and then "
    "fine-tuning all layers boosts generalization. Keeping anywhere from one to seven layers "
    "produces improved performance, with slightly better results as more layers are kept. The "
    "average boost across layers 1 to 7 is 1.6 percent, and the average when at least five "
    "layers are kept is 2.1 percent. The authors note that this benefit lingers even after "
    "substantial retraining on the target dataset."
)

YO_DIST = (
    "Feature transferability also decreases as the distance between the base task and the "
    "target task grows. Even so, features transferred from a distant task still beat random "
    "weights. In the experiment with random, untrained lower layers, performance fell to "
    "near chance level from layer 3 onward, which suggests that making random weights work is "
    "not straightforward. Overall, transferred features are a better starting point than "
    "random initialization."
)

# --------------------------------------------------------------------------
# LONG contexts (> 2000 characters, i.e. more than one 384-token window).
# They check that the service combines windows correctly: the answer sits near
# the start, the middle or the end of the text.
# --------------------------------------------------------------------------
AT_LONG = (
    "Self-attention was compared with recurrent and convolutional layers on three criteria. "
    "The first criterion is the total computational complexity per layer. The second is how "
    "much computation can be parallelized, measured by the minimum number of sequential "
    "operations that are required. The third is the length of the path that signals must "
    "travel between two distant positions in the network, since shorter paths make "
    "long-range dependencies easier to learn. Learning such dependencies is a key challenge "
    "in many sequence transduction tasks, which is why the maximum path length between any "
    "two input and output positions is compared for each layer type.\n\n"
    "On the second criterion, a self-attention layer connects every position to every other "
    "position using a constant number of sequentially executed operations, while a recurrent "
    "layer needs a number of sequential operations that grows linearly with the sequence "
    "length n. On the first criterion, self-attention layers are faster than recurrent layers "
    "whenever the sequence length n is smaller than the representation dimensionality d. This "
    "is usually the case for the sentence representations used in machine translation, which "
    "are built from word pieces or byte pairs.\n\n"
    "For tasks with very long sequences, self-attention could be restricted to a "
    "neighborhood of size r around each output position. That change would raise the maximum "
    "path length to the ratio of n to r, and the authors leave this idea for future work.\n\n"
    "Convolutions behave differently. A single convolutional layer with kernel width k "
    "smaller than n does not connect all pairs of input and output positions. Connecting "
    "them requires a stack of about n divided by k layers when the kernels are contiguous, "
    "or a logarithmic number of layers, with base k, when dilated convolutions are used. "
    "Either way, the longest paths between positions become longer. Convolutional layers are "
    "also generally more expensive than recurrent layers, by a factor of k. Separable "
    "convolutions bring the cost down considerably, but even with k equal to n their "
    "complexity matches that of a self-attention layer combined with a point-wise "
    "feed-forward layer, which is the design the authors chose. As a side benefit, "
    "self-attention can make models easier to interpret, because individual attention heads "
    "appear to learn different tasks."
)

LO_LONG = (
    "LoRA starts from the observation that pre-trained language models have a low intrinsic "
    "dimension, meaning that they can still learn efficiently after a random projection to a "
    "much smaller subspace. The authors hypothesize that the change in the weights during "
    "adaptation also has a low intrinsic rank. For a pre-trained weight matrix W0 of size d "
    "by k, the update is therefore written as the product BA, with B of size d by r and A of "
    "size r by k, and a rank r far smaller than both d and k. Matrices of this kind exist in "
    "every dense layer of the network, so the same reparametrization can be applied to any of "
    "them.\n\n"
    "During training W0 receives no gradient updates, and both W0 and the product BA are "
    "multiplied with the same input, after which their output vectors are summed. The update "
    "is also scaled by alpha divided by r, where alpha is a constant. When optimizing with "
    "Adam, tuning alpha is roughly the same as tuning the learning rate, so the authors "
    "simply set alpha to the first value of r they try and never tune it. This scaling "
    "reduces the need to retune hyperparameters when r changes.\n\n"
    "The number of trainable parameters follows from the rank and the shapes of the adapted "
    "matrices. On GPT-3 175B the authors set a budget of 18 million parameters, roughly 35 MB "
    "when stored in half precision, spread over all 96 layers. That budget corresponds to a "
    "rank of 8 if one type of attention weight is adapted, or a rank of 4 if two types are "
    "adapted. In the comparison that followed, adapting the query and value matrices together "
    "gave the best overall results, ahead of adapting a single type of matrix with a larger "
    "rank. Surprisingly, a very small rank of one or two already works well, even though the "
    "full dimension d of GPT-3 is as high as 12,288. This is why LoRA is efficient in both "
    "storage and computation.\n\n"
    "LoRA also adds no inference latency. When the model is deployed in production, the "
    "product BA can be computed and added to W0 once, so inference runs exactly as in a "
    "fully fine-tuned model. Switching to another downstream task only requires subtracting "
    "BA and adding a different update, a quick operation with very little memory overhead."
)

YO_LONG = (
    "Many deep networks trained on natural images learn first-layer features that resemble "
    "Gabor filters and color blobs. These features seem not to depend on the dataset or the "
    "task, so the authors call them general. In contrast, the features computed by the last "
    "layer must depend on the task, because in a network with a softmax output each unit is "
    "specific to one class, and the authors call these features specific. The paper measures "
    "how features move from general to specific across the layers of a convolutional network "
    "trained on ImageNet.\n\n"
    "Transferability suffers from two separate problems. The first is the specialization of "
    "higher-layer neurons to their original task at the expense of performance on the new "
    "one, which was expected. The second is optimization difficulty that appears when a "
    "network is split between co-adapted neurons, which was not expected. Depending on "
    "whether features are transferred from the bottom, the middle or the top of the network, "
    "either of these two issues may dominate.\n\n"
    "As a reference point, a network trained on a random subset of 500 classes reaches a "
    "top-1 accuracy of 0.625, or 37.5 percent error, which is lower than the 42.5 percent "
    "top-1 error of the network trained on all 1000 classes. The error is lower not because "
    "the network learned more, but because with only 500 classes there are only half as many "
    "ways to make a mistake. The authors also found that transferability decreases as the "
    "distance between the base task and the target task increases, yet transferring features "
    "even from a distant task can be better than using random features. Finally, they report "
    "that initializing a network with transferred features can give a boost to "
    "generalization that lingers even after fine-tuning on the target dataset.\n\n"
    "A control experiment copies the first n layers of a network, freezes them, reinitializes "
    "the upper layers and retrains on the same task. Performance at layers 1 and 2 matched "
    "the baseline, but layers 3 to 6, especially layers 4 and 5, did worse. The authors read "
    "this drop as evidence of fragile co-adapted features, which interact with each other in "
    "a complex way that is hard to relearn."
)

# --------------------------------------------------------------------------
# Items: (id, context, question, answer or None, anchor or None, category)
#   answer None      -> impossible item (answers = [])
#   anchor           -> a unique phrase inside the context that contains the answer;
#                       used when the answer text alone appears more than once.
#   category         -> answerable | impossible_obvious | impossible_near
# --------------------------------------------------------------------------
ITEMS = [
    # ---- Attention Is All You Need ----
    ("mini-at-01", AT_ARCH, "How many identical layers does the Transformer encoder stack contain?", "six", "stack of six identical layers. Each", "answerable"),
    ("mini-at-02", AT_ARCH, "What output dimension do the sub-layers and embedding layers produce?", "512", None, "answerable"),
    ("mini-at-03", AT_ARCH, "How many attention heads does each encoder layer use?", None, None, "impossible_obvious"),
    ("mini-at-04", AT_HEADS, "How many attention heads does the base Transformer use?", "8", "h is 8", "answerable"),
    ("mini-at-05", AT_HEADS, "What is the dimension of the keys and values in each head?", "64", None, "answerable"),
    ("mini-at-06", AT_TRAIN, "How many warmup steps does the learning rate schedule use?", "4000", None, "answerable"),
    ("mini-at-07", AT_TRAIN, "How many steps were the big models trained for?", "300,000", None, "answerable"),
    ("mini-at-08", AT_TRAIN, "How many warmup steps did the authors use with the SGD optimizer?", None, None, "impossible_near"),
    ("mini-at-09", AT_DATA, "About how many sentence pairs are in the WMT 2014 English-German training set?", "4.5 million", None, "answerable"),
    ("mini-at-10", AT_DATA, "What vocabulary size did the English-German experiments use?", "37000 tokens", None, "answerable"),
    ("mini-at-11", AT_DATA, "How large was the word-piece vocabulary used for the English-German experiments?", None, None, "impossible_near"),
    ("mini-at-12", AT_BLEU, "What BLEU score did the big Transformer reach on English-to-French?", "41.0", None, "answerable"),
    ("mini-at-13", AT_BLEU, "How many checkpoints were averaged for the big models?", "20", "last 20", "answerable"),
    ("mini-at-14", AT_BLEU, "What BLEU score did the big Transformer reach on English-to-Spanish?", None, None, "impossible_obvious"),
    # long context (answer near the END of the text)
    ("mini-at-15", AT_LONG, "By what factor are convolutional layers more expensive than recurrent layers?", "a factor of k", None, "answerable"),
    ("mini-at-16", AT_LONG, "What maximum path length does a recurrent layer have?", None, None, "impossible_near"),
    # ---- LoRA ----
    ("mini-lo-01", LO_METHOD, "What is matrix B initialized to?", "zero", "B starts at zero", "answerable"),
    ("mini-lo-02", LO_METHOD, "How is matrix A initialized?", "a random Gaussian", None, "answerable"),
    ("mini-lo-03", LO_METHOD, "Which matrices receive gradients during training?", "A and B", "Only A and B receive", "answerable"),
    ("mini-lo-04", LO_METHOD, "What variance is used for the random Gaussian that initializes A?", None, None, "impossible_near"),
    ("mini-lo-05", LO_DEPLOY, "How much VRAM did training GPT-3 175B need with LoRA?", "350 GB", "to 350 GB. With", "answerable"),
    ("mini-lo-06", LO_DEPLOY, "How big is the LoRA checkpoint on GPT-3 175B with rank 4?", "35 MB", "to 35 MB", "answerable"),
    ("mini-lo-07", LO_DEPLOY, "Which matrices were adapted in the rank-4 setting?", "query and value projection matrices", None, "answerable"),
    ("mini-lo-08", LO_DEPLOY, "From what size did the checkpoint shrink when LoRA was applied to GPT-2?", None, None, "impossible_near"),
    ("mini-lo-09", LO_EXPER, "How many parameters does RoBERTa large have?", "355 million", None, "answerable"),
    ("mini-lo-10", LO_EXPER, "Which benchmark was used for the language understanding experiments?", "GLUE benchmark", None, "answerable"),
    ("mini-lo-11", LO_EXPER, "How many parameters does DeBERTa XXL have?", "1.5 billion", None, "answerable"),
    ("mini-lo-12", LO_EXPER, "How many parameters does GPT-2 medium have?", None, None, "impossible_obvious"),
    ("mini-lo-13", LO_MOTIV, "How many trainable parameters does full fine-tuning of GPT-3 involve?", "175 billion", None, "answerable"),
    ("mini-lo-14", LO_MOTIV, "How many trainable parameters does LoRA need per task for GPT-3?", None, None, "impossible_near"),
    # long context (answer in the MIDDLE of the text)
    ("mini-lo-15", LO_LONG, "What parameter budget did the authors set on GPT-3 175B?", "18 million parameters", None, "answerable"),
    # ---- How transferable are features ... ----
    ("mini-yo-01", YO_SETUP, "How many classes does each of the two tasks contain?", "500 classes", None, "answerable"),
    ("mini-yo-02", YO_SETUP, "About how many examples does each half contain?", "645,000", None, "answerable"),
    ("mini-yo-03", YO_SETUP, "How many layers does each of the base networks have?", "eight", "one eight-layer network", "answerable"),
    ("mini-yo-04", YO_SETUP, "How many examples did the full 1000-class ImageNet set contain?", None, None, "impossible_near"),
    ("mini-yo-05", YO_BNB, "Which layers showed the biggest performance drop?", "layers 4 and 5", None, "answerable"),
    ("mini-yo-06", YO_BNB, "What explanation do the authors give for the drop?", "fragile co-adapted features", None, "answerable"),
    ("mini-yo-07", YO_BNB, "What was the top-1 accuracy of the BnB network at layer 7?", None, None, "impossible_obvious"),
    ("mini-yo-08", YO_BOOST, "What is the average boost when at least five layers are kept?", "2.1 percent", None, "answerable"),
    ("mini-yo-09", YO_BOOST, "What is the average boost across layers 1 to 7?", "1.6 percent", None, "answerable"),
    ("mini-yo-10", YO_BOOST, "What was the average boost when exactly two layers were kept?", None, None, "impossible_near"),
    ("mini-yo-11", YO_DIST, "From which layer does performance with random weights fall to near chance?", "layer 3", "from layer 3 onward", "answerable"),
    ("mini-yo-12", YO_DIST, "What do features transferred from a distant task still outperform?", "random weights", "still beat random weights", "answerable"),
    # long context (answer near the START of the text)
    ("mini-yo-13", YO_LONG, "What do the authors call the first-layer features that resemble Gabor filters?", "general", "so the authors call them general", "answerable"),
]


def find_start(context, answer, anchor):
    """Return the 0-based offset of `answer` inside `context`, or raise ValueError."""
    if anchor is not None:
        if context.count(anchor) != 1:
            raise ValueError(f"anchor {anchor!r} must appear exactly once")
        if anchor.count(answer) < 1:
            raise ValueError(f"answer {answer!r} is not inside anchor {anchor!r}")
        return context.index(anchor) + anchor.index(answer)
    if context.count(answer) != 1:
        raise ValueError(f"answer {answer!r} appears {context.count(answer)} times; add an anchor")
    return context.index(answer)


def main():
    errors = []
    items, meta = [], {}
    ids = [it[0] for it in ITEMS]
    for dup, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate id: {dup}")

    for item_id, context, question, answer, anchor, category in ITEMS:
        if not 300 <= len(context) <= 5000:   # 5000 = the contract's maximum
            errors.append(f"{item_id}: context length {len(context)} outside 300-5000")
        if len(question) > 500:
            errors.append(f"{item_id}: question longer than 500 characters")
        if answer is None:
            if category == "answerable":
                errors.append(f"{item_id}: answerable item without answer")
            answers = []
        else:
            try:
                start = find_start(context, answer, anchor)
            except ValueError as exc:
                errors.append(f"{item_id}: {exc}")
                continue
            if context[start:start + len(answer)] != answer:  # the contract's rule
                errors.append(f"{item_id}: context[start:start+len] != answer")
                continue
            answers = [{"text": answer, "start": start}]
        items.append({
            "id": item_id,
            "context": context,
            "question": question,
            "is_impossible": answer is None,
            "answers": answers,
        })
        meta[item_id] = category

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    n_imp = sum(1 for it in items if it["is_impossible"])
    n_near = sum(1 for c in meta.values() if c == "impossible_near")
    print(f"items: {len(items)}  impossible: {n_imp} ({n_imp / len(items):.0%})  "
          f"near-answerable: {n_near} ({n_near / max(n_imp, 1):.0%} of impossible)")

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "version": "1.0",
        "description": "Team mini-set on course readings (Attention Is All You Need, LoRA, "
                       "How transferable are features). Same format as public_sample.json.",
        "items": items,
    }
    (OUT_DIR / "team_miniset.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "team_miniset_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote data/team_miniset.json and data/team_miniset_meta.json")


if __name__ == "__main__":
    main()
