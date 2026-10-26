"""Run the model over a set of questions and save raw scores (BEFORE the threshold).

The output file feeds tune_threshold.py.

Examples (repo root):
  python train/evaluate.py --model_dir models/roberta --n 3000                      # SQuAD 2.0 dev subset
  python train/evaluate.py --model_dir models/roberta --source file --file data/public_sample.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qa.model import QAModel  # noqa: E402


def load_items(args):
    if args.source == "file":
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        items = data["items"] if isinstance(data, dict) else data
        return [{"id": it["id"], "context": it["context"], "question": it["question"],
                 "golds": [a["text"] for a in it.get("answers", [])]} for it in items]
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    if args.n:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
    return [{"id": ex["id"], "context": ex["context"], "question": ex["question"],
             "golds": list(ex["answers"]["text"])} for ex in ds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--source", choices=["squad_dev", "file"], default="squad_dev")
    p.add_argument("--file", default="data/public_sample.json")
    p.add_argument("--n", type=int, default=3000, help="0 = whole dev set (11.9k questions)")
    p.add_argument("--out", default=None, help="default: logs/<run>/dev_analysis.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QAModel(model_dir, device=device)
    items = load_items(args)

    out = Path(args.out) if args.out else ROOT / "logs" / model_dir.name / (
        "dev_analysis.jsonl" if args.source == "squad_dev" else f"{Path(args.file).stem}_analysis.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in tqdm(items, desc="evaluating"):
            a = model.analyze(it["context"], it["question"])
            f.write(json.dumps({"id": it["id"], "golds": it["golds"], "pred_text": a.text,
                                "pred_start": a.start, "diff": a.diff}) + "\n")
    print(f"wrote {len(items)} rows to {out}\nNext: python train/tune_threshold.py --analysis {out}")


if __name__ == "__main__":
    main()
