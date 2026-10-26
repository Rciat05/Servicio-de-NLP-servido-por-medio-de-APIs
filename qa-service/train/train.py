"""Fine-tune a BASE encoder on SQuAD 2.0 for extractive QA (plain PyTorch loop, no Trainer).

Examples (run from the repo root):
  # smoke test on a laptop CPU, ~2 minutes: only checks that the pipeline runs
  python train/train.py --model distilbert-base-uncased --max_train_samples 1000 --val_samples 200 --epochs 1 --run_name smoke

  # real runs (GPU: Kaggle / Colab / university)
  python train/train.py --model distilbert-base-uncased --epochs 3 --lr 3e-5 --run_name distilbert
  python train/train.py --model roberta-base            --epochs 2 --lr 3e-5 --run_name roberta

Outputs:
  models/<run_name>/            weights + tokenizer + qa_config.json   (NOT committed)
  logs/<run_name>/train_log.csv, val_log.csv, train_args.json          (committed: they are deliverables)
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from qa.features import make_train_features  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id of a BASE checkpoint, e.g. roberta-base")
    p.add_argument("--run_name", default=None, help="folder name under models/ and logs/")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1, help="use 2 + half batch_size if GPU runs out of memory")
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_train_samples", type=int, default=0, help="0 = full train set")
    p.add_argument("--val_samples", type=int, default=2000, help="dev questions used for validation loss")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_amp", action="store_true", help="disable fp16 (try this if loss becomes NaN)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(raw, tokenizer, args, batch_size, shuffle):
    feats = raw.map(
        lambda ex: make_train_features(ex, tokenizer, args.max_seq_length, args.doc_stride),
        batched=True,
        remove_columns=raw.column_names,
        desc="tokenizing",
    )
    feats.set_format("torch")
    return DataLoader(feats, batch_size=batch_size, shuffle=shuffle, collate_fn=default_data_collator)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item() * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    model.train()
    return total / max(n, 1)


def main():
    args = parse_args()
    run = args.run_name or args.model.split("/")[-1]
    out_dir, log_dir = ROOT / "models" / run, ROOT / "logs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    print(f"device={device} amp={use_amp}")

    raw = load_dataset("rajpurkar/squad_v2")
    train_raw, val_raw = raw["train"], raw["validation"]
    if args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.max_train_samples))
    val_raw = val_raw.shuffle(seed=args.seed).select(range(min(args.val_samples, len(val_raw))))
    print(f"train questions={len(train_raw)}  validation questions={len(val_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_loader = build_loader(train_raw, tokenizer, args, args.batch_size, shuffle=True)
    val_loader = build_loader(val_raw, tokenizer, args, args.batch_size * 2, shuffle=False)
    print(f"train windows={len(train_loader.dataset)}  validation windows={len(val_loader.dataset)}")

    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    model.train()

    no_decay = ("bias", "LayerNorm", "layer_norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_csv = open(log_dir / "train_log.csv", "w", newline="")
    val_csv = open(log_dir / "val_log.csv", "w", newline="")
    tw, vw = csv.writer(train_csv), csv.writer(val_csv)
    tw.writerow(["step", "epoch", "loss", "lr", "elapsed_s"])
    vw.writerow(["step", "epoch", "val_loss"])

    t0, step, running, running_n, done = time.time(), 0, 0.0, 0, False
    epoch_idx = 0
    while not done:
        for i, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch_idx + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()
            running_n += 1

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    tw.writerow([step, round(step / steps_per_epoch, 4), round(running / running_n, 5),
                                 scheduler.get_last_lr()[0], round(time.time() - t0, 1)])
                    train_csv.flush()
                    running, running_n = 0.0, 0
                if step >= total_steps:
                    done = True
                    break
        epoch_idx += 1
        vl = validation_loss(model, val_loader, device)
        vw.writerow([step, round(step / steps_per_epoch, 4), round(vl, 5)])
        val_csv.flush()
        print(f"epoch {epoch_idx} done | step {step}/{total_steps} | val_loss {vl:.4f}")

    train_csv.close()
    val_csv.close()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    cfg = {
        "base_model": args.model,
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride,
        "threshold": 0.0,   # tune_threshold.py rewrites this
        "params": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "qa_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved model to {out_dir}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
