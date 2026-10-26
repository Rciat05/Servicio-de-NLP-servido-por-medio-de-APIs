"""Make logs/<run>/loss.png from the CSV files written by train.py.

  python scripts/plot_loss.py logs/roberta
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    run_dir = Path(sys.argv[1])
    train, val = read(run_dir / "train_log.csv"), read(run_dir / "val_log.csv")
    plt.figure(figsize=(7, 4))
    plt.plot([float(r["epoch"]) for r in train], [float(r["loss"]) for r in train], label="train loss")
    plt.plot([float(r["epoch"]) for r in val], [float(r["val_loss"]) for r in val], "o-", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(run_dir.name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = run_dir / "loss.png"
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
