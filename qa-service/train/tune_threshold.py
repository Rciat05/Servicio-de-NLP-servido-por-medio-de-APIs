"""Choose the no-answer threshold from the output of evaluate.py.

Rule: answer is null when  diff = null_score - best_span_score  >  threshold.
  * threshold too LOW  -> says "no answer" too often (you lose answerable questions)
  * threshold too HIGH -> always forces an answer (you lose unanswerable questions)

IMPORTANT: SQuAD 2.0 dev is ~50% unanswerable but the hidden test is ~25%.
So we re-weight the questions so that unanswerable ones weigh 25% in total,
tune on one half of the data and report on the other half (no fooling ourselves).

  python train/tune_threshold.py --analysis logs/roberta/dev_analysis.jsonl --write_config models/roberta/qa_config.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from qa.metrics import score_item  # noqa: E402


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    d = {
        "diff": np.array([r["diff"] for r in rows], dtype=float),
        "unans": np.array([len(r["golds"]) == 0 for r in rows]),
    }
    em_t, f1_t, em_n, f1_n = [], [], [], []
    for r in rows:
        a, b = score_item(r["golds"], r["pred_text"])     # if we answer with the span
        c, e = score_item(r["golds"], None)                # if we answer null
        em_t.append(a); f1_t.append(b); em_n.append(c); f1_n.append(e)
    d.update(em_t=np.array(em_t), f1_t=np.array(f1_t), em_n=np.array(em_n), f1_n=np.array(f1_n))
    return d


def weights(unans, frac):
    w = np.zeros(len(unans))
    na, nu = (~unans).sum(), unans.sum()
    if na:
        w[~unans] = (1 - frac) / na
    if nu:
        w[unans] = frac / nu
    return w


def evaluate(d, mask, thr, frac):
    sub = {k: v[mask] for k, v in d.items()}
    is_null = sub["diff"] > thr
    em = np.where(is_null, sub["em_n"], sub["em_t"])
    f1 = np.where(is_null, sub["f1_n"], sub["f1_t"])
    w = weights(sub["unans"], frac)
    return {"EM": float((w * em).sum()), "F1": float((w * f1).sum()),
            "EM_answerable": float(em[~sub["unans"]].mean()) if (~sub["unans"]).any() else float("nan"),
            "null_acc_unanswerable": float(em[sub["unans"]].mean()) if sub["unans"].any() else float("nan")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--unans_frac", type=float, default=0.25)
    p.add_argument("--write_config", default=None, help="path to qa_config.json to update with the best threshold")
    args = p.parse_args()

    d = load(args.analysis)
    n = len(d["diff"])
    tune = np.arange(n) % 2 == 0
    hold = ~tune
    grid = np.arange(-10, 10.01, 0.25)

    sweep = [(t, evaluate(d, tune, t, args.unans_frac)) for t in grid]
    best_t, _ = max(sweep, key=lambda x: x[1]["F1"])
    print(f"questions: {n} ({int(d['unans'].sum())} unanswerable)  weighting: {args.unans_frac:.0%} unanswerable\n")
    print(f"{'threshold':>9} | {'EM':>6} {'F1':>6} {'EM answerable':>14} {'null acc':>9}   (tuning half)")
    for t, m in sweep[::8]:
        print(f"{t:9.2f} | {m['EM']:6.3f} {m['F1']:6.3f} {m['EM_answerable']:14.3f} {m['null_acc_unanswerable']:9.3f}")

    print(f"\nBEST threshold on tuning half: {best_t:.2f}")
    for name, mask in [("tuning half ", tune), ("HOLD-OUT half", hold)]:
        m = evaluate(d, mask, best_t, args.unans_frac)
        print(f"{name}: EM={m['EM']:.3f} F1={m['F1']:.3f} | answerable EM={m['EM_answerable']:.3f} | null acc={m['null_acc_unanswerable']:.3f}")
    m0 = evaluate(d, hold, 0.0, args.unans_frac)
    print(f"hold-out with default threshold 0.00: EM={m0['EM']:.3f} F1={m0['F1']:.3f}")

    out_csv = Path(args.analysis).with_name("threshold_sweep.csv")
    with open(out_csv, "w") as f:
        f.write("threshold,EM,F1,EM_answerable,null_acc_unanswerable\n")
        for t, m in sweep:
            f.write(f"{t},{m['EM']},{m['F1']},{m['EM_answerable']},{m['null_acc_unanswerable']}\n")
    print(f"full sweep saved to {out_csv}")

    if args.write_config:
        cfg_path = Path(args.write_config)
        cfg = json.loads(cfg_path.read_text())
        cfg["threshold"] = float(best_t)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"updated {cfg_path} -> threshold = {best_t:.2f}")


if __name__ == "__main__":
    main()
