"""Report metrics at a FIXED threshold over ALL items of an analysis file. No tuning, no writing.

Use it to measure how the model behaves on a new set (e.g. our course-readings mini-set) with the
threshold we already chose on SQuAD.

  python train/report_at_threshold.py --analysis logs/distilbert/team_miniset_analysis.jsonl --threshold 3.25 --file data/team_miniset.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qa.metrics import score_item  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--threshold", type=float, default=3.25)
    p.add_argument("--file", default=None, help="original JSON (public_sample format) to print the questions")
    p.add_argument("--unans_frac", type=float, default=0.25)
    a = p.parse_args()

    questions = {}
    if a.file:
        data = json.loads(Path(a.file).read_text(encoding="utf-8"))
        items = data["items"] if isinstance(data, dict) else data
        questions = {it["id"]: it["question"] for it in items}

    rows = [json.loads(line) for line in open(a.analysis, encoding="utf-8")]
    res = []
    for r in rows:
        pred = None if (r["diff"] > a.threshold or r["pred_text"] is None) else r["pred_text"]
        em, f1 = score_item(r["golds"], pred)
        res.append({**r, "pred": pred, "em": em, "f1": f1, "unans": len(r["golds"]) == 0})

    ans = [x for x in res if not x["unans"]]
    una = [x for x in res if x["unans"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    em_ans, f1_ans, null_acc = mean([x["em"] for x in ans]), mean([x["f1"] for x in ans]), mean([x["em"] for x in una])
    w_em = (1 - a.unans_frac) * em_ans + a.unans_frac * null_acc          # same weighting as tune_threshold.py
    w_f1 = (1 - a.unans_frac) * f1_ans + a.unans_frac * null_acc

    print(f"threshold = {a.threshold}   items = {len(res)} ({len(ans)} answerable, {len(una)} unanswerable)\n")
    print(f"EM overall (what the live score resembles) : {mean([x['em'] for x in res]):.3f}")
    print(f"EM / F1 reweighted to {a.unans_frac:.0%} unanswerable : {w_em:.3f} / {w_f1:.3f}")
    print(f"EM on answerable questions                 : {em_ans:.3f}  ({sum(x['em'] for x in ans)}/{len(ans)})")
    print(f"F1 on answerable questions                 : {f1_ans:.3f}")
    print(f"correct null on unanswerable questions     : {null_acc:.3f}  ({sum(x['em'] for x in una)}/{len(una)})")
    print("\nReference (SQuAD hold-out, same threshold): answerable EM 0.748, correct null 0.416, EM 0.665, F1 0.720")

    wrong = [x for x in res if x["em"] == 0]
    print(f"\n--- {len(wrong)} items without exact match ---")
    for x in wrong:
        kind = "UNANSWERABLE" if x["unans"] else "answerable"
        print(f"[{x['id']}] {kind} | diff={x['diff']:.2f} (null if > {a.threshold})")
        if x["id"] in questions:
            print(f"    Q: {questions[x['id']]}")
        print(f"    expected: {x['golds'] or 'null'}")
        print(f"    model   : {x['pred']!r}   (best span: {x['pred_text']!r})")


if __name__ == "__main__":
    main()
