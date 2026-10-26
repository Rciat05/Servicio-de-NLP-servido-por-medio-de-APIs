"""Runs the 5 curl tests from the brief + the public sample set, and measures latency.
Works the same on Windows / Linux / macOS (no curl quoting problems).

  python scripts/check_service.py --url http://127.0.0.1:8000
  python scripts/check_service.py --url https://<your-tunnel-url>          # the dry run MUST be done through the tunnel

The token is read from the QA_API_TOKEN environment variable (or --token).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qa.metrics import score_item  # noqa: E402

FAILS = []


def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--token", default=os.environ.get("QA_API_TOKEN", ""))
    p.add_argument("--sample", default=str(ROOT / "data" / "public_sample.json"))
    a = p.parse_args()
    url, auth = a.url.rstrip("/"), {"Authorization": f"Bearer {a.token}"}
    if not a.token:
        sys.exit("Set QA_API_TOKEN (or pass --token) first.")

    print("== access control and metadata ==")
    r = requests.get(f"{url}/health", timeout=10)
    check("1 /health, no token -> 200 ok", r.status_code == 200 and r.json().get("status") == "ok", f"({r.status_code})")
    r = requests.get(f"{url}/info", timeout=10)
    check("2 /info, no token -> 401", r.status_code == 401 and r.json().get("error") == "unauthorized", f"({r.status_code})")
    r = requests.get(f"{url}/info", headers={"Authorization": "Bearer wrong"}, timeout=10)
    check("3 /info, wrong token -> 401", r.status_code == 401, f"({r.status_code})")
    r = requests.get(f"{url}/info", headers=auth, timeout=10)
    body = r.json() if r.status_code == 200 else {}
    check("4 /info, valid token -> 200 with team/model/params",
          r.status_code == 200 and all(k in body for k in ("team", "model", "params")), json.dumps(body))
    r = requests.post(f"{url}/predict", json={"id": "x", "context": "a", "question": "b"}, timeout=10)
    check("  /predict, no token -> 401", r.status_code == 401, f"({r.status_code})")
    r = requests.post(f"{url}/predict", headers=auth, json={"id": "x", "context": "a"}, timeout=10)
    check("  /predict, missing field -> 400/422 + error body",
          r.status_code in (400, 422) and "error" in r.json(), f"({r.status_code})")

    print("\n== public sample set ==")
    items = json.loads(Path(a.sample).read_text(encoding="utf-8"))["items"]
    latencies, total_score = [], 0.0
    for it in items:
        t0 = time.perf_counter()
        r = requests.post(f"{url}/predict", headers=auth, timeout=5,
                          json={"id": it["id"], "context": it["context"], "question": it["question"]})
        dt = time.perf_counter() - t0
        latencies.append(dt)
        ok_shape = r.status_code == 200
        res = r.json() if ok_shape else {}
        ans, st = res.get("answer"), res.get("start")
        offsets_ok = (ans is None and st is None) or (
            ans is not None and st is not None and it["context"][st:st + len(ans)] == ans)
        golds = [x["text"] for x in it["answers"]]
        em, f1 = score_item(golds, ans)
        total_score += em
        check(f"{it['id']} shape/offsets/id", ok_shape and offsets_ok and res.get("id") == it["id"],
              f"-> {ans!r} start={st} | expected {golds or 'null'} | EM={em} F1={f1:.2f} | {dt:.2f}s")

    mean = sum(latencies) / len(latencies)
    print(f"\nlatency per request (round trip): mean {mean:.2f}s, max {max(latencies):.2f}s")
    print(f"projection for 50 items: {mean * 50:.0f}s  (global cap: 150s, per-request cap: 5s)")
    print(f"exact match on the sample: {total_score:.0f}/{len(items)}")
    if mean * 50 > 150 or max(latencies) > 5:
        FAILS.append("latency")
        print("[FAIL] too slow for the time caps")
    print("\nALL GOOD" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
