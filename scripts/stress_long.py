import json, os, sys, time, requests

url = sys.argv[1].rstrip("/")
tok = os.environ["QA_API_TOKEN"]
it = json.load(open("data/public_sample.json", encoding="utf-8"))["items"][4]
ctx = it["context"]
while len(ctx) < 5000:
    ctx += " " + it["context"]
ctx = ctx[:5000]
for _ in range(3):
    t = time.perf_counter()
    r = requests.post(url + "/predict", headers={"Authorization": "Bearer " + tok},
                      json={"id": "long", "context": ctx, "question": it["question"]}, timeout=20)
    print(r.status_code, f"{time.perf_counter() - t:.2f}s", len(ctx), "chars")
    