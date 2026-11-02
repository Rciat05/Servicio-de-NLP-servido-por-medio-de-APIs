# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | Robertito|
| Members | Roberto Carlos Iglesias Álvarez, Andres Arturo Marroquin Soto, Cristian Jafet Rosales Martinez|
| Repository URL | https://github.com/Rciat05/Servicio-de-NLP-servido-por-medio-de-APIs |
| `submission` commit hash | 29885fe |
| Public service URL used at the dry run | https://judge-habitat-unwed.ngrok-free.dev (ngrok free static domain, confirmed unchanged during dry run) |
| AI tools used | Claude (Anthropic): generated the initial project skeleton (FastAPI service, training/evaluation scripts, tests) and guided setup; we ran, tested and reviewed everything. Cristian Jafet used Claude (chat and Claude Code): it wrote the mini-set questions from the course readings and their validation script, drafted sections 8 and 9 from the measured results, and helped prepare the Q&A and the pitch; he ran the remote tests and made the commits. Andres Marroquin used Gemini: guided the step-by-step RoBERTa fine-tuning process on Kaggle, provided guidance on Git commands, and assisted in formatting and completing REPORT.md. Antigravity was used to audit the repository to ensure no sensitive data or large files were tracked and to verify the final code structure before submission. |

## 1. Model choice and justification

- **Base checkpoint:** `roberta-base` (≈125 M parameters).
- **Alternatives considered:** `distilbert-base-uncased` (66.4 M parameters, lower latency but lower accuracy, which was our initial choice); `bert-base-uncased` (≈110 M, dominated by RoBERTa); `deberta-v3-base` (≈184 M, best accuracy but the slowest on CPU and more fragile to fine-tune); `ModernBERT-base` (≈149 M, little published evidence on SQuAD 2.0). 
- **Trade-off:** the service runs on a CPU-only laptop under a 5 s per-request cap and a 150 s cap for 50 items (about 3 s per item). RoBERTa gave a mean latency of **0.29 s** locally, projecting **15 s** for 50 items — fitting comfortably within both caps (10× margin on the 150 s global cap). The cost of using RoBERTa over DistilBERT is increased latency and size, but the benefit is higher accuracy on out-of-domain texts like the hidden test.

## 2. Training

| Item | Value |
|---|---|
| Training data | Full SQuAD 2.0 train: 130,319 questions → 131,751 windows |
| Validation data | 2,000 random SQuAD 2.0 dev questions (2,036 windows) for validation loss; 3,000 random dev questions (1,519 unanswerable) for EM/F1 and threshold tuning |
| Epochs | 2 |
| Learning rate and schedule | 3e-5, AdamW, weight decay 0.01, linear decay with 10 % warmup, gradient clipping 1.0, fp16 |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | Kaggle, Tesla T4 (one GPU used); ≈ 91.7 min total (≈ 45 min per epoch) |

- **Loss curve:** [`logs/roberta/loss.png`](logs/roberta/loss.png); raw values in `logs/roberta/train_log.csv` and `val_log.csv`. Validation loss was 0.8185 after epoch 1 and rose slightly to 0.8377 after epoch 2.
- **Validation results** (hold-out half of the 3,000 questions, reweighted to 25 % unanswerable like the hidden test, threshold 2.00): EM 0.804, F1 0.849; EM on answerable questions 0.812; correct `null` on unanswerable questions 0.778. Full table: `logs/roberta/threshold_sweep.csv`. 

## 3. No-answer threshold

- **Decision rule:** for each window, `null_score = start_logit[CLS] + end_logit[CLS]` and `span_score` is the best `start_logit[i] + end_logit[j]` over context tokens. Over all windows we keep the best span and the lowest `null_score`. The answer is `null` when `null_score − span_score > threshold`.
- **Choice:** `train/tune_threshold.py` sweeps −10 to 10 in steps of 0.25. SQuAD dev is about 50 % unanswerable but the hidden test is about 25 %, so questions are reweighted to 25 % unanswerable. We tune on one half and report on the other half. Final value: **2.00** (qa_config.json).
- **Too low / too high** (tuning half): at −10 the service says "no answer" almost always (null accuracy 0.987, answerable EM 0.335, F1 0.509). At +10 it always answers (answerable EM 0.826, null accuracy 0.442, F1 0.786). The best threshold balances the trade-off at 2.00 (tuning half F1 0.834).

## 4. Serving architecture

- **Framework:** FastAPI + uvicorn, 1 worker, listening on `127.0.0.1:8000`; only the tunnel is exposed.
- **Model loading:** a background thread loads the model at startup and runs one warm-up inference. `/health` returns 503 `loading` until then.
- **Request flow** (`src/qa/server.py`): check token → parse JSON → validate with pydantic (`extra=forbid`) → one request at a time under a lock → tokenize question and context into windows of 384 tokens with stride 128 (long contexts become several windows, padded only to the longest) → forward pass → best span over all windows and minimum `null_score` → apply threshold → `answer = context[start:end]` (whitespace trimmed, `start` adjusted). A final check enforces `context[start:start+len(answer)] == answer`, otherwise it returns `null`.
- **Access control:** `_check_auth` in `src/qa/server.py` requires `Authorization: Bearer <token>`, compared in constant time (`hmac.compare_digest`) with `QA_API_TOKEN`, which is read from the environment (or an untracked `.env`). If the variable is empty, protected endpoints always return 401.
- **Tunnel:** ngrok (primary, free static domain: `ngrok http --url=judge-habitat-unwed.ngrok-free.dev 8000`); Cloudflare tunnel as tested backup (`cloudflared tunnel --url http://localhost:8000`, providing a fallback `*.trycloudflare.com` URL if ngrok drops or hits bandwidth limits).
- **Weights:**  https://drive.google.com/drive/folders/18qtA5rVOHLhKIKbyIVfrOYV7qDH0aViF?usp=drive_link.
- **How to run** from a clean machine:

```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# put the downloaded weights in models/roberta
$env:QA_API_TOKEN = "<token>" ; $env:QA_TEAM = "<team-id>" ; $env:MODEL_DIR = "models/roberta"
python scripts/run_server.py
python scripts/check_service.py --url http://127.0.0.1:8000   # contract + sample + latency
```

## 5. Results on the public sample set

Results measured with `check_service.py` against `http://127.0.0.1:8000` running `roberta-base` (threshold 2.00), on 2026-11-01.

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | — | Yes |
| sample-04 | `null` | `3` | 310 | No |
| sample-05 | `50,257` | `50,257` | 3157 | Yes |

**4/5 correct.** The only miss is **sample-04** (false positive): the question asks by how many times LoRA reduces GPU memory *when fine-tuning BERT-base*, which is never stated — the context only mentions the comparison against GPT-3 175B. The model extracted `3` (the correct reduction factor for GPT-3, present in the context) with a `diff` score of −4.11, well below the threshold of 2.00, so the no-answer signal was not strong enough. This is a classic near-answerable failure: the number exists in the context but belongs to a different comparison than the one asked.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | ≈125,000,000 |
| Size on disk | 499 MB (`model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | **0.29 s** mean, 0.95 s max (5 requests, `check_service.py`, 2026-11-01); projection for 50 items: **15 s** (cap: 150 s) |
| Laptop CPU and RAM | Intel Core i7-6700HQ (4 cores / 8 threads), 32 GB RAM, Windows 10 |
| Mean latency from another network (ngrok tunnel, Sun 01 Nov 2026, 21:46 El Salvador time) | **1.84 s** mean, 2.74 s max (5 requests, `check_service.py` via ngrok dry run, archived in `docs/dry_run/`); projection for 50 items: **92 s** (cap: 150 s, max per-request cap: 5 s) |

## 7. Problems encountered and how you fixed them

1. **Problem:** the repository had the whole project inside a `qa-service/` subfolder, but the brief requires `REPORT.md` at the repository root. **Cause:** the project folder was copied into the repo as a subfolder. **Fix:** moved everything to the root. **Commit:** `ff3a76f`
2. **Problem:** the service gave no sign that the model had loaded. **Cause:** uvicorn's default logging hides the INFO message `model loaded, service ready`, so the only symptom of a failed start was a 503 on `/health`. **Fix:** `logging.basicConfig(level=logging.INFO)` in `scripts/run_server.py`. **Commit:** `e5d0307`
3. **Problem:** Underperformance on out-of-domain text. **Cause:** We originally chose DistilBERT (66M params) for strict latency adherence, but it struggled significantly with context understanding outside of SQuAD. **Fix:** We switched to `roberta-base` (125M params) and retrained the model, accepting the latency trade-off to ensure better Exact Match performance. **Commit:** `9e28b4f` 

## 8. Decoder investigation

*Based on sample-04 and our 44-question mini-set (RoBERTa, threshold 2.00). What we say about decoders is a hypothesis; we did not run one.*

- **Near-answerable questions (sample-04; mini-set).** The number is in the context but belongs to something else. An encoder can only pick a span or apply one global threshold; a decoder could notice that the figure belongs to another method. Cost: token generation on a laptop CPU is slow and threatens the 5 s cap; such models are far bigger than our ~500 MB model; free text cannot be checked against the verbatim-span contract.
- **Wrong entity on answerable questions.** Asked for one specific metric, the model returns another nearby metric. Same cause, fix and cost.
- **Boundary mismatches.** `GLUE` versus the reference `GLUE benchmark`. A decoder would not help: free generation makes exact match harder. The live LLM judge is the realistic mitigation.
- **Long contexts (sample-05).** Windows are scored independently, so evidence split across windows is never combined. 
- **Non-contiguous answers and yes/no questions.** An extractive head cannot produce them; an encoder–decoder (T5-style) could. Cost: the contract requires a verbatim span with a character `start`, so a generated answer must be mapped back to the context, and free text rarely matches Exact Match.

## 9. Known limitations

- English only; contexts up to 5,000 characters and questions up to 500 (larger gets 400). One contiguous span of at most 30 tokens: no yes/no, multi-span, counting or multi-step answers.
- **Quality outside SQuAD (measured).** On our 44-question mini-set on the course readings (32 answerable, 12 unanswerable; threshold 2.00): EM on answerable questions **0.750** (24/32), correct `null` on unanswerable **0.833** (10/12), reweighted EM (75 % answerable + 25 % unanswerable) **0.771**, mean F1 **0.860**. The SQuAD hold-out was EM 0.804 / F1 0.849, so the drop on out-of-domain text is modest (−0.033 EM, +0.011 F1). Main error modes: boundary mismatches (`GLUE` vs. `GLUE benchmark`, `18 million` vs. `18 million parameters`), one wrong entity (`1.2 TB` instead of `350 GB` for mini-lo-05), and two false positives where the model answered an unanswerable question with a plausible-looking span (mini-at-03, mini-lo-14). With 44 items each unanswerable example is about 8 percentage points of null accuracy, so these numbers are indicative rather than definitive.
- One request at a time, one worker: concurrent clients queue up.
- Depends on one laptop and a free ngrok tunnel (20,000 requests and 1 GB per month): if either drops, every request fails.
- `start` is a Python string index (code points).