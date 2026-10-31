# Project Report — Extractive QA as a Service

<!--
╔══════════════════════════════════════════════════════════════════════════════╗
║  PENDIENTES DEL EQUIPO  (borrar TODO este bloque y todos los "TODO" antes   ║
║  de crear el tag `submission`; un TODO que quede = sección incompleta)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Este borrador ya trae los datos reales de DistilBERT (entrenamiento, umbral, sample público,
latencia). Lo que falta está marcado como  **TODO (quién):**  dentro del documento.

ROBERTO (servicio, túnel, reporte)
 [ ] Pedir al instructor: token de desarrollo y nombre de equipo (team id). Poner QA_TEAM en el .env
     y actualizar la sección 0.
 [ ] Enviar al instructor la URL del túnel antes del dry run y preguntar la fecha/hora del dry run.
 [ ] Hacer commit del cambio de logging en scripts/run_server.py (ver Problema 2, sección 7) y copiar
     su hash (git log --oneline).
 [ ] Subir los pesos (models/distilbert, ~223 MB zip) a un GitHub Release o carpeta compartida y poner
     el enlace en la sección 4 y en el README. Los pesos NO van en git.
 [ ] Instalar y probar el túnel de respaldo (Cloudflare) y anotar el comando en la sección 4.
 [ ] Verificar antes del tag:  git ls-files  (sin models/ ni .env)  y  git log --all -S"<token>" --oneline
     (no debe imprimir nada).

COMPAÑERO 2 (RoBERTa)
 [ ] Entrenar roberta-base en Kaggle con el notebook y comandos acordados (ver GUIA_PASOS.md, Fase 5).
 [ ] Commit de logs/roberta/ con su propio usuario de Git (nunca models/).
 [ ] Pasar su zip a Roberto para medir la latencia de RoBERTa en la laptop de servicio.
 [ ] Rellenar las filas "RoBERTa" de las secciones 1 y 2 y decidir el modelo final con Roberto
     (criterio: F1 del hold-out + latencia por el túnel). Si gana RoBERTa, actualizar secciones 1–6.

COMPAÑERO 3 (pruebas externas, mini-set, decoder)
 [ ] Probar la URL pública desde OTRA red (casa / datos móviles) con scripts/check_service.py y anotar
     red, hora, latencia media y máxima (sección 6).
 [ ] Escribir un mini-set de 30–50 preguntas en inglés sobre las lecturas del curso (BERT, Transformer,
     GPT-1, LoRA, SQuAD 2.0), ~25–30 % sin respuesta, SIN mirar lo que responde el modelo. Guardarlo en
     data/team_miniset.json con el formato de data/public_sample.json. Evaluarlo con
     train/evaluate.py --source file y comparar el resultado con el hold-out de SQuAD (sección 9).
 [ ] Reemplazar el borrador de la sección 8 con los errores reales que encuentre en el mini-set
     (leer Radford 2018 / GPT-1 y, opcional, Hu 2021 / LoRA).

TODOS
 [ ] Nombres de los integrantes y herramientas de IA usadas por cada uno (sección 0). Confirmar con el
     instructor el tamaño del grupo (la guía dice 4–5 y somos 3).
 [ ] Leer el código para el Q&A individual (tabla al final de GUIA_PASOS.md). No se puede usar IA ahí.
 [ ] Preparar el pitch de 3 min: modelo + trade-off CPU, entrenamiento (datos, hiperparámetros, umbral,
     curva), serving (framework, carga del modelo, flujo, token) y decoder atado a NUESTROS problemas.
 [ ] Revisar el largo (~1000 palabras sin tablas), el inglés, y que NINGUNA sección quede vacía.
 [ ] git tag submission && git push origin submission  (con este REPORT.md ya dentro del commit).
 [ ] Día de la presentación: token nuevo en QA_API_TOKEN, reiniciar servidor, abrir túnel, correr
     check_service.py, enviar la URL si cambió, laptop enchufada y sin suspensión.
-->

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-XX` — **TODO (Roberto):** registered name from the instructor |
| Members | Roberto Iglesias, **TODO:** Andrés' full name, Cristian Jafet Rosales Martínez |
| Repository URL | https://github.com/Rciat05/Servicio-de-NLP-servido-por-medio-de-APIs |
| `submission` commit hash | **TODO:** a commit cannot contain its own hash; write the hash of the parent commit or "see tag `submission`" |
| Public service URL used at the dry run | https://judge-habitat-unwed.ngrok-free.dev (ngrok free static domain) — **TODO (Roberto):** confirm it is unchanged at the dry run |
| AI tools used (which, and for what) | Claude (Anthropic): generated the initial project skeleton (FastAPI service, training/evaluation scripts, tests) and guided setup; we ran, tested and reviewed everything. Cristian Jafet used Claude (chat and Claude Code): it wrote the mini-set questions from the course readings and their validation script, drafted sections 8 and 9 from the measured results, and helped prepare the Q&A and the pitch; he ran the remote tests and made the commits. **TODO (Roberto, Andrés):** add your tools. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert-base-uncased` (66.4 M parameters).
- **Alternatives considered:** `roberta-base` (≈125 M parameters, higher accuracy in the literature at about twice the compute); `bert-base-uncased` (≈110 M, dominated by RoBERTa); `deberta-v3-base` (≈184 M, best accuracy but the slowest on CPU and more fragile to fine-tune); `ModernBERT-base` (≈149 M, little published evidence on SQuAD 2.0). **TODO (Compañero 2):** add the measured RoBERTa result (hold-out F1 and latency) and state the final decision.
- **Trade-off:** the service runs on a CPU-only laptop under a 5 s per-request cap and a 150 s cap for 50 items (about 3 s per item). DistilBERT gave a mean latency of 0.32 s locally and 1.36 s through the tunnel, so the 50 items are projected at about 68 s, with room to spare. Its cost is lower accuracy than larger encoders, which matters because the hidden test uses texts outside SQuAD.

## 2. Training

| Item | Value |
|---|---|
| Training data | Full SQuAD 2.0 train: 130,319 questions → 131,692 windows |
| Validation data | 2,000 random SQuAD 2.0 dev questions (2,033 windows) for validation loss; 3,000 random dev questions (1,519 unanswerable) for EM/F1 and threshold tuning |
| Epochs | 3 |
| Learning rate and schedule | 3e-5, AdamW, weight decay 0.01, linear decay with 10 % warmup, gradient clipping 1.0, fp16 |
| Batch size | 32 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | Kaggle, Tesla T4 (one GPU used); ≈ 58 min total (≈ 19 min per epoch) |

- **Loss curve:** [`logs/distilbert/loss.png`](logs/distilbert/loss.png); raw values in `logs/distilbert/train_log.csv` and `val_log.csv`. Training loss falls from about 5.8 to about 0.8. Validation loss was 1.43, 1.26 and 1.49 after epochs 1–3, so epoch 3 shows some overfitting.
- **Validation results** (hold-out half of the 3,000 questions, reweighted to 25 % unanswerable like the hidden test, threshold 3.25): EM 0.665, F1 0.720; EM on answerable questions 0.748; correct `null` on unanswerable questions 0.416. Full table: `logs/distilbert/threshold_sweep.csv`. **TODO (Compañero 2):** add the RoBERTa row.

## 3. No-answer threshold

- **Decision rule:** for each window, `null_score = start_logit[CLS] + end_logit[CLS]` and `span_score` is the best `start_logit[i] + end_logit[j]` over context tokens. Over all windows we keep the best span and the lowest `null_score`. The answer is `null` when `null_score − span_score > threshold`.
- **Choice:** `train/tune_threshold.py` sweeps −10 to 10 in steps of 0.25. SQuAD dev is about 50 % unanswerable but the hidden test is about 25 %, so questions are reweighted to 25 % unanswerable. We tune on one half and report on the other half. Final value: **3.25** (qa_config.json).
- **Too low / too high** (tuning half): at −10 the service says "no answer" almost always (null accuracy 0.93, answerable EM 0.19, F1 0.38). At +10 it always answers (answerable EM 0.75, null accuracy 0.13, F1 0.66). F1 is flat between 0 and 6 (0.69–0.70).

## 4. Serving architecture

- **Framework:** FastAPI + uvicorn, 1 worker, listening on `127.0.0.1:8000`; only the tunnel is exposed.
- **Model loading:** a background thread loads the model at startup and runs one warm-up inference. `/health` returns 503 `loading` until then.
- **Request flow** (`src/qa/server.py`): check token → parse JSON → validate with pydantic (`extra=forbid`) → one request at a time under a lock → tokenize question and context into windows of 384 tokens with stride 128 (long contexts become several windows, padded only to the longest) → forward pass → best span over all windows and minimum `null_score` → apply threshold → `answer = context[start:end]` (whitespace trimmed, `start` adjusted). A final check enforces `context[start:start+len(answer)] == answer`, otherwise it returns `null`.
- **Access control:** `_check_auth` in `src/qa/server.py` requires `Authorization: Bearer <token>`, compared in constant time (`hmac.compare_digest`) with `QA_API_TOKEN`, which is read from the environment (or an untracked `.env`). If the variable is empty, protected endpoints always return 401.
- **Tunnel:** ngrok, `ngrok http 8000` (free static domain). **TODO (Roberto):** backup tunnel (Cloudflare: `cloudflared tunnel --url http://localhost:8000`) after testing it.
- **Weights:** **TODO (Roberto):** link to the Release / shared folder with `models/distilbert`.
- **How to run** from a clean machine:

```powershell
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# put the downloaded weights in models/distilbert
$env:QA_API_TOKEN = "<token>" ; $env:QA_TEAM = "<team-id>" ; $env:MODEL_DIR = "models/distilbert"
python scripts/run_server.py
python scripts/check_service.py --url http://127.0.0.1:8000   # contract + sample + latency
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3` | 310 | **No** |
| sample-05 | `50,257` | `50,257` | 3157 | Yes (3,171-character context, answer at its very end) |

sample-04 asks how much LoRA reduces GPU memory for BERT-base, but the context only gives that figure (3×) for GPT-3 175B. The model took the nearby number as the answer: the question and the context share many words, and our global threshold, tuned on Wikipedia, did not abstain. 4 of 5 items are correct.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 265 MB (`model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | 0.32 s on localhost (max 1.13 s, 3,171-character context); 1.36 s through the ngrok tunnel (max 1.85 s) → about 68 s projected for 50 items |
| Laptop CPU and RAM | Intel Core i7-6700HQ (4 cores / 8 threads), 32 GB RAM, Windows 10 |
| Mean latency from another network (home Wi-Fi, Sat 31 Oct 2026, 10:44 El Salvador time) | 1.36 s mean, 2.01 s max per request through the ngrok tunnel; 68 s projected for 50 items; 7 of 7 contract checks passed, public sample 4/5 |

**TODO (Compañero 2 / Roberto):** add RoBERTa latency if it is evaluated.

## 7. Problems encountered and how you fixed them

1. **Problem:** the repository had the whole project inside a `qa-service/` subfolder, but the brief requires `REPORT.md` at the repository root. **Cause:** the project folder was copied into the repo as a subfolder. **Fix:** moved everything to the root. **Commit:** `ff3a76f`
2. **Problem:** the service gave no sign that the model had loaded. **Cause:** uvicorn's default logging hides the INFO message `model loaded, service ready`, so the only symptom of a failed start was a 503 on `/health`. **Fix:** `logging.basicConfig(level=logging.INFO)` in `scripts/run_server.py`. **Commit:** `e5d0307`
3. **TODO (all):** add real problems as they appear (for example RoBERTa training, tunnel issues, mini-set findings), each with its cause, fix and commit.

## 8. Decoder investigation

*Based on sample-04 and our 44-question mini-set (DistilBERT, threshold 3.25). What we say about decoders is a hypothesis; we did not run one.*

- **Near-answerable questions (sample-04; mini-set: 6 of 8 wrong).** The number is in the context but belongs to something else: asked how many trainable parameters LoRA needs per task, the encoder returned `175 billion` (the full fine-tuning figure) with `diff` = −10.0, far from the threshold. An encoder can only pick a span or apply one global threshold; a decoder could notice that the figure belongs to another method. Cost: token generation on a laptop CPU is slow (not measured by us) and threatens the 5 s cap; such models are far bigger than our 265 MB model; free text cannot be checked against the verbatim-span contract.
- **Wrong entity on answerable questions (5 of 13 misses).** Asked for the English-to-French BLEU, the model returned the English-to-German score (28.4). Same cause, fix and cost.
- **Boundary mismatches (8 of 13 misses).** `GLUE` versus the reference `GLUE benchmark`. A decoder would not help: free generation makes exact match harder. The live LLM judge is the realistic mitigation.
- **Long contexts (sample-05; 4 long mini-set items).** Windows are scored independently, so evidence split across windows is never combined. None of the four long items was an exact match (two boundary mismatches, one wrong answer, and an answer given to the unanswerable item): too few to conclude. GPT-1 was pre-trained on 512-token sequences (Radford et al., 2018), so a model of that kind has a similar window limit.
- **Non-contiguous answers and yes/no questions.** An extractive head cannot produce them; an encoder–decoder (T5-style) could. Cost: the contract requires a verbatim span with a character `start`, so a generated answer must be mapped back to the context, and free text rarely matches Exact Match.

## 9. Known limitations

- English only; contexts up to 5,000 characters and questions up to 500 (larger gets 400). One contiguous span of at most 30 tokens: no yes/no, multi-span, counting or multi-step answers.
- **Quality outside SQuAD (measured).** The threshold (3.25) was tuned on SQuAD. On our 44-question mini-set on the course readings, threshold unchanged, versus the SQuAD hold-out in parentheses: EM on answerable 0.594 (0.748), correct `null` 0.500 (0.416), EM reweighted to 25 % unanswerable 0.570 (0.665), F1 0.704 (0.720). Most of the EM drop is boundary mismatch (8 of 13 misses overlap the reference). The service abstains correctly on 4 of 4 obviously unanswerable questions but only 2 of 8 near-answerable ones. With 44 items (one unanswerable item is about 8 points of `null` accuracy) the numbers are indicative. We did not retune the threshold: 3.25 gave the best overall EM of the six values checked (−4 to 5), and tuning on 44 items would overfit.
- One request at a time, one worker: concurrent clients queue up.
- Depends on one laptop and a free ngrok tunnel (20,000 requests and 1 GB per month): if either drops, every request fails.
- `start` is a Python string index (code points).
- Validation loss rose in epoch 3; another epoch count or checkpoint selection might do better (untested).
