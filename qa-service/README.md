# Extractive QA as a Service

Fine-tuned encoder for SQuAD 2.0-style extractive question answering, served through a
token-protected FastAPI service (contract: `contract/qa_api_contract.yaml`).

```
src/qa/          service code (server, model wrapper, post-processing, features, metrics)
train/           train.py, evaluate.py, tune_threshold.py
scripts/         run_server.py, check_service.py, plot_loss.py
tests/           pytest suite (no GPU, no internet, no model needed)
logs/            training logs and loss curves (deliverable)
data/            public_sample.json (released by the instructor)
models/          weights live here locally but are NOT committed
```

Quick start (see `GUIA_PASOS.md` for the step-by-step version):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
pytest
cp .env.example .env            # then edit it (the .env file is git-ignored)
python scripts/run_server.py    # http://127.0.0.1:8000
python scripts/check_service.py # contract + public sample + latency
```

The access token is read from the `QA_API_TOKEN` environment variable. It is never stored in the code.
