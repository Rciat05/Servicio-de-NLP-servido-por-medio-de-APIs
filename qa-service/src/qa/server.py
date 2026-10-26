"""FastAPI service implementing contract/qa_api_contract.yaml.

Run:  python scripts/run_server.py      (listens on 127.0.0.1:8000; the tunnel connects to it)

Design notes
  * The model loads in a background thread, so /health can answer 503 {"status":"loading"} meanwhile.
  * /predict is parsed BY HAND (not with FastAPI's automatic body parsing) so that:
      - authentication is checked BEFORE the body (no token -> always 401), and
      - every error uses the contract's body {"error": "...", "detail": "..."}.
  * Only one request runs the model at a time (lock): the harness is sequential anyway.
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Callable, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qa import config

load_dotenv(config.ROOT / ".env")   # does NOT override variables already set in the terminal
log = logging.getLogger("qa.server")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def _error(status: int, code: str, detail: str | None = None, headers: dict | None = None):
    body = {"error": code}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, headers=headers)


def _check_auth(request: Request) -> Optional[JSONResponse]:
    """Returns None if the bearer token is valid, otherwise a ready-made 401 response."""
    expected = config.api_token()
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    ok = bool(expected) and scheme == "Bearer" and hmac.compare_digest(
        token.encode("utf-8"), expected.encode("utf-8")
    )
    if ok:
        return None
    return _error(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})


def _default_loader():
    from qa.model import QAModel   # imported lazily: tests do not need torch

    model = QAModel(config.model_dir(), device="cpu")
    model.warmup()
    return model


def create_app(model_loader: Callable[[], object] | None = None) -> FastAPI:
    loader = model_loader or _default_loader
    state: dict = {"model": None, "error": None}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not config.api_token():
            log.warning("QA_API_TOKEN is not set: every protected endpoint will answer 401")

        def _load():
            try:
                state["model"] = loader()
                log.info("model loaded, service ready")
            except Exception as exc:  # noqa: BLE001
                log.exception("model failed to load")
                state["error"] = str(exc)

        threading.Thread(target=_load, daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health():
        if state["model"] is not None:
            return {"status": "ok"}
        body = {"status": "loading"}
        if state["error"]:
            body["detail"] = "model failed to load: " + state["error"]
        return JSONResponse(body, status_code=503)

    @app.get("/info")
    def info(request: Request):
        denied = _check_auth(request)
        if denied:
            return denied
        m = state["model"]
        if m is None:
            return _error(503, "internal_error", "model is still loading")
        return {
            "team": config.team_name(),
            "model": m.base_model,
            "params": int(m.params),
            "max_seq_length": int(m.max_seq_length),
            "doc_stride": int(m.doc_stride),
        }

    def _run(model, req: PredictRequest) -> dict:
        with lock:
            answer, start, score = model.predict(req.context, req.question)
        # Last safety net: never return a span that is not verbatim at `start`.
        if answer is not None and (start is None or req.context[start:start + len(answer)] != answer):
            log.error("span/offset mismatch for id=%s; falling back to null", req.id)
            answer, start = None, None
        if answer is None:
            start = None
        return {"id": req.id, "answer": answer, "start": start,
                "score": round(min(1.0, max(0.0, float(score))), 4)}

    @app.post("/predict")
    async def predict(request: Request):
        denied = _check_auth(request)            # 1) auth first
        if denied:
            return denied
        model = state["model"]
        if model is None:
            return _error(503, "internal_error", "model is still loading")
        try:                                      # 2) body must be valid JSON
            payload = json.loads(await request.body())
        except ValueError:
            return _error(400, "bad_request", "body is not valid JSON")
        try:                                      # 3) body must match the schema
            req = PredictRequest.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "body"
            return _error(400, "bad_request", f"{where}: {first['msg']}")
        try:                                      # 4) run the model
            result = await run_in_threadpool(_run, model, req)
        except Exception:  # noqa: BLE001
            log.exception("prediction failed for id=%s", req.id)
            return _error(500, "internal_error")
        return JSONResponse(result)

    return app


app = create_app()
