import threading
import time

import pytest
from fastapi.testclient import TestClient

from qa.server import create_app

TOKEN = "test-token-123"


class FakeModel:
    base_model, params, max_seq_length, doc_stride = "fake-base", 1234, 384, 128

    def predict(self, context, question):
        if question == "boom":
            raise RuntimeError("model exploded")
        if question.startswith("nothing"):
            return None, None, 0.9
        word = context.split()[0]
        return word, context.index(word), 0.8


def wait_ready(client, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError("service never became ready")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("QA_API_TOKEN", TOKEN)
    monkeypatch.setenv("QA_TEAM", "team-test")
    with TestClient(create_app(lambda: FakeModel())) as c:
        wait_ready(c)
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}
BODY = {"id": "q-1", "context": "Lempa is a river in El Salvador.", "question": "What is Lempa?"}


def test_health_is_503_while_loading_then_ok(monkeypatch):
    monkeypatch.setenv("QA_API_TOKEN", TOKEN)
    gate = threading.Event()

    def slow_loader():
        gate.wait(5)
        return FakeModel()

    with TestClient(create_app(slow_loader)) as c:
        r = c.get("/health")
        assert r.status_code == 503 and r.json() == {"status": "loading"}
        gate.set()
        wait_ready(c)
        assert c.get("/health").json() == {"status": "ok"}


def test_info_requires_token(client):
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": f"bearer {TOKEN}"}, {"Authorization": TOKEN}):
        r = client.get("/info", headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"


def test_info_ok(client):
    r = client.get("/info", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"team": "team-test", "model": "fake-base", "params": 1234,
                        "max_seq_length": 384, "doc_stride": 128}


def test_predict_ok_and_offsets(client):
    r = client.post("/predict", headers=AUTH, json=BODY)
    j = r.json()
    assert r.status_code == 200 and j["id"] == "q-1" and 0 <= j["score"] <= 1
    assert BODY["context"][j["start"]:j["start"] + len(j["answer"])] == j["answer"]


def test_predict_null(client):
    r = client.post("/predict", headers=AUTH, json={**BODY, "question": "nothing here"})
    assert r.status_code == 200
    j = r.json()
    assert j["answer"] is None and j["start"] is None and "score" in j


def test_predict_without_token_is_401_even_with_garbage_body(client):
    assert client.post("/predict", json=BODY).status_code == 401
    r = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 401


@pytest.mark.parametrize("payload", [
    {"id": "q", "context": "c"},                                            # missing field
    {**BODY, "extra": 1},                                                   # extra field not allowed
    {**BODY, "context": ""},                                                # too short
    {**BODY, "context": "x" * 5001},                                        # too long
    {**BODY, "question": "x" * 501},
    {**BODY, "id": "x" * 65},
    {**BODY, "id": 17},                                                     # wrong type
    [1, 2, 3],                                                              # not an object
])
def test_predict_bad_requests(client, payload):
    r = client.post("/predict", headers=AUTH, json=payload)
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request" and "detail" in r.json()


def test_predict_malformed_json(client):
    r = client.post("/predict", headers={**AUTH, "Content-Type": "application/json"}, content=b"{oops")
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_predict_model_failure_is_500(client):
    r = client.post("/predict", headers=AUTH, json={**BODY, "question": "boom"})
    assert r.status_code == 500 and r.json() == {"error": "internal_error"}


def test_unset_token_rejects_everything(monkeypatch):
    monkeypatch.delenv("QA_API_TOKEN", raising=False)
    monkeypatch.setattr("qa.config.api_token", lambda: "")
    with TestClient(create_app(lambda: FakeModel())) as c:
        wait_ready(c)
        assert c.get("/info", headers={"Authorization": "Bearer "}).status_code == 401
        assert c.post("/predict", headers={"Authorization": "Bearer "}, json=BODY).status_code == 401
