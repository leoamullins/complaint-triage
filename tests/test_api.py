from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import get_token

from complaint.api import MODEL_DIR, app

# the Hub repo is private, so without a local copy the tests need a token
pytestmark = pytest.mark.skipif(
    not (Path(MODEL_DIR).is_dir() or get_token()),
    reason=f"needs the model in {MODEL_DIR} or a Hugging Face token",
)


@pytest.fixture(scope="module")
def client():
    # the with block runs lifespan, which loads the model once for this file
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert len(r.json()["labels"]) == 10


def test_classify_returns_one_result_per_text_in_order(client):
    texts = [
        "They keep calling me about a debt I already paid off years ago.",
        "There is an account on my credit report that is not mine.",
    ]
    r = client.post("/classify", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["product"] for x in results] == ["Debt collection", "Credit reporting"]
    for x in results:
        assert 0 <= x["confidence"] <= 1
        assert x["route"] in ("auto", "human")


def test_empty_list_is_rejected(client):
    assert client.post("/classify", json={"texts": []}).status_code == 422


def test_too_many_texts_is_rejected(client):
    assert client.post("/classify", json={"texts": ["x"] * 65}).status_code == 422


def test_missing_texts_field_is_rejected(client):
    assert client.post("/classify", json={"text": "singular"}).status_code == 422


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"
