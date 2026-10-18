import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from complaint.config import REPO_ROOT
from complaint.predict import load, predict

MODEL_DIR = os.environ.get("MODEL_DIR", REPO_ROOT / "outputs" / "roberta_512" / "model")

state = {}


@asynccontextmanager
async def lifespan(app):
    state["model"] = load(MODEL_DIR)
    yield


app = FastAPI(title="Complaint triage", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "labels": state["model"].labels}


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class Prediction(BaseModel):
    product: str
    confidence: float
    route: Literal["auto", "human"]


class ClassifyResponse(BaseModel):
    results: list[Prediction]


@app.post("/classify")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    return ClassifyResponse(results=predict(state["model"], req.texts))
