from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from complaint.calibration import corrected_logits
from complaint.evaluation import load_prior_adjustment


@dataclass
class Model:
    tokeniser: object
    net: object
    labels: list[str]
    log_ratio: np.ndarray
    tau: float
    temperature: float
    threshold: float
    device: str


def postprocess(logits, log_ratio, tau, temperature, threshold):
    """
    raw logits (n, n_classes) - > predicted ids, confidences, routes
    """
    logits = corrected_logits(logits, log_ratio, temperature=temperature, tau=tau)
    probs = softmax(logits, axis=1)

    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    route = np.where(conf >= threshold, "auto", "human")
    return pred, conf, route


def load(model_dir, device=None):
    """
    read a saved model directory into a Model
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_dir)
    net = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    labels = [net.config.id2label[i] for i in range(len(net.config.id2label))]
    log_ratio, tau, temperature, threshold = load_prior_adjustment(model_dir)
    return Model(
        tokeniser=tok,
        net=net,
        labels=labels,
        log_ratio=log_ratio,
        tau=tau,
        temperature=temperature,
        threshold=threshold,
        device=device,
    )


def predict(model, texts, batch_size=32, max_len=512):
    """
    texts -> one {'product', 'confidence', 'route'} dict per text, in input order
    """
    if not texts:
        return []

    logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = model.tokeniser(
            batch,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            net = model.net(**enc)

        logits.append(net.logits.cpu().numpy())

    arr = np.concatenate(logits, axis=0)
    pred, conf, route = postprocess(
        arr,
        log_ratio=model.log_ratio,
        tau=model.tau,
        temperature=model.temperature,
        threshold=model.threshold,
    )

    results = []
    for p, c, r in zip(pred, conf, route):
        res = {"product": model.labels[p], "confidence": float(c), "route": str(r)}
        results.append(res)
    return results
