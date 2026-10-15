import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def corrected_logits(logits, log_ratio, temperature=1.0, tau=1.0):
    return (logits + tau * log_ratio) / temperature


def nll(logits, y):
    return -log_softmax(logits, axis=1)[np.arange(len(y)), y].mean()


def fit_temperature(logits, y, log_ratio, tau=1.0):
    obj = lambda t: nll(corrected_logits(logits, log_ratio, temperature=t, tau=tau), y)
    res = minimize_scalar(obj, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def reliability_table(probs, y, n_bins=20):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(m.sum()),
                    "confidence": conf[m].mean(),
                    "accuracy": correct[m].mean(),
                }
            )
    return rows


def routing_table(probs, y, thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)):
    """
    send predictions below each confidence threshold to a human; report the
    share reviewed and the accuracy of the auto-routed rest
    """
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append(
            {
                "threshold": t,
                "human_share": 1 - auto.mean(),
                "auto_accuracy": correct[auto].mean() if auto.any() else float("nan"),
                "errors_caught": (~correct & ~auto).sum() / max((~correct).sum(), 1),
            }
        )
    return rows
