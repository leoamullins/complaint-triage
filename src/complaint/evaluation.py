import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(y, preds, average="macro"),
        "accuracy": (preds == y).mean(),
    }


def prior_log_ratio(train_labels, true_labels, n_classes):
    """
    log(p_true / p_train) per class; true_labels=None means a uniform target prior
    """
    p_train = np.bincount(train_labels, minlength=n_classes) / len(train_labels)
    if true_labels is None:
        p_true = np.full(n_classes, 1 / n_classes)
    else:
        p_true = np.bincount(true_labels, minlength=n_classes) / len(true_labels)
    eps = 1e-12

    return np.log(p_true + eps) - np.log(p_train + eps)


def adjusted_preds(logits, log_ratio, tau=1.0):
    return (logits + tau * log_ratio).argmax(-1)


def tau_sweep(logits, y, log_ratio, taus=(0, 0.25, 0.5, 0.75, 1.0, 1.25)):
    return {
        t: f1_score(y, adjusted_preds(logits, log_ratio, t), average="macro")
        for t in taus
    }


def save_prior_adjustment(log_ratio, tau, model_dir, best_tau=None):
    d = {"log_ratio": log_ratio.tolist(), "tau": tau}
    if best_tau is not None:
        d["best_tau"] = best_tau
    Path(model_dir, "prior_adjustment.json").write_text(json.dumps(d, indent=2))


def load_prior_adjustment(model_dir):
    d = json.loads(Path(model_dir, "prior_adjustment.json").read_text())
    return np.array(d["log_ratio"]), d["tau"]
