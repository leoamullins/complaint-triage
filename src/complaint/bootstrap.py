import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y, p, n_classes):
    return f1_score(y, p, labels=range(n_classes), average="macro", zero_division=0)


def paired_bootstrap(y, pred_a, pred_b, n_classes, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    a, b = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a[i] = macro_f1(y[idx], pred_a[idx], n_classes)
        b[i] = macro_f1(y[idx], pred_b[idx], n_classes)
    diff = b - a
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    return {
        "a": (macro_f1(y, pred_a, n_classes), ci(a)),
        "b": (macro_f1(y, pred_b, n_classes), ci(b)),
        "diff": (
            macro_f1(y, pred_b, n_classes) - macro_f1(y, pred_a, n_classes),
            ci(diff),
        ),
        "p_b_not_better": (diff <= 0).mean(),
    }
