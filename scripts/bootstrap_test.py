# paired bootstrap of test macro-F1: TF-IDF baseline vs fine-tuned RoBERTa
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from complaint.bootstrap import macro_f1, paired_bootstrap

DATA = Path("/mnt/c/Users/leomu/Downloads/complaint-triage")
# downloaded from complaint-outputs/roberta-512 on Drive
ROBERTA_PREDS = Path(__file__).resolve().parents[1] / "data" / "preds_test.parquet"
BASELINE_TAU = 0.5  # chosen on val in 02_baseline.ipynb

assert ROBERTA_PREDS.exists(), f"download preds_test.parquet from Drive to {ROBERTA_PREDS}"

train, val, test = (
    pd.read_parquet(DATA / f"{s}.parquet") for s in ["train", "val", "test"]
)

# baseline, same setup as 02_baseline.ipynb
vec = TfidfVectorizer(
    ngram_range=(1, 2), min_df=5, max_features=200_000, sublinear_tf=True
)
clf = LogisticRegression(class_weight="balanced", max_iter=1000)

Xtr = vec.fit_transform(train["text"])
t = time.time()
print("fitting logreg...")
clf.fit(Xtr, train["label"])
print(f"  done in {time.time() - t:.0f}s")
classes = clf.classes_

train_prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy()
val_prior = val["label"].value_counts(normalize=True).reindex(classes).to_numpy()
log_ratio = np.log(val_prior) - np.log(train_prior)

logp = clf.predict_log_proba(vec.transform(test["text"]))
baseline = pd.DataFrame(
    {
        "complaint_id": test["complaint_id"],
        "label": test["label"],
        "pred_baseline": classes[(logp + BASELINE_TAU * log_ratio).argmax(1)],
    }
)

# align both models on the same complaints
roberta = pd.read_parquet(ROBERTA_PREDS, columns=["complaint_id", "label", "pred"])
df = baseline.merge(roberta, on="complaint_id", suffixes=("", "_roberta"))
assert len(df) == len(test) == len(roberta), "test rows don't line up"
assert (df["label"] == df["label_roberta"]).all(), "labels disagree between files"

# sorted names -> ids, same order as make_label_maps and clf.classes_
label2id = {name: i for i, name in enumerate(sorted(df["label"].unique()))}
assert list(label2id) == list(classes)
n_classes = len(label2id)
y = df["label"].map(label2id).to_numpy()
pred_a = df["pred_baseline"].map(label2id).to_numpy()
pred_b = df["pred"].map(label2id).to_numpy()

# should reproduce 0.7242 and 0.7534
print(f"baseline test macro-F1: {macro_f1(y, pred_a, n_classes):.4f}")
print(f"roberta  test macro-F1: {macro_f1(y, pred_b, n_classes):.4f}")

t = time.time()
res = paired_bootstrap(y, pred_a, pred_b, n_classes)
print(f"\nbootstrap done in {time.time() - t:.0f}s")

for name, key in [("baseline", "a"), ("roberta", "b"), ("difference", "diff")]:
    point, (lo, hi) = res[key]
    print(f"{name:<11} {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"P(roberta <= baseline): {res['p_b_not_better']:.4f}")
