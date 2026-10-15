# scripts/baseline_on_sample.py
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import time


DATA = Path("/mnt/c/Users/leomu/Downloads/complaint-triage")
train, val, test = (
    pd.read_parquet(DATA / f"{s}.parquet") for s in ["train", "val", "test"]
)

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

# prior correction, same as RoBERTa: true prior from val, training prior from train
train_prior = train["label"].value_counts(normalize=True).reindex(classes).to_numpy()
val_prior = val["label"].value_counts(normalize=True).reindex(classes).to_numpy()
log_ratio = np.log(val_prior) - np.log(train_prior)

for name, df in [("val", val), ("test", test)]:
    logp = clf.predict_log_proba(vec.transform(df["text"]))
    raw = classes[logp.argmax(1)]
    adj = classes[(logp + log_ratio).argmax(1)]
    print(f"\n=== {name} ===")
    print("raw macro-F1:     ", round(f1_score(df["label"], raw, average="macro"), 4))
    print("adjusted macro-F1:", round(f1_score(df["label"], adj, average="macro"), 4))
    if name == "test":
        print(classification_report(df["label"], adj, digits=3))
