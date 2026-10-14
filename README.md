# Complaints Triage

NLP on 3.85M US consumer complaints against banks and lenders. It classifies each complaint by product, discovers emerging issues and then turns the findings into recommendations.

## Results so far

| Model | Validation macro-F1 |
|---|---|
| TF-IDF + logistic regression | **0.77** |
| Fine-tuned transformer | *in progress* |

Mortgage and student loan complaints are easiest to classify (F1 0.93–0.94); "Other" and personal/payday loans are hardest. Credit reporting has high recall but lower precision: it absorbs debt collection complaints that mention credit reports.
