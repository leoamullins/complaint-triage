# Complaints Triage

NLP on 3.85M US consumer complaints against banks and lenders. It classifies each complaint by product, discovers emerging issues and then turns the findings into recommendations.

## Results so far

| Model | Val macro-F1 (raw) | Val macro-F1 (prior-adjusted) | Test macro-F1 |
|---|---|---|---|
| TF-IDF + logistic regression | 0.671 | 0.710 (τ = 0.5) | 0.724 |
| RoBERTa-base, 512 tokens, 2 epochs | 0.691 | 0.738 (τ = 0.75) | **0.753** |

Training data is class-balanced, but validation and test are not (credit reporting is 67% of validation and 51% of test). Prior adjustment adds τ · log(p_val / p_train) to each class score to correct for this. τ is chosen on validation for both models, then applied once to test.

On test, RoBERTa beats the baseline on all ten classes (accuracy 0.850 vs 0.838). The largest gains are on the hardest classes: personal/payday loans (F1 0.58 → 0.66) and "Other" (0.28 → 0.36). Credit reporting (0.91) and mortgage (0.90) are easiest to classify. "Other" remains weak: precision is 0.78 but recall only 0.23, as the model rarely predicts it.
