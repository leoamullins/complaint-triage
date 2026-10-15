# Complaints Triage

NLP on 3.85M US consumer complaints against banks and lenders. It classifies each complaint by product, discovers emerging issues and then turns the findings into recommendations.

## Results so far

| Model | Val macro-F1 (raw) | Val macro-F1 (prior-adjusted) | Test macro-F1 |
|---|---|---|---|
| TF-IDF + logistic regression | 0.671 | 0.710 (τ = 0.5) | 0.724 [0.711, 0.737] |
| RoBERTa-base, 512 tokens, 2 epochs | 0.691 | 0.738 (τ = 0.75) | **0.753** [0.740, 0.766] |

Training data is class-balanced, but validation and test are not (credit reporting is 67% of validation and 51% of test). Prior adjustment adds τ · log(p_val / p_train) to each class score to correct for this. τ is chosen on validation for both models, then applied once to test.

Brackets are 95% confidence intervals from a paired bootstrap over test complaints (2,000 resamples, `scripts/bootstrap_test.py`). RoBERTa's gain is +2.9 points [+1.8, +4.0], and it scores below the baseline in none of the resamples.

On test, RoBERTa beats the baseline on all ten classes (accuracy 0.850 vs 0.838). The largest gains are on the hardest classes: personal/payday loans (F1 0.58 → 0.66) and "Other" (0.28 → 0.36). Credit reporting (0.91) and mortgage (0.90) are easiest to classify. "Other" remains weak: precision is 0.78 but recall only 0.23, as the model rarely predicts it.

## Calibration

For triage, the model's confidence has to be trustworthy, so that uncertain complaints can be routed to a person. Calibration checks whether a stated confidence of 90% means the model is right about 90% of the time.

| RoBERTa on test | ECE | Log loss | Macro-F1 |
|---|---|---|---|
| Raw | 0.036 | 0.522 | 0.729 |
| Prior-adjusted (τ = 0.75) | 0.039 | 0.461 | 0.753 |
| Prior-adjusted + temperature (T = 1.13) | **0.018** | **0.451** | 0.753 |

ECE is expected calibration error over 15 equal-width confidence bins. The prior adjustment mainly improves log loss by correcting class frequencies. Temperature scaling, with T fitted on validation, halves ECE: T > 1 means the fine-tuned model was overconfident. It divides all scores by the same number, so predictions and macro-F1 are unchanged.

![Reliability diagram on test](figures/reliability_test.png)

Points below the diagonal are overconfident. The lower panel shows how many predictions fall in each bin; about half are above 0.93 confidence, where the calibrated model is close to the diagonal. Bins below 0.3 hold only a handful of predictions.
