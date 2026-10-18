import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from complaint.config import REPO_ROOT, default_data_dir
from complaint.predict import load, postprocess, predict

MODEL_DIR = REPO_ROOT / "outputs" / "roberta_512" / "model"
PREDS_TEST = default_data_dir() / "preds_test.parquet"

rng = np.random.default_rng(0)
LOGITS = rng.normal(size=(50, 10)) * 3
LOG_RATIO = rng.normal(size=10)


# postprocess: fast, no model needed


def test_no_adjustment_is_plain_softmax():
    pred, conf, _ = postprocess(LOGITS, np.zeros(10), tau=0.0, temperature=1.0, threshold=0.8)
    probs = softmax(LOGITS, axis=1)
    np.testing.assert_array_equal(pred, probs.argmax(1))
    np.testing.assert_allclose(conf, probs.max(1))


def test_temperature_never_changes_the_prediction():
    base, _, _ = postprocess(LOGITS, LOG_RATIO, tau=0.75, temperature=1.0, threshold=0.8)
    for t in (0.5, 1.1283, 3.0):
        pred, _, _ = postprocess(LOGITS, LOG_RATIO, tau=0.75, temperature=t, threshold=0.8)
        np.testing.assert_array_equal(pred, base)


def test_higher_temperature_lowers_confidence():
    _, low_t, _ = postprocess(LOGITS, LOG_RATIO, tau=0.75, temperature=1.0, threshold=0.8)
    _, high_t, _ = postprocess(LOGITS, LOG_RATIO, tau=0.75, temperature=2.0, threshold=0.8)
    assert (high_t <= low_t).all()


def test_prior_adjustment_can_flip_a_close_call():
    logits = np.array([[1.0, 1.1]])
    log_ratio = np.array([2.0, -2.0])  # class 0 is far more common in the real mix
    pred, _, _ = postprocess(logits, log_ratio, tau=1.0, temperature=1.0, threshold=0.8)
    assert pred[0] == 0


def test_routing_threshold_boundary_goes_to_auto():
    _, conf, _ = postprocess(LOGITS, LOG_RATIO, tau=0.75, temperature=1.0, threshold=0.8)
    t = conf[0]
    _, _, route = postprocess(LOGITS, LOG_RATIO, tau=0.75, temperature=1.0, threshold=t)
    assert route[0] == "auto"
    assert all(r == ("auto" if c >= t else "human") for c, r in zip(conf, route))


# the real model: slower, skipped if the files are not downloaded

needs_model = pytest.mark.skipif(
    not (MODEL_DIR.exists() and PREDS_TEST.exists()),
    reason="needs outputs/roberta_512/model and data/preds_test.parquet",
)


@pytest.fixture(scope="module")
def model():
    return load(MODEL_DIR)


@needs_model
def test_model_settings_match_reported_run(model):
    assert len(model.labels) == 10
    assert model.tau == 0.75
    assert model.temperature == pytest.approx(1.13, abs=0.01)
    assert model.threshold == 0.8


@needs_model
def test_output_shape_and_order(model):
    texts = [
        "They keep calling me about a debt I already paid off years ago.",
        "There is an account on my credit report that is not mine.",
    ]
    out = predict(model, texts, batch_size=1)
    assert len(out) == 2
    assert set(out[0]) == {"product", "confidence", "route"}
    assert out[0]["product"] == "Debt collection"
    assert out[1]["product"] == "Credit reporting"
    assert all(isinstance(r["confidence"], float) for r in out)


@needs_model
def test_accuracy_matches_reported_run(model):
    # compares against true labels, not the file's pred column: the copy on
    # Drive was saved before the row-order fix (2893d41), so pred is shuffled
    df = pd.read_parquet(PREDS_TEST).sample(500, random_state=0)
    out = predict(model, list(df["text"]))
    acc = np.mean([r["product"] == y for r, y in zip(out, df["label"])])
    # README: test accuracy 0.850; 500 samples gives a standard error of ~1.6 points
    assert acc >= 0.80, f"test accuracy {acc:.3f}, README reports 0.850"
