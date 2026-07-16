from __future__ import annotations

import pytest

from churn_model.exceptions import ModelEvaluationError
from churn_model.models.evaluate import (
    compute_invalid_prediction_percentage,
    compute_model_size_bytes,
    evaluate_model,
    measure_inference_latency_ms,
)
from churn_model.models.train import prepare_datasets, train_candidate


@pytest.fixture(scope="module")
def fitted_pipeline(module_customers_df, module_config):
    train_df, val_df, test_df = prepare_datasets(module_customers_df, module_config)
    result = train_candidate(train_df, val_df, test_df, module_config)
    return result.pipeline_model, val_df


def test_evaluate_model_returns_expected_metric_keys(fitted_pipeline):
    model, val_df = fitted_pipeline
    metrics = evaluate_model(model, val_df, "churn")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "latency_ms", "model_size_bytes"):
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_model_rejects_empty_dataframe(fitted_pipeline):
    model, val_df = fitted_pipeline
    with pytest.raises(ModelEvaluationError):
        evaluate_model(model, val_df.limit(0), "churn")


def test_compute_model_size_bytes_positive(fitted_pipeline):
    model, _ = fitted_pipeline
    assert compute_model_size_bytes(model) > 0


def test_measure_inference_latency_is_positive(fitted_pipeline):
    model, val_df = fitted_pipeline
    latency = measure_inference_latency_ms(model, val_df.limit(10), repeats=1)
    assert latency >= 0.0


def test_compute_invalid_prediction_percentage_zero_on_valid_predictions(fitted_pipeline):
    model, val_df = fitted_pipeline
    predictions = model.transform(val_df)
    assert compute_invalid_prediction_percentage(predictions) == 0.0
