"""Integração real com MLflow Tracking (file store local), sem mocks."""

from __future__ import annotations

import pytest

from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run
from churn_model.models.train import prepare_datasets, train_candidate


@pytest.fixture(scope="module")
def fitted_result(module_customers_df, module_config):
    train_df, val_df, test_df = prepare_datasets(module_customers_df, module_config)
    result = train_candidate(train_df, val_df, test_df, module_config)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)
    return result, train_df, metrics


def test_training_run_logs_params_metrics_tags_and_model(module_config, fitted_result):
    result, train_df, metrics = fitted_result
    client = configure_mlflow(module_config)
    run_id = log_training_run(
        client=client,
        config=module_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="dataset-v1",
    )

    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["algorithm"] == "logistic_regression"
    assert run.data.tags["dataset_version"] == "dataset-v1"
    assert run.data.tags["model_framework"] == "spark_ml"

    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert "model" in artifacts
    assert "feature_config.json" in artifacts


def test_experiment_is_created_once_and_reused(module_config, fitted_result):
    result, train_df, metrics = fitted_result
    client = configure_mlflow(module_config)
    run_id_1 = log_training_run(
        client=client,
        config=module_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="v1",
    )
    run_id_2 = log_training_run(
        client=client,
        config=module_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="v2",
    )
    run_1 = client.get_run(run_id_1)
    run_2 = client.get_run(run_id_2)
    assert run_1.info.experiment_id == run_2.info.experiment_id
    assert run_id_1 != run_id_2
