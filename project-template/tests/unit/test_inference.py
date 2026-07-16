from __future__ import annotations

import pytest

from churn_model.exceptions import InferenceError, SchemaValidationError
from churn_model.models import inference
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets, train_candidate


@pytest.fixture(scope="module")
def registered_champion(module_customers_df, module_config):
    train_df, val_df, test_df = prepare_datasets(module_customers_df, module_config)
    result = train_candidate(train_df, val_df, test_df, module_config)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)

    client = configure_mlflow(module_config)
    run_id = log_training_run(
        client=client,
        config=module_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="v1",
    )
    version = register_model_version(client, module_config.mlflow.registered_model_name, run_id)
    set_alias(client, module_config.mlflow.registered_model_name, "champion", version)
    return test_df


def test_score_adds_required_columns(module_config, registered_champion):
    result = inference.score(module_config, registered_champion)
    for column in (
        "prediction",
        "probability_churn",
        "model_alias",
        "model_version",
        "scoring_timestamp",
        "git_commit",
    ):
        assert column in result.columns
    assert result.count() == registered_champion.count()


def test_score_rejects_missing_required_columns(module_config, registered_champion):
    broken_df = registered_champion.drop("tenure_months")
    with pytest.raises(SchemaValidationError):
        inference.score(module_config, broken_df)


def test_score_rejects_empty_dataframe(module_config, registered_champion):
    with pytest.raises(InferenceError):
        inference.score(module_config, registered_champion.limit(0))


def test_score_can_target_specific_alias(module_config, registered_champion):
    result = inference.score(module_config, registered_champion, model_alias="champion")
    assert result.filter(result.model_alias == "champion").count() == result.count()
