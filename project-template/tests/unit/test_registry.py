from __future__ import annotations

import pytest

from churn_model.exceptions import ChampionNotFoundError
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import (
    configure_mlflow,
    get_model_version_by_alias,
    load_model_by_alias,
    log_training_run,
    register_model_version,
    require_model_version_by_alias,
    set_alias,
)
from churn_model.models.train import prepare_datasets, train_candidate


@pytest.fixture(scope="module")
def trained_run(module_customers_df, module_config):
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
        dataset_version="test-dataset-v1",
    )
    return client, run_id


def test_log_training_run_produces_real_run_id(trained_run):
    client, run_id = trained_run
    assert run_id is not None
    run = client.get_run(run_id)
    assert run.data.tags["project_name"] == "ml-churn-test"
    assert run.data.tags["git_commit"] == "local"
    assert "accuracy" in run.data.metrics


def test_register_model_version_and_set_alias(module_config, trained_run):
    client, run_id = trained_run
    version = register_model_version(client, module_config.mlflow.registered_model_name, run_id)
    assert version is not None

    set_alias(client, module_config.mlflow.registered_model_name, "challenger", version)
    version_info = get_model_version_by_alias(
        client, module_config.mlflow.registered_model_name, "challenger"
    )
    assert version_info.version == version
    assert version_info.run_id == run_id


def test_get_model_version_by_alias_returns_none_when_absent(module_config, trained_run):
    client, _ = trained_run
    result = get_model_version_by_alias(
        client, module_config.mlflow.registered_model_name, "does_not_exist_alias"
    )
    assert result is None


def test_require_model_version_by_alias_raises_when_absent(module_config, trained_run):
    client, _ = trained_run
    with pytest.raises(ChampionNotFoundError):
        require_model_version_by_alias(
            client, module_config.mlflow.registered_model_name, "does_not_exist_alias"
        )


def test_load_model_by_alias_round_trips(module_config, trained_run):
    client, run_id = trained_run
    version = register_model_version(client, module_config.mlflow.registered_model_name, run_id)
    set_alias(client, module_config.mlflow.registered_model_name, "champion", version)

    loaded_model = load_model_by_alias(module_config.mlflow.registered_model_name, "champion")
    assert loaded_model is not None
