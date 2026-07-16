"""Fluxo completo Champion/Challenger com um Champion PRÉ-EXISTENTE real (não o cenário de primeiro modelo)."""

from __future__ import annotations

from churn_model.models.champion_challenger import run_champion_challenger_comparison
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets, train_candidate


def _train_and_register(sample_customers_df, test_config, client, alias, dataset_version, max_iter):
    train_df, val_df, test_df = prepare_datasets(sample_customers_df, test_config)
    hyperparameters = dict(test_config.model.hyperparameters)
    hyperparameters["max_iter"] = max_iter
    config_variant = test_config.__class__(
        **{
            **test_config.__dict__,
            "model": test_config.model.__class__(
                type=test_config.model.type,
                algorithm=test_config.model.algorithm,
                primary_metric=test_config.model.primary_metric,
                hyperparameters=hyperparameters,
                thresholds=test_config.model.thresholds,
            ),
        }
    )
    result = train_candidate(train_df, val_df, test_df, config_variant)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)
    run_id = log_training_run(
        client=client,
        config=config_variant,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version=dataset_version,
        model_alias=alias,
    )
    version = register_model_version(client, config_variant.mlflow.registered_model_name, run_id)
    set_alias(client, config_variant.mlflow.registered_model_name, alias, version)
    return result.pipeline_model, run_id, version, test_df


def test_comparison_against_existing_champion(sample_customers_df, test_config, tmp_path):
    client = configure_mlflow(test_config)

    _, champion_run_id, champion_version, test_df = _train_and_register(
        sample_customers_df, test_config, client, "champion", "v1", max_iter=5
    )
    challenger_model, challenger_run_id, challenger_version, _ = _train_and_register(
        sample_customers_df, test_config, client, "challenger", "v2", max_iter=15
    )

    recommendation = run_champion_challenger_comparison(
        client=client,
        config=test_config,
        challenger_model=challenger_model,
        challenger_run_id=challenger_run_id,
        challenger_version=challenger_version,
        test_df=test_df,
        dataset_version="v2",
        output_dir=tmp_path / "comparison",
    )

    assert recommendation["is_first_model"] is False
    assert recommendation["champion_version"] == champion_version
    assert recommendation["challenger_version"] == challenger_version
    assert recommendation["champion_run_id"] == champion_run_id
    assert recommendation["challenger_run_id"] == challenger_run_id
    assert recommendation["champion_metric"] is not None
    assert recommendation["absolute_improvement"] is not None
