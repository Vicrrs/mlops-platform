"""Integração real com o MLflow Model Registry (aliases, sem estágios legados)."""

from __future__ import annotations

import pytest

from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import (
    configure_mlflow,
    get_model_version_by_alias,
    log_training_run,
    register_model_version,
    set_alias,
)
from churn_model.models.train import prepare_datasets, train_candidate


@pytest.fixture(scope="module")
def fitted_result(module_customers_df, module_config):
    train_df, val_df, test_df = prepare_datasets(module_customers_df, module_config)
    result = train_candidate(train_df, val_df, test_df, module_config)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)
    return result, train_df, metrics


def _register_new_version(client, module_config, fitted_result, dataset_version):
    """Registra uma NOVA versão do modelo a partir do mesmo pipeline já ajustado
    (uma chamada a log_training_run = um novo run_id = uma nova versão), sem
    reatribuir nenhum alias -- o chamador decide o alias."""
    result, train_df, metrics = fitted_result
    run_id = log_training_run(
        client=client,
        config=module_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version=dataset_version,
    )
    return register_model_version(client, module_config.mlflow.registered_model_name, run_id)


def test_multiple_versions_can_coexist_with_different_aliases(module_config, fitted_result):
    client = configure_mlflow(module_config)
    model_name = module_config.mlflow.registered_model_name

    version_1 = _register_new_version(client, module_config, fitted_result, "v1")
    set_alias(client, model_name, "champion", version_1)
    version_2 = _register_new_version(client, module_config, fitted_result, "v2")
    set_alias(client, model_name, "challenger", version_2)

    champion = get_model_version_by_alias(client, model_name, "champion")
    challenger = get_model_version_by_alias(client, model_name, "challenger")

    assert champion.version == version_1
    assert challenger.version == version_2
    assert champion.version != challenger.version


def test_reassigning_alias_moves_it_to_new_version(module_config, fitted_result):
    client = configure_mlflow(module_config)
    model_name = module_config.mlflow.registered_model_name

    version_1 = _register_new_version(client, module_config, fitted_result, "v3")
    set_alias(client, model_name, "champion", version_1)
    version_2 = _register_new_version(client, module_config, fitted_result, "v4")

    # Simula promoção: alias 'champion' passa da versão 1 para a versão 2, e a
    # versão 1 vira 'previous_champion' -- exatamente o que promote_model.py faz.
    set_alias(client, model_name, "previous_champion", version_1)
    set_alias(client, model_name, "champion", version_2)

    assert get_model_version_by_alias(client, model_name, "champion").version == version_2
    assert get_model_version_by_alias(client, model_name, "previous_champion").version == version_1
