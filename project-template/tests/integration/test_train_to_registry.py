"""Fluxo completo: treinar -> registrar -> montar os metadados exigidos pela plataforma."""

from __future__ import annotations

from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import (
    GitContext,
    configure_mlflow,
    log_training_run,
    register_model_version,
    set_alias,
)
from churn_model.models.train import prepare_datasets, train_candidate


def test_run_metadata_fields_are_all_real_values(monkeypatch, sample_customers_df, test_config):
    monkeypatch.setenv("BUILD_SOURCEVERSION", "abc1234")
    monkeypatch.setenv("BUILD_SOURCEBRANCH", "refs/heads/feature/x")
    monkeypatch.setenv("BUILD_BUILDID", "999")
    monkeypatch.setenv("BUILD_BUILDNUMBER", "999.1")

    train_df, val_df, test_df = prepare_datasets(sample_customers_df, test_config)
    result = train_candidate(train_df, val_df, test_df, test_config)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)

    client = configure_mlflow(test_config)
    run_id = log_training_run(
        client=client,
        config=test_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="v1",
    )
    version = register_model_version(client, test_config.mlflow.registered_model_name, run_id)
    set_alias(client, test_config.mlflow.registered_model_name, "challenger", version)

    git_context = GitContext.from_environment()
    run_metadata = {
        "project_name": test_config.project_name,
        "git_commit": git_context.commit,
        "git_branch": git_context.branch,
        "build_id": git_context.build_id,
        "mlflow_run_id": run_id,
        "registered_model_name": test_config.mlflow.registered_model_name,
        "model_version": version,
        "model_alias": "challenger",
    }

    # Nenhum campo obrigatório pode ser fictício/None em uma execução real.
    for key, value in run_metadata.items():
        assert value not in (None, "", "local-invented"), f"{key} está vazio/fictício"

    assert run_metadata["git_commit"] == "abc1234"
    assert run_metadata["build_id"] == "999"
    # run_id real do MLflow tem 32 caracteres hexadecimais.
    assert len(run_metadata["mlflow_run_id"]) == 32
    int(run_metadata["model_version"])  # deve ser conversível para inteiro (versão real)
