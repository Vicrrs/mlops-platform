"""Smoke test do pacote: carrega o Champion registrado e faz uma inferência pequena.

Corresponde ao mesmo contrato usado por resources/smoke-test.job.yml em
produção: carregar o alias 'champion', pontuar uma amostra pequena, validar
schema e ausência de previsões nulas.
"""

from __future__ import annotations

import json
import subprocess
import sys

from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets, train_candidate


def test_smoke_test_cli_passes_against_real_champion(sample_customers_df, test_config, tmp_path):
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
    set_alias(client, test_config.mlflow.registered_model_name, "champion", version)

    conf_path = tmp_path / "smoke.yml"
    conf_path.write_text(
        "\n".join(
            [
                f"environment: {test_config.environment}",
                "data:",
                "  source: local",
                "  raw_path: data/sample/customers_churn.csv",
                "mlflow:",
                f"  tracking_uri: {test_config.mlflow.tracking_uri}",
                f"  registry_uri: {test_config.mlflow.registry_uri}",
                f"  registered_model_name: {test_config.mlflow.registered_model_name}",
            ]
        ),
        encoding="utf-8",
    )

    # Executado em subprocesso (não in-process) para não derrubar a SparkSession
    # compartilhada da suíte de testes: o CLI encerra a sua própria sessão ao final,
    # como faria de fato um job Databricks de curta duração.
    output_dir = tmp_path / "smoke_test"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_model.cli.run_smoke_test",
            "--config",
            str(conf_path),
            "--sample-size",
            "15",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output_dir / "smoke_test_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["row_count"] == 15
    assert report["null_predictions"] == 0
