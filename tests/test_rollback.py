"""Testa scripts/rollback.py contra um MLflow Model Registry real (file store local).

Não usa PySpark: rollback.py só reatribui aliases via MlflowClient, nunca
carrega o modelo -- por isso os testes registram versões "vazias" (sem um
artefato de modelo real), o suficiente para exercitar toda a lógica de aliasing.
"""

from __future__ import annotations

import json
import sys

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from rollback import main as rollback_main


@pytest.fixture
def registry(tmp_path):
    tracking_uri = f"file:{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = "test_catalog.fraude.modelo_fraude"
    client.create_registered_model(model_name)

    versions = []
    for i in range(2):
        with mlflow.start_run() as run:
            mlflow.log_param("i", i)
            run_id = run.info.run_id
        mv = client.create_model_version(name=model_name, source=f"runs:/{run_id}/model", run_id=run_id)
        versions.append(mv.version)

    client.set_registered_model_alias(model_name, "champion", versions[1])  # versão "ruim" em produção
    return tracking_uri, client, model_name, versions


def test_dry_run_makes_no_changes(registry, tmp_path):
    tracking_uri, client, model_name, versions = registry
    output_path = tmp_path / "rollback_report.json"

    exit_code = rollback_main(
        [
            "--project", "ml-fraude",
            "--environment", "prd",
            "--tracking-uri", tracking_uri,
            "--registered-model-name", model_name,
            "--target-version", str(versions[0]),
            "--author", "jane@empresa.com",
            "--reason", "Regressão de recall detectada",
            "--build-id", "123",
            "--dry-run",
            "--output-path", str(output_path),
        ]
    )
    assert exit_code == 0
    champion_after = client.get_model_version_by_alias(model_name, "champion")
    assert champion_after.version == versions[1]  # inalterado

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["to_version"] == str(versions[0])


def test_confirm_flag_rolls_back_alias(registry, tmp_path):
    tracking_uri, client, model_name, versions = registry
    output_path = tmp_path / "rollback_report.json"

    exit_code = rollback_main(
        [
            "--project", "ml-fraude",
            "--environment", "prd",
            "--tracking-uri", tracking_uri,
            "--registered-model-name", model_name,
            "--target-version", str(versions[0]),
            "--author", "jane@empresa.com",
            "--reason", "Regressão de recall detectada",
            "--build-id", "123",
            "--confirm",
            "--output-path", str(output_path),
        ]
    )
    assert exit_code == 0

    champion_after = client.get_model_version_by_alias(model_name, "champion")
    previous_champion_after = client.get_model_version_by_alias(model_name, "previous_champion")
    assert champion_after.version == versions[0]
    assert previous_champion_after.version == versions[1]

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is False
    assert report["author"] == "jane@empresa.com"


def test_dry_run_flag_overrides_confirm_for_safety(registry, tmp_path):
    """--dry-run e --confirm juntos: a segurança vence, nada é alterado."""
    tracking_uri, client, model_name, versions = registry
    output_path = tmp_path / "rollback_report.json"

    rollback_main(
        [
            "--project", "ml-fraude", "--environment", "prd",
            "--tracking-uri", tracking_uri, "--registered-model-name", model_name,
            "--target-version", str(versions[0]),
            "--author", "jane@empresa.com", "--reason", "teste", "--build-id", "1",
            "--confirm", "--dry-run",
            "--output-path", str(output_path),
        ]
    )
    champion_after = client.get_model_version_by_alias(model_name, "champion")
    assert champion_after.version == versions[1]


def test_fails_when_target_version_does_not_exist(registry, tmp_path):
    tracking_uri, client, model_name, versions = registry
    output_path = tmp_path / "rollback_report.json"

    exit_code = rollback_main(
        [
            "--project", "ml-fraude", "--environment", "prd",
            "--tracking-uri", tracking_uri, "--registered-model-name", model_name,
            "--target-version", "9999",
            "--author", "jane@empresa.com", "--reason", "teste", "--build-id", "1",
            "--confirm",
            "--output-path", str(output_path),
        ]
    )
    assert exit_code == 1


def test_smoke_test_failure_aborts_rollback(registry, tmp_path):
    tracking_uri, client, model_name, versions = registry
    output_path = tmp_path / "rollback_report.json"

    exit_code = rollback_main(
        [
            "--project", "ml-fraude", "--environment", "prd",
            "--tracking-uri", tracking_uri, "--registered-model-name", model_name,
            "--target-version", str(versions[0]),
            "--author", "jane@empresa.com", "--reason", "teste", "--build-id", "1",
            "--confirm",
            "--smoke-test-command", f"{sys.executable} -c \"import sys; sys.exit(1)\"",
            "--output-path", str(output_path),
        ]
    )
    assert exit_code == 1
    champion_after = client.get_model_version_by_alias(model_name, "champion")
    assert champion_after.version == versions[1]  # não mudou -- smoke test bloqueou
