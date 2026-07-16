"""Integração com o script central scripts/validate_model_artifact.py (via subprocess).

Este teste roda o repositório consumidor contra o script de governança do
repositório central, exatamente como a pipeline Azure DevOps faria.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CENTRAL_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def _run_validate_model_artifact(**kwargs) -> subprocess.CompletedProcess:
    args = [sys.executable, str(CENTRAL_SCRIPTS_DIR / "validate_model_artifact.py")]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    return subprocess.run(args, capture_output=True, text=True)


def test_validate_model_artifact_passes_when_consistent(tmp_path):
    run_metadata = {
        "git_commit": "abc123",
        "mlflow_run_id": "runid-001",
        "model_version": "3",
    }
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata), encoding="utf-8")

    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"conteudo-imutavel-do-modelo")

    import hashlib

    sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    sha256_path = tmp_path / "artifact_sha256.txt"
    sha256_path.write_text(sha256, encoding="utf-8")

    result = _run_validate_model_artifact(
        run_metadata_path=run_metadata_path,
        expected_commit="abc123",
        expected_run_id="runid-001",
        expected_model_version="3",
        artifact_path=artifact_path,
        expected_sha256_path=sha256_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_model_artifact_fails_on_commit_mismatch(tmp_path):
    run_metadata = {"git_commit": "abc123", "mlflow_run_id": "runid-001", "model_version": "3"}
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata), encoding="utf-8")

    result = _run_validate_model_artifact(
        run_metadata_path=run_metadata_path,
        expected_commit="DIFFERENT-SHA",
        expected_run_id="runid-001",
        expected_model_version="3",
    )
    assert result.returncode == 1


def test_validate_model_artifact_fails_on_sha256_mismatch(tmp_path):
    run_metadata = {"git_commit": "abc123", "mlflow_run_id": "runid-001", "model_version": "3"}
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata), encoding="utf-8")

    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"conteudo-original")
    sha256_path = tmp_path / "artifact_sha256.txt"
    sha256_path.write_text("0" * 64, encoding="utf-8")  # hash divergente de propósito

    result = _run_validate_model_artifact(
        run_metadata_path=run_metadata_path,
        expected_commit="abc123",
        expected_run_id="runid-001",
        expected_model_version="3",
        artifact_path=artifact_path,
        expected_sha256_path=sha256_path,
    )
    assert result.returncode == 1
