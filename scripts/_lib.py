"""Utilitários compartilhados pelos scripts de governança da plataforma central.

Todos os scripts em ``scripts/`` são deliberadamente independentes de PySpark:
eles operam sobre artefatos JSON já produzidos pelos jobs Databricks do projeto
e sobre o MLflow Model Registry via ``MlflowClient``, para que possam rodar em
qualquer agente Azure DevOps sem precisar de um cluster Spark.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return output_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(logger: logging.Logger, message: str, **fields: Any) -> int:
    logger.error(json.dumps({"error": message, **fields}, default=str))
    return 1


def succeed(logger: logging.Logger, message: str, **fields: Any) -> int:
    logger.info(json.dumps({"message": message, **fields}, default=str))
    return 0


def get_mlflow_client(tracking_uri: str, registry_uri: str | None = None):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri or tracking_uri)


def exit_with(code: int) -> None:
    sys.exit(code)
