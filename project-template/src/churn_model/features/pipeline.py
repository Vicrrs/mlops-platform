"""Construção do Spark ML Pipeline completo (pré-processamento + estimador).

O mesmo :class:`~pyspark.ml.PipelineModel` ajustado no treino é o que é
registrado no MLflow e usado na inferência -- não há divergência entre os
dois fluxos porque ambos chamam ``PipelineModel.transform``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

from churn_model.exceptions import FeatureEngineeringError
from churn_model.features.transformations import build_feature_stages
from churn_model.logging_config import get_logger

logger = get_logger(__name__)

_SUPPORTED_ALGORITHMS = {"logistic_regression", "random_forest"}


def build_estimator(algorithm: str, label_column: str, features_column: str, hyperparameters: dict):
    if algorithm == "logistic_regression":
        return LogisticRegression(
            featuresCol=features_column,
            labelCol=label_column,
            maxIter=int(hyperparameters.get("max_iter", 50)),
            regParam=float(hyperparameters.get("reg_param", 0.01)),
            elasticNetParam=float(hyperparameters.get("elastic_net_param", 0.0)),
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            featuresCol=features_column,
            labelCol=label_column,
            numTrees=int(hyperparameters.get("num_trees", 100)),
            maxDepth=int(hyperparameters.get("max_depth", 5)),
            seed=int(hyperparameters.get("seed", 42)),
        )
    raise FeatureEngineeringError(
        f"Algoritmo '{algorithm}' não suportado. Use um de {_SUPPORTED_ALGORITHMS}."
    )


def build_full_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_column: str,
    algorithm: str,
    hyperparameters: dict,
) -> tuple[Pipeline, str]:
    """Monta o Pipeline completo: imputação já deve ter sido aplicada antes; aqui
    entram indexação/encoding/assembling/scaling + o estimador configurado."""
    stages, features_column = build_feature_stages(numeric_columns, categorical_columns)
    estimator = build_estimator(algorithm, label_column, features_column, hyperparameters)
    stages.append(estimator)
    return Pipeline(stages=stages), features_column


def feature_config_dict(
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_column: str,
    version: str,
    algorithm: str,
    hyperparameters: dict,
) -> dict:
    return {
        "version": version,
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "label_column": label_column,
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "all_features": list(numeric_columns) + list(categorical_columns),
    }


def save_feature_config(config: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Configuração de features salva", extra={"extra_fields": {"path": str(output_path)}})
    return output_path


def load_feature_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
