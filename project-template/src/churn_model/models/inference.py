"""Inferência em lote usando o modelo registrado (Champion por padrão)."""

from __future__ import annotations

from datetime import UTC, datetime

import mlflow
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_model.config import AppConfig
from churn_model.exceptions import InferenceError, SchemaValidationError
from churn_model.features.transformations import handle_missing_values
from churn_model.logging_config import get_logger
from churn_model.models.registry import GitContext, load_model_by_alias

logger = get_logger(__name__)

_VECTOR_LIKE_COLUMNS = {"features", "features_raw", "rawPrediction", "probability"}


def score(
    config: AppConfig,
    df: DataFrame,
    model_alias: str | None = None,
    model_version: str | None = None,
) -> DataFrame:
    """Carrega o modelo (por alias, padrão Champion) e gera previsões padronizadas.

    Aplica a MESMA imputação usada no treino antes de pontuar, garantindo que
    a transformação não diverge entre treino e inferência.
    """
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.registry_uri)

    alias = model_alias or config.mlflow.registered_model_alias_champion
    required = list(config.features.numeric_columns) + list(config.features.categorical_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Colunas obrigatórias ausentes para inferência: {missing}")

    if df.rdd.isEmpty():
        raise InferenceError("Não é possível pontuar um DataFrame vazio.")

    prepared = handle_missing_values(
        df, list(config.features.numeric_columns), list(config.features.categorical_columns)
    )

    try:
        model = load_model_by_alias(config.mlflow.registered_model_name, alias)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            f"Falha ao carregar modelo '{config.mlflow.registered_model_name}@{alias}': {exc}"
        ) from exc

    try:
        predictions = model.transform(prepared)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Falha ao executar inferência: {exc}") from exc

    git_commit = GitContext.from_environment().commit
    scoring_timestamp = datetime.now(UTC).isoformat()

    probability_expr = (
        F.element_at(vector_to_array(F.col("probability")), 2)
        if "probability" in predictions.columns
        else F.lit(None).cast("double")
    )

    output_columns = [c for c in df.columns if c not in _VECTOR_LIKE_COLUMNS]
    result = (
        predictions.select(*output_columns, "prediction", probability_expr.alias("probability_churn"))
        .withColumn("model_alias", F.lit(alias))
        .withColumn("model_version", F.lit(model_version or "unknown"))
        .withColumn("scoring_timestamp", F.lit(scoring_timestamp))
        .withColumn("git_commit", F.lit(git_commit))
    )

    logger.info(
        "Inferência em lote concluída",
        extra={"extra_fields": {"model_alias": alias, "row_count": result.count()}},
    )
    return result
