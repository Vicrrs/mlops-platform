"""Monitoramento da distribuição das previsões, latência e taxa de erro operacional."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_model.config import AppConfig
from churn_model.logging_config import get_logger
from churn_model.monitoring.data_monitoring import Alert

logger = get_logger(__name__)

MAX_EXPECTED_POSITIVE_RATE = 0.6
MIN_EXPECTED_POSITIVE_RATE = 0.01


def monitor_predictions(
    predictions_df: DataFrame,
    config: AppConfig,
    model_version: str,
    inference_latency_ms: float | None = None,
) -> tuple[dict, list[Alert]]:
    """Analisa a distribuição de previsões de um lote de escoragem já gravado."""
    total = predictions_df.count()
    alerts: list[Alert] = []

    if total == 0:
        payload = {"row_count": 0, "timestamp": datetime.now(UTC).isoformat()}
        return payload, alerts

    positive_count = predictions_df.where(F.col("prediction") == 1.0).count()
    positive_rate = round(positive_count / total, 4)
    null_predictions = predictions_df.where(F.col("prediction").isNull()).count()
    error_rate = round(null_predictions / total, 4)

    if positive_rate > MAX_EXPECTED_POSITIVE_RATE or positive_rate < MIN_EXPECTED_POSITIVE_RATE:
        alerts.append(
            Alert(
                rule="prediction_distribution",
                severity="warning",
                observed_value=positive_rate,
                threshold=f"[{MIN_EXPECTED_POSITIVE_RATE}, {MAX_EXPECTED_POSITIVE_RATE}]",
                model_name=config.mlflow.registered_model_name,
                model_version=model_version,
                environment=config.environment,
            )
        )
    if error_rate > 0.01:
        alerts.append(
            Alert(
                rule="prediction_error_rate",
                severity="error",
                observed_value=error_rate,
                threshold=0.01,
                model_name=config.mlflow.registered_model_name,
                model_version=model_version,
                environment=config.environment,
            )
        )

    payload = {
        "row_count": total,
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "error_rate": error_rate,
        "inference_latency_ms": inference_latency_ms,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    logger.info("Monitoramento de previsões concluído", extra={"extra_fields": payload})
    return payload, alerts


def write_prediction_monitoring_report(payload: dict, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "prediction_monitoring.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target
