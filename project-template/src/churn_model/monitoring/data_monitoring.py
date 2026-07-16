"""Monitoramento de qualidade, schema e volume dos dados de entrada em produção."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame

from churn_model.config import AppConfig
from churn_model.data.quality import compute_basic_data_drift, run_data_quality_checks
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Alert:
    rule: str
    severity: str
    observed_value: object
    threshold: object
    model_name: str
    model_version: str
    environment: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


def monitor_incoming_data(
    df: DataFrame,
    config: AppConfig,
    reference_df: DataFrame | None,
    model_version: str,
    previous_row_count: int | None = None,
) -> tuple[dict, list[Alert]]:
    """Executa as regras de qualidade sobre um lote de produção e gera alertas para falhas."""
    report = run_data_quality_checks(df, config.data, previous_row_count=previous_row_count)

    alerts: list[Alert] = [
        Alert(
            rule=rule.name,
            severity=rule.severity.value,
            observed_value=rule.observed_value,
            threshold=rule.expected_value,
            model_name=config.mlflow.registered_model_name,
            model_version=model_version,
            environment=config.environment,
        )
        for rule in report.failed_rules()
    ]

    drift: dict = {}
    if reference_df is not None:
        drift = compute_basic_data_drift(df, reference_df, config.features.numeric_columns)
        for column, stats in drift.items():
            shift = stats.get("relative_mean_shift")
            if shift is not None and shift > 0.2:
                alerts.append(
                    Alert(
                        rule=f"data_drift[{column}]",
                        severity="warning",
                        observed_value=shift,
                        threshold=0.2,
                        model_name=config.mlflow.registered_model_name,
                        model_version=model_version,
                        environment=config.environment,
                    )
                )

    payload = {
        "quality_report": report.to_dict(),
        "data_drift": drift,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "Monitoramento de dados concluído",
        extra={"extra_fields": {"alert_count": len(alerts), "quality_passed": report.passed}},
    )
    return payload, alerts


def write_data_monitoring_report(payload: dict, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "data_monitoring.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target
