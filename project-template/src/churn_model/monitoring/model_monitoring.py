"""Monitoramento de performance real do modelo (quando labels ficam disponíveis)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame

from churn_model.config import AppConfig
from churn_model.logging_config import get_logger
from churn_model.models.evaluate import evaluate_binary_classifier
from churn_model.models.registry import get_model_version_by_alias, load_model_by_alias
from churn_model.monitoring.data_monitoring import Alert

logger = get_logger(__name__)


def monitor_model_performance(
    client: MlflowClient,
    config: AppConfig,
    labeled_df: DataFrame,
    model_alias: str | None = None,
) -> tuple[dict, list[Alert]]:
    """Recalcula métricas reais com labels disponíveis e compara com os limites mínimos.

    Também compara Champion vs. Challenger (se ambos existirem) usando o mesmo
    lote rotulado, sinalizando quando o Challenger em produção supera o Champion.
    """
    alias = model_alias or config.mlflow.registered_model_alias_champion
    version_info = get_model_version_by_alias(client, config.mlflow.registered_model_name, alias)
    alerts: list[Alert] = []

    if version_info is None or labeled_df.rdd.isEmpty():
        payload = {
            "evaluated": False,
            "reason": "sem modelo com alias configurado ou sem dados rotulados",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return payload, alerts

    model = load_model_by_alias(config.mlflow.registered_model_name, alias)
    predictions = model.transform(labeled_df)
    metrics = evaluate_binary_classifier(predictions, config.features.label_column)

    thresholds = config.model.thresholds
    checks = {
        "accuracy": (metrics["accuracy"], thresholds.minimum_accuracy),
        "precision": (metrics["precision"], thresholds.minimum_precision),
        "recall": (metrics["recall"], thresholds.minimum_recall),
        "f1_score": (metrics["f1_score"], thresholds.minimum_f1_score),
        "roc_auc": (metrics["roc_auc"], thresholds.minimum_roc_auc),
    }
    for metric_name, (observed, minimum) in checks.items():
        if observed < minimum:
            alerts.append(
                Alert(
                    rule=f"model_performance_degradation[{metric_name}]",
                    severity="critical",
                    observed_value=observed,
                    threshold=minimum,
                    model_name=config.mlflow.registered_model_name,
                    model_version=version_info.version,
                    environment=config.environment,
                )
            )

    payload = {
        "evaluated": True,
        "model_alias": alias,
        "model_version": version_info.version,
        "metrics": metrics,
        "thresholds": {k: v[1] for k, v in checks.items()},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "Monitoramento de performance concluído", extra={"extra_fields": {"alert_count": len(alerts)}}
    )
    return payload, alerts


def write_model_performance_report(payload: dict, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "model_performance.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target


def write_alerts(alerts: list[Alert], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "alerts.json"
    target.write_text(
        json.dumps([alert.__dict__ for alert in alerts], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Alertas de monitoramento gravados", extra={"extra_fields": {"count": len(alerts)}})
    return target
