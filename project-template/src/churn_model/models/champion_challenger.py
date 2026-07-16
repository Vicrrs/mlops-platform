"""Comparação Champion vs. Challenger no mesmo dataset de teste.

Champion e Challenger são pontuados sobre exatamente o mesmo DataFrame de
teste (mesmos registros), garantindo comparação justa. O resultado técnico
(``technical_approval``) é calculado aqui a partir de limites absolutos e do
limite de regressão máxima permitida -- a decisão final de promoção ainda
depende de aprovação manual na pipeline (ver ``docs/champion-challenger.md``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mlflow.tracking import MlflowClient
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from churn_model.config import AppConfig, ModelThresholds
from churn_model.exceptions import ChallengerNotApprovedError
from churn_model.logging_config import get_logger
from churn_model.models import registry as registry_module
from churn_model.models.evaluate import evaluate_model

logger = get_logger(__name__)


@dataclass
class ComparisonContext:
    project_name: str
    registered_model_name: str
    primary_metric: str
    champion_version: str | None
    challenger_version: str
    champion_run_id: str | None
    challenger_run_id: str
    git_commit: str
    dataset_version: str


def compute_promotion_recommendation(
    champion_metrics: dict | None,
    challenger_metrics: dict,
    thresholds: ModelThresholds,
    context: ComparisonContext,
) -> dict:
    """Aplica as regras de aprovação técnica e produz o dicionário de recomendação.

    Regras (seção 14.3 do padrão da plataforma):
        challenger_metric >= minimum_<metric>  (todos os limites absolutos)
        challenger_metric >= champion_metric - maximum_metric_regression (quando há champion)
    """
    primary_metric = context.primary_metric
    is_first_model = champion_metrics is None

    absolute_checks = {
        "accuracy": challenger_metrics["accuracy"] >= thresholds.minimum_accuracy,
        "precision": challenger_metrics["precision"] >= thresholds.minimum_precision,
        "recall": challenger_metrics["recall"] >= thresholds.minimum_recall,
        "f1_score": challenger_metrics["f1_score"] >= thresholds.minimum_f1_score,
        "roc_auc": challenger_metrics["roc_auc"] >= thresholds.minimum_roc_auc,
    }
    absolute_approval = all(absolute_checks.values())

    champion_metric_value = champion_metrics[primary_metric] if champion_metrics else None
    challenger_metric_value = challenger_metrics[primary_metric]

    regression_approval = True
    if champion_metric_value is not None:
        regression_approval = (
            challenger_metric_value >= champion_metric_value - thresholds.maximum_metric_regression
        )

    technical_approval = absolute_approval and regression_approval

    absolute_improvement = (
        round(challenger_metric_value - champion_metric_value, 6)
        if champion_metric_value is not None
        else None
    )
    relative_improvement = (
        round(absolute_improvement / champion_metric_value, 6)
        if (champion_metric_value not in (None, 0) and absolute_improvement is not None)
        else None
    )

    recommendation = "promote" if technical_approval else "reject"

    return {
        "project_name": context.project_name,
        "registered_model_name": context.registered_model_name,
        "champion_version": context.champion_version,
        "challenger_version": context.challenger_version,
        "champion_run_id": context.champion_run_id,
        "challenger_run_id": context.challenger_run_id,
        "primary_metric": primary_metric,
        "champion_metric": champion_metric_value,
        "challenger_metric": challenger_metric_value,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "is_first_model": is_first_model,
        "absolute_threshold_checks": absolute_checks,
        "regression_within_limit": regression_approval,
        "technical_approval": technical_approval,
        "recommendation": recommendation,
        "git_commit": context.git_commit,
        "dataset_version": context.dataset_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def render_comparison_report_markdown(
    champion_metrics: dict | None, challenger_metrics: dict, recommendation: dict
) -> str:
    lines = [
        f"# Comparação Champion vs. Challenger — {recommendation['project_name']}",
        "",
        f"- Modelo: `{recommendation['registered_model_name']}`",
        f"- Métrica principal: **{recommendation['primary_metric']}**",
        f"- Champion: versão `{recommendation['champion_version']}` (run `{recommendation['champion_run_id']}`)",
        f"- Challenger: versão `{recommendation['challenger_version']}` (run `{recommendation['challenger_run_id']}`)",
        "",
        "## Métricas",
        "",
        "| Métrica | Champion | Challenger |",
        "|---|---|---|",
    ]
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "latency_ms",
        "model_size_bytes",
    ]
    for metric in metric_names:
        champ_val = champion_metrics.get(metric) if champion_metrics else "—"
        chal_val = challenger_metrics.get(metric, "—")
        lines.append(f"| {metric} | {champ_val} | {chal_val} |")

    lines += [
        "",
        f"**Aprovação técnica:** {'✅ Sim' if recommendation['technical_approval'] else '❌ Não'}",
        f"**Recomendação:** `{recommendation['recommendation']}`",
        "",
        "Esta recomendação é técnica; a promoção ainda depende de aprovação manual.",
    ]
    return "\n".join(lines)


def run_champion_challenger_comparison(
    client: MlflowClient,
    config: AppConfig,
    challenger_model: PipelineModel,
    challenger_run_id: str,
    challenger_version: str,
    test_df: DataFrame,
    dataset_version: str,
    output_dir: str | Path,
) -> dict:
    """Executa a comparação completa e grava os 5 artefatos exigidos pela plataforma."""
    from churn_model.models.registry import GitContext

    champion_version_info = registry_module.get_model_version_by_alias(
        client, config.mlflow.registered_model_name, config.mlflow.registered_model_alias_champion
    )

    champion_metrics: dict | None = None
    champion_run_id: str | None = None
    champion_version: str | None = None
    if champion_version_info is not None:
        champion_model = registry_module.load_model_by_alias(
            config.mlflow.registered_model_name, config.mlflow.registered_model_alias_champion
        )
        champion_metrics = evaluate_model(champion_model, test_df, config.features.label_column)
        champion_run_id = champion_version_info.run_id
        champion_version = champion_version_info.version
    else:
        logger.info("Nenhum Champion encontrado -- Challenger tratado como primeiro modelo.")

    challenger_metrics = evaluate_model(challenger_model, test_df, config.features.label_column)

    context = ComparisonContext(
        project_name=config.project_name,
        registered_model_name=config.mlflow.registered_model_name,
        primary_metric=config.model.primary_metric,
        champion_version=champion_version,
        challenger_version=challenger_version,
        champion_run_id=champion_run_id,
        challenger_run_id=challenger_run_id,
        git_commit=GitContext.from_environment().commit,
        dataset_version=dataset_version,
    )
    recommendation = compute_promotion_recommendation(
        champion_metrics, challenger_metrics, config.model.thresholds, context
    )
    comparison = {
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "context": recommendation,
    }
    report_md = render_comparison_report_markdown(champion_metrics, challenger_metrics, recommendation)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "champion_metrics.json").write_text(
        json.dumps(champion_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "challenger_metrics.json").write_text(
        json.dumps(challenger_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "comparison_report.md").write_text(report_md, encoding="utf-8")
    (output_path / "promotion_recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Comparação Champion/Challenger concluída",
        extra={
            "extra_fields": {
                "technical_approval": recommendation["technical_approval"],
                "recommendation": recommendation["recommendation"],
            }
        },
    )
    return recommendation


def enforce_technical_approval(recommendation: dict) -> None:
    if not recommendation["technical_approval"]:
        raise ChallengerNotApprovedError(
            f"Challenger não aprovado tecnicamente: {recommendation['recommendation']} "
            f"({recommendation['primary_metric']}={recommendation['challenger_metric']})"
        )
