#!/usr/bin/env python3
"""Recalcula a recomendação de promoção Champion vs. Challenger (governança central).

Esta é a lógica reutilizável e independente de projeto que decide se um
Challenger é tecnicamente aprovado: aplica limites absolutos e o limite de
regressão máxima permitida sobre métricas JÁ calculadas pelo job Spark do
projeto (``champion_metrics.json`` / ``challenger_metrics.json``), sem precisar
de PySpark neste passo -- por isso pode rodar em qualquer agente Azure DevOps.

Uso:
    python scripts/compare_champion_challenger.py \
        --challenger-metrics-path artifacts/model_comparison/challenger_metrics.json \
        --champion-metrics-path artifacts/model_comparison/champion_metrics.json \
        --primary-metric f1_score \
        --minimum-accuracy 0.85 --minimum-precision 0.80 --minimum-recall 0.80 \
        --minimum-f1-score 0.82 --minimum-roc-auc 0.85 \
        --maximum-metric-regression 0.01 \
        --project-name ml-fraude --registered-model-name prd_catalog.fraude.modelo_fraude \
        --champion-version 4 --challenger-version 5 \
        --champion-run-id <run_id> --challenger-run-id <run_id> \
        --git-commit <sha> --dataset-version v3 \
        --output-dir artifacts/model_comparison
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from _lib import fail, get_logger, read_json, succeed, write_json

logger = get_logger(__name__)


def compute_recommendation(
    champion_metrics: dict | None,
    challenger_metrics: dict,
    thresholds: dict[str, float],
    maximum_metric_regression: float,
    primary_metric: str,
    context: dict,
) -> dict:
    absolute_checks = {
        metric: challenger_metrics[metric] >= minimum
        for metric, minimum in thresholds.items()
        if metric in challenger_metrics
    }
    absolute_approval = all(absolute_checks.values()) if absolute_checks else False

    champion_metric_value = champion_metrics.get(primary_metric) if champion_metrics else None
    challenger_metric_value = challenger_metrics[primary_metric]

    regression_approval = True
    if champion_metric_value is not None:
        regression_approval = challenger_metric_value >= champion_metric_value - maximum_metric_regression

    technical_approval = absolute_approval and regression_approval

    absolute_improvement = (
        round(challenger_metric_value - champion_metric_value, 6) if champion_metric_value is not None else None
    )
    relative_improvement = (
        round(absolute_improvement / champion_metric_value, 6)
        if (champion_metric_value not in (None, 0) and absolute_improvement is not None)
        else None
    )

    return {
        **context,
        "primary_metric": primary_metric,
        "champion_metric": champion_metric_value,
        "challenger_metric": challenger_metric_value,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "is_first_model": champion_metrics is None,
        "absolute_threshold_checks": absolute_checks,
        "regression_within_limit": regression_approval,
        "technical_approval": technical_approval,
        "recommendation": "promote" if technical_approval else "reject",
        "timestamp": datetime.now(UTC).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger-metrics-path", required=True)
    parser.add_argument("--champion-metrics-path", default=None, help="Omitir quando não houver Champion ainda.")
    parser.add_argument("--primary-metric", default="f1_score")
    parser.add_argument("--minimum-accuracy", type=float, default=0.0)
    parser.add_argument("--minimum-precision", type=float, default=0.0)
    parser.add_argument("--minimum-recall", type=float, default=0.0)
    parser.add_argument("--minimum-f1-score", type=float, default=0.0)
    parser.add_argument("--minimum-roc-auc", type=float, default=0.0)
    parser.add_argument("--maximum-metric-regression", type=float, default=0.01)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--champion-version", default=None)
    parser.add_argument("--challenger-version", required=True)
    parser.add_argument("--champion-run-id", default=None)
    parser.add_argument("--challenger-run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", default="artifacts/model_comparison")
    parser.add_argument(
        "--allow-reject",
        action="store_true",
        help="Não retorna código de saída 1 quando a recomendação for 'reject' (apenas grava o resultado).",
    )
    args = parser.parse_args(argv)

    try:
        challenger_metrics = read_json(args.challenger_metrics_path)
    except FileNotFoundError:
        return fail(logger, "Métricas do Challenger não encontradas", path=args.challenger_metrics_path)

    champion_metrics = None
    if args.champion_metrics_path:
        try:
            champion_metrics = read_json(args.champion_metrics_path)
        except FileNotFoundError:
            return fail(logger, "Métricas do Champion não encontradas", path=args.champion_metrics_path)

    thresholds = {
        "accuracy": args.minimum_accuracy,
        "precision": args.minimum_precision,
        "recall": args.minimum_recall,
        "f1_score": args.minimum_f1_score,
        "roc_auc": args.minimum_roc_auc,
    }
    context = {
        "project_name": args.project_name,
        "registered_model_name": args.registered_model_name,
        "champion_version": args.champion_version,
        "challenger_version": args.challenger_version,
        "champion_run_id": args.champion_run_id,
        "challenger_run_id": args.challenger_run_id,
        "git_commit": args.git_commit,
        "dataset_version": args.dataset_version,
    }

    recommendation = compute_recommendation(
        champion_metrics, challenger_metrics, thresholds, args.maximum_metric_regression, args.primary_metric, context
    )

    write_json(f"{args.output_dir}/promotion_recommendation.json", recommendation)
    write_json(
        f"{args.output_dir}/comparison.json",
        {"champion_metrics": champion_metrics, "challenger_metrics": challenger_metrics, "context": recommendation},
    )

    if not recommendation["technical_approval"] and not args.allow_reject:
        return fail(
            logger,
            "Challenger não aprovado tecnicamente",
            recommendation=recommendation["recommendation"],
            primary_metric=args.primary_metric,
            challenger_metric=recommendation["challenger_metric"],
            champion_metric=recommendation["champion_metric"],
        )

    return succeed(
        logger,
        "Comparação Champion/Challenger concluída",
        recommendation=recommendation["recommendation"],
        technical_approval=recommendation["technical_approval"],
    )


if __name__ == "__main__":
    sys.exit(main())
