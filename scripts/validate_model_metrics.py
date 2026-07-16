#!/usr/bin/env python3
"""Valida as métricas do modelo candidato contra os limites absolutos mínimos.

Uso:
    python scripts/validate_model_metrics.py \
        --metrics-path artifacts/training/model_metrics.json \
        --minimum-accuracy 0.85 --minimum-precision 0.80 \
        --minimum-recall 0.80 --minimum-f1-score 0.82 --minimum-roc-auc 0.85
"""

from __future__ import annotations

import argparse
import sys

from _lib import fail, get_logger, read_json, succeed

logger = get_logger(__name__)

_THRESHOLD_ARGS = {
    "accuracy": "minimum_accuracy",
    "precision": "minimum_precision",
    "recall": "minimum_recall",
    "f1_score": "minimum_f1_score",
    "roc_auc": "minimum_roc_auc",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--minimum-accuracy", type=float, default=0.0)
    parser.add_argument("--minimum-precision", type=float, default=0.0)
    parser.add_argument("--minimum-recall", type=float, default=0.0)
    parser.add_argument("--minimum-f1-score", type=float, default=0.0)
    parser.add_argument("--minimum-roc-auc", type=float, default=0.0)
    parser.add_argument(
        "--maximum-invalid-prediction-percentage",
        type=float,
        default=0.01,
        help="Percentual máximo de previsões inválidas tolerado.",
    )
    args = parser.parse_args(argv)

    try:
        metrics = read_json(args.metrics_path)
    except FileNotFoundError:
        return fail(logger, "Arquivo de métricas não encontrado", metrics_path=args.metrics_path)

    thresholds = {
        "accuracy": args.minimum_accuracy,
        "precision": args.minimum_precision,
        "recall": args.minimum_recall,
        "f1_score": args.minimum_f1_score,
        "roc_auc": args.minimum_roc_auc,
    }

    failures = []
    for metric_name, minimum in thresholds.items():
        observed = metrics.get(metric_name)
        if observed is None:
            failures.append(f"{metric_name}: ausente no relatório de métricas")
        elif observed < minimum:
            failures.append(f"{metric_name}: observado={observed} < mínimo={minimum}")

    invalid_pct = metrics.get("invalid_prediction_percentage", 0.0)
    if invalid_pct is not None and invalid_pct > args.maximum_invalid_prediction_percentage:
        failures.append(
            f"invalid_prediction_percentage: observado={invalid_pct} > máximo={args.maximum_invalid_prediction_percentage}"
        )

    if failures:
        return fail(logger, "Métricas do modelo abaixo dos limites mínimos", failures=failures)

    return succeed(logger, "Métricas do modelo aprovadas nos limites mínimos", metrics=metrics)


if __name__ == "__main__":
    sys.exit(main())
