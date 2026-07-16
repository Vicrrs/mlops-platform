"""CLI: smoke test do modelo Champion pós-promoção (``resources/smoke-test.job.yml``)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError, InferenceError
from churn_model.logging_config import get_logger
from churn_model.models.inference import score

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Executa um smoke test: carrega o Champion e faz uma inferência pequena.")
    parser.add_argument("--model-alias", default=None, help="Padrão: champion.")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/smoke_test")
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-smoke-test")
    alias = args.model_alias or config.mlflow.registered_model_alias_champion
    report: dict = {
        "model_alias": alias,
        "registered_model_name": config.mlflow.registered_model_name,
        "environment": config.environment,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        sample_df = load_raw_customers(spark, config).limit(args.sample_size)
        if sample_df.count() == 0:
            raise InferenceError("Amostra de smoke test vazia -- não há dados para pontuar.")

        predictions_df = score(config, sample_df, model_alias=alias)
        row_count = predictions_df.count()
        expected_columns = {
            "prediction",
            "probability_churn",
            "model_alias",
            "model_version",
            "scoring_timestamp",
        }
        missing_columns = expected_columns - set(predictions_df.columns)
        null_predictions = predictions_df.where(predictions_df["prediction"].isNull()).count()

        passed = row_count > 0 and not missing_columns and null_predictions == 0
        report.update(
            {
                "passed": passed,
                "row_count": row_count,
                "missing_columns": sorted(missing_columns),
                "null_predictions": null_predictions,
            }
        )

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "smoke_test_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if not passed:
            logger.error("Smoke test reprovado", extra={"extra_fields": report})
            return 1

        logger.info("Smoke test aprovado", extra={"extra_fields": report})
        return 0
    except ChurnModelError as exc:
        report.update({"passed": False, "error": str(exc)})
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "smoke_test_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.error("Smoke test falhou com erro", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
