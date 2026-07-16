"""CLI: valida a qualidade dos dados brutos (mapeia para ``resources/data-quality.job.yml``)."""

from __future__ import annotations

import json
import sys

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.data.quality import (
    enforce_quality_gate,
    run_data_quality_checks,
    write_data_quality_artifacts,
)
from churn_model.exceptions import ChurnModelError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Executa as regras de qualidade de dados sobre o dataset bruto.")
    parser.add_argument("--output-dir", default="artifacts/data_quality")
    parser.add_argument(
        "--previous-row-count-file",
        default=None,
        help="Arquivo JSON opcional com {'row_count': N} da execução anterior, para checar variação de volume.",
    )
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-data-quality")
    try:
        df = load_raw_customers(spark, config)

        previous_row_count = None
        if args.previous_row_count_file:
            try:
                previous_row_count = json.loads(open(args.previous_row_count_file, encoding="utf-8").read())[
                    "row_count"
                ]
            except FileNotFoundError:
                previous_row_count = None

        report = run_data_quality_checks(df, config.data, previous_row_count=previous_row_count)
        write_data_quality_artifacts(report, args.output_dir)
        enforce_quality_gate(report)
        logger.info("Quality gate de dados aprovado.")
        return 0
    except ChurnModelError as exc:
        logger.error("Quality gate de dados reprovado", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
