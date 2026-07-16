"""CLI: inferência em lote usando o modelo registrado (``resources/batch-inference.job.yml``)."""

from __future__ import annotations

import sys

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError
from churn_model.io.writers import write_delta, write_table
from churn_model.logging_config import get_logger
from churn_model.models.inference import score

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser(
        "Executa inferência em lote com o modelo Champion (ou outro alias) e grava o resultado."
    )
    parser.add_argument("--model-alias", default=None, help="Alias do modelo a usar (padrão: champion).")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--output-path", default="artifacts/predictions")
    parser.add_argument(
        "--output-table", default=None, help="Tabela Unity Catalog de destino (ambientes hml/prd)."
    )
    parser.add_argument("--mode", default="overwrite", choices=["append", "overwrite", "errorifexists"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-batch-inference")
    try:
        input_df = load_raw_customers(spark, config)
        predictions_df = score(
            config, input_df, model_alias=args.model_alias, model_version=args.model_version
        )

        if args.output_table:
            write_table(
                predictions_df,
                args.output_table,
                mode=args.mode,
                environment=config.environment,
                dry_run=args.dry_run,
            )
        else:
            write_delta(
                predictions_df,
                args.output_path,
                mode=args.mode,
                environment=config.environment,
                dry_run=args.dry_run,
            )

        logger.info("Inferência em lote concluída com sucesso.")
        return 0
    except ChurnModelError as exc:
        logger.error("Falha na inferência em lote", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
