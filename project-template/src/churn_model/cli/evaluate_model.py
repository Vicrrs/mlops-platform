"""CLI: compara Challenger vs. Champion no mesmo dataset de teste (``resources/model-validation.job.yml``)."""

from __future__ import annotations

import sys

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError
from churn_model.logging_config import get_logger
from churn_model.models.champion_challenger import (
    enforce_technical_approval,
    run_champion_challenger_comparison,
)
from churn_model.models.registry import configure_mlflow, load_model_by_alias
from churn_model.models.train import prepare_datasets

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Compara o Challenger com o Champion atual sobre o mesmo dataset de teste.")
    parser.add_argument("--output-dir", default="artifacts/model_comparison")
    parser.add_argument("--dataset-version", default="local-dev")
    parser.add_argument(
        "--challenger-alias",
        default=None,
        help="Alias a avaliar como Challenger (padrão: o configurado em mlflow.registered_model_alias_challenger).",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Retorna código de saída 1 quando a aprovação técnica falhar (usado como quality gate).",
    )
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-model-validation")
    try:
        raw_df = load_raw_customers(spark, config)
        _, _, test_df = prepare_datasets(raw_df, config)

        client = configure_mlflow(config)
        challenger_alias = args.challenger_alias or config.mlflow.registered_model_alias_challenger
        challenger_version_info = client.get_model_version_by_alias(
            config.mlflow.registered_model_name, challenger_alias
        )
        challenger_model = load_model_by_alias(config.mlflow.registered_model_name, challenger_alias)

        recommendation = run_champion_challenger_comparison(
            client=client,
            config=config,
            challenger_model=challenger_model,
            challenger_run_id=challenger_version_info.run_id,
            challenger_version=challenger_version_info.version,
            test_df=test_df,
            dataset_version=args.dataset_version,
            output_dir=args.output_dir,
        )

        logger.info(
            "Comparação Champion/Challenger publicada",
            extra={"extra_fields": {"recommendation": recommendation["recommendation"]}},
        )

        if args.enforce:
            enforce_technical_approval(recommendation)
        return 0
    except ChurnModelError as exc:
        logger.error("Falha na avaliação do candidato", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
