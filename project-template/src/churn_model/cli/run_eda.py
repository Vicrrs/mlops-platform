"""CLI: EDA automatizada (mapeia para ``resources/eda.job.yml``)."""

from __future__ import annotations

import sys

import mlflow

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.eda import run_eda, write_eda_artifacts
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError
from churn_model.logging_config import get_logger
from churn_model.models.registry import configure_mlflow, get_or_create_experiment

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser(
        "Executa a EDA automatizada sobre o dataset bruto e registra os artefatos no MLflow."
    )
    parser.add_argument("--output-dir", default="artifacts/eda")
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-eda")
    try:
        df = load_raw_customers(spark, config)
        summary = run_eda(
            df,
            required_columns=config.data.required_columns,
            numeric_columns=config.features.numeric_columns,
            categorical_columns=config.features.categorical_columns,
            target_column=config.features.label_column,
            date_columns=("signup_date",),
            dataset_name=config.project_name,
        )
        output_path = write_eda_artifacts(summary, args.output_dir)

        client = configure_mlflow(config)
        experiment_id = get_or_create_experiment(client, config.mlflow.experiment_name)
        with mlflow.start_run(
            experiment_id=experiment_id, tags={"stage": "eda", "project_name": config.project_name}
        ):
            mlflow.log_metric("row_count", summary["row_count"])
            mlflow.log_metric("column_count", summary["column_count"])
            for artifact_file in output_path.iterdir():
                mlflow.log_artifact(str(artifact_file), artifact_path="eda")

        logger.info("EDA concluída com sucesso.")
        return 0
    except ChurnModelError as exc:
        logger.error("Falha na EDA", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
