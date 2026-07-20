"""CLI: treina o modelo candidato e o registra como Challenger (``resources/training.job.yml``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError
from churn_model.logging_config import get_logger
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import (
    GitContext,
    configure_mlflow,
    log_training_run,
    register_model_version,
    set_alias,
)
from churn_model.models.train import prepare_datasets, prepare_datasets_from_feature_store, train_candidate

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Treina o modelo candidato e registra a versão como Challenger no MLflow.")
    parser.add_argument("--output-dir", default="artifacts/training")
    parser.add_argument("--dataset-version", default="local-dev")
    parser.add_argument(
        "--use-feature-store",
        action="store_true",
        help="Monta o dataset de treino a partir da feature store (requer feature_store.enabled=true "
        "e a tabela já publicada via churn_model.cli.run_feature_engineering).",
    )
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-training")
    try:
        raw_df = load_raw_customers(spark, config)
        if args.use_feature_store:
            labels_df = raw_df.select(*config.feature_store.primary_keys, config.features.label_column)
            train_df, validation_df, test_df = prepare_datasets_from_feature_store(spark, config, labels_df)
        else:
            train_df, validation_df, test_df = prepare_datasets(raw_df, config)

        result = train_candidate(train_df, validation_df, test_df, config)
        validation_metrics = evaluate_model(
            result.pipeline_model, validation_df, config.features.label_column
        )

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "model_metrics.json").write_text(
            json.dumps(validation_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        feature_config_path = output_path / "feature_config.json"
        feature_config_path.write_text(
            json.dumps(result.feature_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        client = configure_mlflow(config)
        run_id = log_training_run(
            client=client,
            config=config,
            pipeline_model=result.pipeline_model,
            train_sample=train_df,
            metrics=validation_metrics,
            feature_config=result.feature_config,
            dataset_version=args.dataset_version,
            model_alias="challenger",
        )
        model_version = register_model_version(client, config.mlflow.registered_model_name, run_id)
        set_alias(
            client,
            config.mlflow.registered_model_name,
            config.mlflow.registered_model_alias_challenger,
            model_version,
        )

        git_context = GitContext.from_environment()
        run_metadata = {
            "project_name": config.project_name,
            "package_name": config.package_name,
            "environment": config.environment,
            "git_commit": git_context.commit,
            "git_branch": git_context.branch,
            "build_id": git_context.build_id,
            "build_number": git_context.build_number,
            "pipeline_template_version": __import__("os").environ.get(
                "MLOPS_PIPELINE_TEMPLATE_VERSION", "local"
            ),
            "dataset_version": args.dataset_version,
            "mlflow_experiment_name": config.mlflow.experiment_name,
            "mlflow_run_id": run_id,
            "registered_model_name": config.mlflow.registered_model_name,
            "model_version": model_version,
            "model_alias": "challenger",
            "training_timestamp": validation_metrics.get("timestamp", ""),
        }
        (output_path / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Também gravados como arquivos simples para consumo direto por steps de shell da pipeline.
        (output_path / "mlflow_run_id.txt").write_text(run_id, encoding="utf-8")
        (output_path / "model_version.txt").write_text(str(model_version), encoding="utf-8")
        (output_path / "commit_sha.txt").write_text(git_context.commit, encoding="utf-8")

        logger.info(
            "Modelo candidato treinado e registrado como Challenger",
            extra={"extra_fields": {"run_id": run_id, "model_version": model_version}},
        )
        return 0
    except ChurnModelError as exc:
        logger.error("Falha no treinamento do modelo candidato", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
