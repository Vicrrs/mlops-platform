"""Integração com MLflow Tracking e Model Registry (Unity Catalog / aliases).

Nunca inventa ``run_id`` ou ``model_version`` -- ambos vêm sempre da execução
real do MLflow. Usa exclusivamente aliases (``champion``/``challenger``/
``previous_champion``), nunca os antigos estágios Staging/Production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from churn_model.config import AppConfig
from churn_model.exceptions import ChampionNotFoundError, ModelRegistryError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GitContext:
    commit: str
    branch: str
    build_id: str
    build_number: str

    @classmethod
    def from_environment(cls) -> GitContext:
        return cls(
            commit=os.environ.get("BUILD_SOURCEVERSION", "local"),
            branch=os.environ.get("BUILD_SOURCEBRANCH", "local"),
            build_id=os.environ.get("BUILD_BUILDID", "local"),
            build_number=os.environ.get("BUILD_BUILDNUMBER", "local"),
        )


def configure_mlflow(config: AppConfig) -> MlflowClient:
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.registry_uri)
    return MlflowClient(tracking_uri=config.mlflow.tracking_uri, registry_uri=config.mlflow.registry_uri)


def get_or_create_experiment(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(experiment_name)
    except MlflowException as exc:
        raise ModelRegistryError(f"Falha ao criar/obter experimento '{experiment_name}': {exc}") from exc


def build_run_tags(
    config: AppConfig,
    git_context: GitContext,
    model_alias: str,
    dataset_version: str,
) -> dict[str, str]:
    return {
        "project_name": config.project_name,
        "environment": config.environment,
        "git_commit": git_context.commit,
        "git_branch": git_context.branch,
        "azure_build_id": git_context.build_id,
        "azure_build_number": git_context.build_number,
        "package_version": _package_version(),
        "dataset_version": dataset_version,
        "model_name": config.mlflow.registered_model_name,
        "model_framework": config.model.type,
        "pipeline_version": os.environ.get("MLOPS_PIPELINE_TEMPLATE_VERSION", "local"),
        "training_timestamp": datetime.now(UTC).isoformat(),
        "model_alias": model_alias,
    }


def _package_version() -> str:
    from churn_model import __version__

    return __version__


def log_training_run(
    client: MlflowClient,
    config: AppConfig,
    pipeline_model: PipelineModel,
    train_sample: DataFrame,
    metrics: dict,
    feature_config: dict,
    dataset_version: str,
    extra_artifacts: dict[str, str] | None = None,
    model_alias: str = "challenger",
) -> str:
    """Executa uma run de treino completa no MLflow e retorna o ``run_id`` real.

    Registra: parâmetros, métricas, tags, schema/EDA/qualidade (se fornecidos em
    ``extra_artifacts``), o modelo (com ``input_example`` e assinatura) e as
    dependências. O ``run_id`` retornado nunca é sintético.
    """
    experiment_id = get_or_create_experiment(client, config.mlflow.experiment_name)
    git_context = GitContext.from_environment()
    tags = build_run_tags(config, git_context, model_alias, dataset_version)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.registry_uri)

    with mlflow.start_run(experiment_id=experiment_id, tags=tags) as run:
        mlflow.log_params(
            {
                "algorithm": config.model.algorithm,
                "features_version": feature_config["version"],
                **{f"hp_{k}": v for k, v in config.model.hyperparameters.items()},
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict(feature_config, "feature_config.json")

        for artifact_name, artifact_path in (extra_artifacts or {}).items():
            try:
                mlflow.log_artifact(artifact_path, artifact_path=artifact_name)
            except OSError as exc:
                logger.warning(
                    "Falha ao anexar artefato ao run",
                    extra={"extra_fields": {"artifact_name": artifact_name, "error": str(exc)}},
                )

        input_example = train_sample.limit(5).toPandas()
        predictions_sample = pipeline_model.transform(train_sample.limit(5))
        output_example = predictions_sample.select("prediction").limit(5).toPandas()
        signature = infer_signature(input_example, output_example)

        mlflow.spark.log_model(
            pipeline_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        run_id = run.info.run_id

    logger.info(
        "Run de treinamento registrada no MLflow",
        extra={"extra_fields": {"run_id": run_id, "experiment_id": experiment_id}},
    )
    return run_id


def register_model_version(client: MlflowClient, registered_model_name: str, run_id: str) -> str:
    """Cria (se necessário) o modelo registrado e uma nova versão a partir do run real."""
    try:
        client.get_registered_model(registered_model_name)
    except MlflowException:
        client.create_registered_model(registered_model_name)

    model_uri = f"runs:/{run_id}/model"
    try:
        model_version = client.create_model_version(
            name=registered_model_name, source=model_uri, run_id=run_id
        )
    except MlflowException as exc:
        raise ModelRegistryError(
            f"Falha ao registrar versão do modelo '{registered_model_name}' a partir de {model_uri}: {exc}"
        ) from exc

    logger.info(
        "Nova versão de modelo registrada",
        extra={
            "extra_fields": {"registered_model_name": registered_model_name, "version": model_version.version}
        },
    )
    return model_version.version


def set_alias(client: MlflowClient, registered_model_name: str, alias: str, version: str) -> None:
    try:
        client.set_registered_model_alias(registered_model_name, alias, version)
    except MlflowException as exc:
        raise ModelRegistryError(f"Falha ao definir alias '{alias}' para versão {version}: {exc}") from exc
    logger.info(
        "Alias de modelo atualizado",
        extra={
            "extra_fields": {
                "registered_model_name": registered_model_name,
                "alias": alias,
                "version": version,
            }
        },
    )


def get_model_version_by_alias(client: MlflowClient, registered_model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(registered_model_name, alias)
    except MlflowException:
        return None


def require_model_version_by_alias(client: MlflowClient, registered_model_name: str, alias: str):
    version = get_model_version_by_alias(client, registered_model_name, alias)
    if version is None:
        raise ChampionNotFoundError(
            f"Nenhuma versão com alias '{alias}' encontrada para '{registered_model_name}'."
        )
    return version


def load_model_by_alias(registered_model_name: str, alias: str) -> PipelineModel:
    model_uri = f"models:/{registered_model_name}@{alias}"
    try:
        return mlflow.spark.load_model(model_uri)
    except MlflowException as exc:
        raise ModelRegistryError(f"Falha ao carregar modelo '{model_uri}': {exc}") from exc
