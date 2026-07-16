"""Criação centralizada da SparkSession, local ou no Databricks."""

from __future__ import annotations

import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from churn_model.exceptions import SparkSessionError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)

_ACTIVE_SESSION: SparkSession | None = None


def is_running_on_databricks() -> bool:
    """Detecta execução em cluster/serverless Databricks via variáveis de ambiente."""
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION") or os.environ.get("DB_HOME"))


def get_spark_session(
    app_name: str,
    environment: str = "local",
    shuffle_partitions: int = 8,
    enable_delta: bool = True,
) -> SparkSession:
    """Retorna a SparkSession ativa, criando-a se necessário.

    No Databricks, o builder recupera a sessão já provisionada pelo cluster.
    Localmente, cria uma sessão ``local[*]`` com suporte a Delta Lake habilitado
    quando ``enable_delta`` é verdadeiro (necessário para os testes e para o
    modo ``dev`` funcionarem sem um workspace Databricks).

    Args:
        app_name: nome da aplicação Spark, usado para rastreabilidade nos logs.
        environment: ``local``, ``dev``, ``hml`` ou ``prd``.
        shuffle_partitions: número de partições de shuffle (baixo em dev/testes).
        enable_delta: habilita os pacotes/extensões Delta Lake quando fora do Databricks.

    Raises:
        SparkSessionError: quando a sessão não pode ser criada.
    """
    global _ACTIVE_SESSION

    if _ACTIVE_SESSION is not None:
        try:
            _ACTIVE_SESSION.sparkContext.getConf()
            return _ACTIVE_SESSION
        except Exception:  # noqa: BLE001 - sessão anterior morta, recriar
            _ACTIVE_SESSION = None

    try:
        if is_running_on_databricks():
            logger.info(
                "Reutilizando SparkSession do cluster Databricks",
                extra={"extra_fields": {"app_name": app_name, "environment": environment}},
            )
            _ACTIVE_SESSION = SparkSession.builder.appName(app_name).getOrCreate()
            return _ACTIVE_SESSION

        builder = (
            SparkSession.builder.appName(app_name)
            .master(os.environ.get("SPARK_MASTER", "local[*]"))
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.session.timeZone", "UTC")
        )

        if enable_delta:
            builder = builder.config(
                "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
            ).config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            builder = configure_spark_with_delta_pip(builder)

        logger.info(
            "Criando SparkSession local",
            extra={"extra_fields": {"app_name": app_name, "environment": environment}},
        )
        _ACTIVE_SESSION = builder.getOrCreate()
        _ACTIVE_SESSION.sparkContext.setLogLevel("WARN")
        return _ACTIVE_SESSION
    except Exception as exc:  # noqa: BLE001
        raise SparkSessionError(f"Falha ao criar SparkSession para '{app_name}': {exc}") from exc


def stop_spark_session() -> None:
    """Encerra a SparkSession ativa, se houver. Útil para testes e CLIs de curta duração."""
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is not None:
        _ACTIVE_SESSION.stop()
        _ACTIVE_SESSION = None
