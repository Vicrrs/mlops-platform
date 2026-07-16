"""Pontos de entrada de linha de comando, um por job Databricks (``resources/*.job.yml``).

Cada módulo é executável via ``python -m churn_model.cli.<nome> --config conf/<env>.yml``
e mapeia 1:1 para um job do Databricks Asset Bundle, mantendo toda a lógica em
código testável (não em notebook).
"""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from churn_model.config import AppConfig, load_config
from churn_model.logging_config import bind_context, get_logger
from churn_model.spark import get_spark_session, stop_spark_session

logger = get_logger(__name__)


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Caminho para o YAML de ambiente (ex.: conf/dev.yml)")
    return parser


def bootstrap(config_path: str, app_name: str) -> tuple[AppConfig, SparkSession]:
    config = load_config(config_path)
    bind_context(project_name=config.project_name, environment=config.environment)
    spark = get_spark_session(
        app_name=app_name,
        environment=config.environment,
        shuffle_partitions=config.spark.shuffle_partitions,
    )
    return config, spark


def shutdown() -> None:
    stop_spark_session()
