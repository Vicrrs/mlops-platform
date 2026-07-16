"""Carregamento dos dados brutos de acordo com a fonte configurada por ambiente.

``dev``/local lê CSV com schema explícito de ``data/sample/``; ``hml``/``prd``
leem uma tabela do Unity Catalog (``catalog.schema.raw_table``). Nenhuma
inferência de schema é usada fora de ``dev``.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from churn_model.config import AppConfig
from churn_model.data.schemas import RAW_CUSTOMER_CHURN_SCHEMA
from churn_model.exceptions import ConfigurationError
from churn_model.io.readers import read_csv, read_table


def load_raw_customers(spark: SparkSession, config: AppConfig) -> DataFrame:
    if config.data.source == "local":
        if not config.data.raw_path:
            raise ConfigurationError("data.raw_path deve ser definido quando data.source='local'.")
        return read_csv(spark, config.data.raw_path, schema=RAW_CUSTOMER_CHURN_SCHEMA, header=True)

    if config.data.source == "unity_catalog":
        if not (config.data.catalog and config.data.schema and config.data.raw_table):
            raise ConfigurationError(
                "data.catalog, data.schema e data.raw_table devem ser definidos para data.source='unity_catalog'."
            )
        table_name = f"{config.data.catalog}.{config.data.schema}.{config.data.raw_table}"
        return read_table(spark, table_name)

    raise ConfigurationError(f"data.source desconhecido: '{config.data.source}'.")
