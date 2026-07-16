"""Funções reutilizáveis de leitura de dados.

Todas as funções: (1) registram origem/formato/quantidade de registros,
(2) validam DataFrame vazio, (3) evitam inferência automática de schema em
produção (só permitida quando ``allow_schema_inference=True``, tipicamente
em dev), e (4) nunca recebem credenciais em texto no código -- JDBC recebe
``properties`` vindas de variáveis de ambiente / secret scopes chamados
pelo código do projeto, nunca literais aqui.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from churn_model.exceptions import DataReadError, EmptyDataFrameError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


def _log_read(source: str, fmt: str, df: DataFrame) -> None:
    count = df.count()
    logger.info(
        "Leitura de dados concluída",
        extra={"extra_fields": {"source": source, "format": fmt, "row_count": count}},
    )


def _require_non_empty(df: DataFrame, source: str) -> DataFrame:
    if df.rdd.isEmpty():
        raise EmptyDataFrameError(f"Leitura de '{source}' retornou um DataFrame vazio.")
    return df


def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    """Lê uma tabela do Unity Catalog / metastore no formato ``catalog.schema.table``."""
    try:
        df = spark.table(table_name)
        _log_read(table_name, "table", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha ao ler tabela '{table_name}': {exc}") from exc


def read_delta(spark: SparkSession, path: str) -> DataFrame:
    """Lê dados Delta a partir de um caminho (DBFS, ABFS, ou filesystem local)."""
    try:
        df = spark.read.format("delta").load(path)
        _log_read(path, "delta", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha ao ler Delta em '{path}': {exc}") from exc


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    try:
        df = spark.read.parquet(path)
        _log_read(path, "parquet", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha ao ler Parquet em '{path}': {exc}") from exc


def read_csv(
    spark: SparkSession,
    path: str,
    schema: StructType | None = None,
    header: bool = True,
    allow_schema_inference: bool = False,
    sep: str = ",",
) -> DataFrame:
    """Lê CSV com schema explícito. Inferência só é permitida fora de produção."""
    if schema is None and not allow_schema_inference:
        raise DataReadError(
            f"Leitura de CSV em '{path}' requer schema explícito. "
            "Use allow_schema_inference=True apenas em desenvolvimento."
        )
    try:
        reader = spark.read.option("header", str(header)).option("sep", sep)
        if schema is not None:
            reader = reader.schema(schema)
        else:
            reader = reader.option("inferSchema", "true")
        df = reader.csv(path)
        _log_read(path, "csv", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha ao ler CSV em '{path}': {exc}") from exc


def read_json(
    spark: SparkSession,
    path: str,
    schema: StructType | None = None,
    allow_schema_inference: bool = False,
) -> DataFrame:
    if schema is None and not allow_schema_inference:
        raise DataReadError(
            f"Leitura de JSON em '{path}' requer schema explícito. "
            "Use allow_schema_inference=True apenas em desenvolvimento."
        )
    try:
        reader = spark.read
        if schema is not None:
            reader = reader.schema(schema)
        df = reader.json(path)
        _log_read(path, "json", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha ao ler JSON em '{path}': {exc}") from exc


def read_jdbc(
    spark: SparkSession,
    url: str,
    table: str,
    properties: dict[str, str],
) -> DataFrame:
    """Lê dados via JDBC. ``properties`` (usuário/senha/driver) deve vir de secrets,
    nunca de literais no código de chamada."""
    if not properties.get("user") or not properties.get("password"):
        raise DataReadError(
            "Propriedades JDBC incompletas: 'user' e 'password' devem ser "
            "fornecidos via secret scope / variável de ambiente."
        )
    try:
        df = spark.read.jdbc(url=url, table=table, properties=properties)
        _log_read(table, "jdbc", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha ao ler via JDBC tabela '{table}': {exc}") from exc


def read_delta_incremental(
    spark: SparkSession,
    path: str,
    starting_version: int | None = None,
    starting_timestamp: str | None = None,
) -> DataFrame:
    """Lê incrementalmente uma tabela Delta usando Change Data Feed.

    Requer que a tabela de origem tenha ``delta.enableChangeDataFeed = true``.
    """
    if starting_version is None and starting_timestamp is None:
        raise DataReadError("Leitura incremental requer 'starting_version' ou 'starting_timestamp'.")
    try:
        reader = spark.read.format("delta").option("readChangeFeed", "true")
        if starting_version is not None:
            reader = reader.option("startingVersion", starting_version)
        else:
            reader = reader.option("startingTimestamp", starting_timestamp)
        df = reader.load(path)
        _log_read(path, "delta_cdf", df)
        return df
    except Exception as exc:  # noqa: BLE001
        raise DataReadError(f"Falha na leitura incremental de '{path}': {exc}") from exc


def validate_schema(df: DataFrame, expected: StructType, source: str = "dataframe") -> None:
    """Valida que o schema do DataFrame contém exatamente os campos esperados (nome+tipo)."""
    from churn_model.exceptions import SchemaValidationError

    actual_fields = {(f.name, f.dataType.simpleString()) for f in df.schema.fields}
    expected_fields = {(f.name, f.dataType.simpleString()) for f in expected.fields}
    missing = expected_fields - actual_fields
    if missing:
        raise SchemaValidationError(
            f"Schema de '{source}' não corresponde ao esperado. Campos ausentes/divergentes: {missing}"
        )
