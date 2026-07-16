"""Funções reutilizáveis de escrita de dados.

Regras aplicadas por todas as funções: validam DataFrame vazio antes de escrever,
registram destino/modo/quantidade de registros, suportam ``dry_run`` (não escreve,
apenas loga o que seria feito) e bloqueiam ``overwrite`` em PRD a menos que
``allow_prd_overwrite=True`` seja passado explicitamente pelo chamador.
"""

from __future__ import annotations

from pyspark.sql import DataFrame

from churn_model.exceptions import DataWriteError, EmptyDataFrameError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)

_VALID_MODES = {"append", "overwrite", "errorifexists", "ignore"}


def _guard_write(
    df: DataFrame,
    destination: str,
    mode: str,
    environment: str,
    allow_prd_overwrite: bool,
    partition_by: list[str] | None,
) -> int:
    if mode not in _VALID_MODES:
        raise DataWriteError(f"Modo de escrita inválido: '{mode}'. Use um de {_VALID_MODES}.")

    if df.rdd.isEmpty():
        raise EmptyDataFrameError(f"Tentativa de escrever DataFrame vazio em '{destination}'.")

    if mode == "overwrite" and environment == "prd" and not allow_prd_overwrite:
        raise DataWriteError(
            f"Overwrite em PRD bloqueado para '{destination}'. "
            "Passe allow_prd_overwrite=True explicitamente se isso for intencional."
        )

    if partition_by:
        missing = set(partition_by) - set(df.columns)
        if missing:
            raise DataWriteError(f"Colunas de particionamento inexistentes no DataFrame: {missing}")

    return df.count()


def write_table(
    dataframe: DataFrame,
    table_name: str,
    mode: str,
    partition_by: list[str] | None = None,
    environment: str = "dev",
    allow_prd_overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Escreve em uma tabela gerenciada (Unity Catalog / metastore), formato Delta."""
    count = _guard_write(dataframe, table_name, mode, environment, allow_prd_overwrite, partition_by)

    if dry_run:
        logger.info(
            "[dry-run] escrita não realizada",
            extra={
                "extra_fields": {
                    "destination": table_name,
                    "mode": mode,
                    "row_count": count,
                }
            },
        )
        return

    try:
        writer = dataframe.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(table_name)
        logger.info(
            "Escrita em tabela concluída",
            extra={
                "extra_fields": {
                    "destination": table_name,
                    "mode": mode,
                    "row_count": count,
                    "partition_by": partition_by or [],
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise DataWriteError(f"Falha ao escrever tabela '{table_name}': {exc}") from exc


def write_delta(
    dataframe: DataFrame,
    path: str,
    mode: str,
    partition_by: list[str] | None = None,
    environment: str = "dev",
    allow_prd_overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Escreve dados no formato Delta em um caminho explícito."""
    count = _guard_write(dataframe, path, mode, environment, allow_prd_overwrite, partition_by)

    if dry_run:
        logger.info(
            "[dry-run] escrita não realizada",
            extra={"extra_fields": {"destination": path, "mode": mode, "row_count": count}},
        )
        return

    try:
        writer = dataframe.write.format("delta").mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
        logger.info(
            "Escrita Delta concluída",
            extra={
                "extra_fields": {
                    "destination": path,
                    "mode": mode,
                    "row_count": count,
                    "partition_by": partition_by or [],
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise DataWriteError(f"Falha ao escrever Delta em '{path}': {exc}") from exc


def write_parquet(
    dataframe: DataFrame,
    path: str,
    mode: str,
    partition_by: list[str] | None = None,
    environment: str = "dev",
    allow_prd_overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    count = _guard_write(dataframe, path, mode, environment, allow_prd_overwrite, partition_by)
    if dry_run:
        logger.info(
            "[dry-run] escrita não realizada",
            extra={"extra_fields": {"destination": path, "mode": mode, "row_count": count}},
        )
        return
    try:
        writer = dataframe.write.mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.parquet(path)
        logger.info(
            "Escrita Parquet concluída",
            extra={"extra_fields": {"destination": path, "mode": mode, "row_count": count}},
        )
    except Exception as exc:  # noqa: BLE001
        raise DataWriteError(f"Falha ao escrever Parquet em '{path}': {exc}") from exc


def write_csv(
    dataframe: DataFrame,
    path: str,
    mode: str,
    header: bool = True,
    environment: str = "dev",
    allow_prd_overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Escrita CSV -- reservada para casos pontuais (ex.: exportações), não fluxo principal."""
    count = _guard_write(dataframe, path, mode, environment, allow_prd_overwrite, None)
    if dry_run:
        logger.info(
            "[dry-run] escrita não realizada",
            extra={"extra_fields": {"destination": path, "mode": mode, "row_count": count}},
        )
        return
    try:
        dataframe.write.mode(mode).option("header", str(header)).csv(path)
        logger.info(
            "Escrita CSV concluída",
            extra={"extra_fields": {"destination": path, "mode": mode, "row_count": count}},
        )
    except Exception as exc:  # noqa: BLE001
        raise DataWriteError(f"Falha ao escrever CSV em '{path}': {exc}") from exc


def write_jdbc(
    dataframe: DataFrame,
    url: str,
    table: str,
    mode: str,
    properties: dict[str, str],
    environment: str = "dev",
    allow_prd_overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    if not properties.get("user") or not properties.get("password"):
        raise DataWriteError("Propriedades JDBC incompletas: 'user' e 'password' devem vir de secrets.")
    count = _guard_write(dataframe, table, mode, environment, allow_prd_overwrite, None)
    if dry_run:
        logger.info(
            "[dry-run] escrita não realizada",
            extra={"extra_fields": {"destination": table, "mode": mode, "row_count": count}},
        )
        return
    try:
        dataframe.write.jdbc(url=url, table=table, mode=mode, properties=properties)
        logger.info(
            "Escrita JDBC concluída",
            extra={"extra_fields": {"destination": table, "mode": mode, "row_count": count}},
        )
    except Exception as exc:  # noqa: BLE001
        raise DataWriteError(f"Falha ao escrever via JDBC em '{table}': {exc}") from exc


def merge_delta(
    dataframe: DataFrame,
    path: str,
    merge_keys: list[str],
    environment: str = "dev",
    dry_run: bool = False,
) -> None:
    """Executa merge/upsert em uma tabela Delta existente com base em ``merge_keys``.

    Se a tabela ainda não existir no caminho, realiza a escrita inicial (overwrite).
    """
    from delta.tables import DeltaTable

    if dataframe.rdd.isEmpty():
        raise EmptyDataFrameError(f"Tentativa de merge com DataFrame vazio em '{path}'.")
    if not merge_keys:
        raise DataWriteError("merge_delta requer ao menos uma chave em 'merge_keys'.")

    count = dataframe.count()
    if dry_run:
        logger.info(
            "[dry-run] merge não realizado",
            extra={"extra_fields": {"destination": path, "row_count": count, "merge_keys": merge_keys}},
        )
        return

    spark = dataframe.sparkSession
    try:
        if DeltaTable.isDeltaTable(spark, path):
            target = DeltaTable.forPath(spark, path)
            condition = " AND ".join(f"target.{k} = source.{k}" for k in merge_keys)
            (
                target.alias("target")
                .merge(dataframe.alias("source"), condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            dataframe.write.format("delta").mode("errorifexists").save(path)
        logger.info(
            "Merge Delta concluído",
            extra={"extra_fields": {"destination": path, "row_count": count, "merge_keys": merge_keys}},
        )
    except Exception as exc:  # noqa: BLE001
        raise DataWriteError(f"Falha ao executar merge Delta em '{path}': {exc}") from exc
