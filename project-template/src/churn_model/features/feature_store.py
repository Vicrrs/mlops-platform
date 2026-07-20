"""Feature Store: tabelas de features reutilizáveis, desacopladas do treino.

Duas implementações por trás da mesma interface, escolhidas por ambiente
(mesmo padrão de ``churn_model.spark`` e ``churn_model.models.registry``):

- ``DatabricksFeatureStore`` (hml/prd): usa o Databricks Feature Engineering
  client sobre tabelas reais do Unity Catalog, com ``FeatureLookup`` para
  montar o dataset de treino e ``score_batch`` para inferência com lookup
  automático de features.
- ``LocalFeatureStore`` (dev/local): sem Unity Catalog disponível, persiste a
  tabela de features como Delta local (reaproveitando ``io.writers.merge_delta``)
  e mantém um registro (``registry.json``) para permitir o mesmo fluxo de
  "escrever uma vez, consumir por nome" -- útil para desenvolver e testar a
  lógica de ponta a ponta sem workspace.

Em ambos os casos, a tabela de features é escrita por um job separado
(``cli/run_feature_engineering.py``) e consumida pelo treino/inferência via
``build_training_set``/``score_batch`` -- o objetivo de uma feature store:
uma única fonte de features reaproveitável entre execuções e (no caso real)
entre modelos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from churn_model.config import AppConfig, FeatureStoreSettings
from churn_model.exceptions import ConfigurationError, FeatureEngineeringError
from churn_model.io.readers import read_delta
from churn_model.io.writers import merge_delta
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


def compute_customer_features(df: DataFrame) -> DataFrame:
    """Calcula features derivadas a partir dos dados brutos de clientes.

    Estas são as features que valem a pena centralizar numa feature store:
    combinam colunas brutas de um jeito que outros modelos sobre o mesmo
    cliente (ex.: propensão a upgrade, risco de inadimplência) também
    poderiam reaproveitar, em vez de cada projeto recalcular do zero.
    """
    return (
        df.withColumn(
            "avg_monthly_charge_ratio",
            F.when(F.col("tenure_months") > 0, F.col("total_charges") / F.col("tenure_months"))
            .otherwise(F.col("monthly_charges"))
            .cast("double"),
        )
        .withColumn(
            "support_calls_per_tenure_month",
            (F.col("support_calls") / (F.col("tenure_months") + F.lit(1))).cast("double"),
        )
        .withColumn("feature_timestamp", F.lit(datetime.now(UTC).isoformat()))
    )


@dataclass
class TrainingSetResult:
    """Resultado de ``build_training_set``, com o dataframe e os metadados de
    proveniência que são registrados no MLflow (rastreabilidade features -> modelo)."""

    dataframe: DataFrame
    table_name: str
    feature_names: tuple[str, ...]
    primary_keys: tuple[str, ...]


class LocalFeatureStore:
    """Feature store local (Delta + registro JSON), usada em ``dev``/testes."""

    def __init__(self, spark: SparkSession, artifacts_dir: Path):
        self._spark = spark
        self._root = Path(artifacts_dir) / "feature_store"
        self._registry_path = self._root / "registry.json"

    def _load_registry(self) -> dict:
        if not self._registry_path.exists():
            return {}
        return json.loads(self._registry_path.read_text(encoding="utf-8"))

    def _save_registry(self, registry: dict) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    def _table_path(self, table_name: str) -> Path:
        safe_name = table_name.replace(".", "_")
        return self._root / "tables" / safe_name

    def create_or_update_table(self, df: DataFrame, spec: FeatureStoreSettings) -> None:
        table_path = self._table_path(spec.full_table_name)
        merge_delta(df, str(table_path), merge_keys=list(spec.primary_keys), environment="dev")

        registry = self._load_registry()
        registry[spec.full_table_name] = {
            "path": str(table_path),
            "primary_keys": list(spec.primary_keys),
            "feature_names": list(spec.feature_names),
            "timestamp_column": spec.timestamp_column,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._save_registry(registry)
        logger.info(
            "Tabela de features atualizada (local)",
            extra={"extra_fields": {"table_name": spec.full_table_name, "path": str(table_path)}},
        )

    def read_table(self, table_name: str) -> DataFrame:
        registry = self._load_registry()
        entry = registry.get(table_name)
        if entry is None:
            raise FeatureEngineeringError(
                f"Tabela de features '{table_name}' não encontrada no registro local. "
                "Rode a feature engineering job antes do treino/inferência."
            )
        return read_delta(self._spark, entry["path"])

    def build_training_set(
        self, labels_df: DataFrame, spec: FeatureStoreSettings, label_column: str
    ) -> TrainingSetResult:
        features_df = self.read_table(spec.full_table_name)
        select_cols = list(spec.primary_keys) + list(spec.feature_names)
        joined = labels_df.select(*spec.primary_keys, label_column).join(
            features_df.select(*select_cols), on=list(spec.primary_keys), how="inner"
        )
        logger.info(
            "Training set montado a partir da feature store (local)",
            extra={"extra_fields": {"table_name": spec.full_table_name, "row_count": joined.count()}},
        )
        return TrainingSetResult(
            dataframe=joined,
            table_name=spec.full_table_name,
            feature_names=spec.feature_names,
            primary_keys=spec.primary_keys,
        )

    def score_batch(self, keys_df: DataFrame, spec: FeatureStoreSettings) -> DataFrame:
        features_df = self.read_table(spec.full_table_name)
        select_cols = list(spec.primary_keys) + list(spec.feature_names)
        return keys_df.join(features_df.select(*select_cols), on=list(spec.primary_keys), how="left")

    def supports_online_model_lookup(self) -> bool:
        """LocalFeatureStore não tem serving online -- o chamador deve fazer o
        join manualmente antes de pontuar (ver ``score_batch``)."""
        return False


class DatabricksFeatureStore:
    """Feature store real, sobre o Databricks Feature Engineering client (Unity Catalog)."""

    def __init__(self):
        try:
            from databricks.feature_engineering import FeatureEngineeringClient
        except ImportError as exc:  # pragma: no cover - exercitado apenas fora de dev
            raise ConfigurationError(
                "databricks-feature-engineering não está instalado. Instale o extra "
                "'feature-store' do projeto para rodar em hml/prd."
            ) from exc
        self._client = FeatureEngineeringClient()

    def create_or_update_table(self, df: DataFrame, spec: FeatureStoreSettings) -> None:
        spark = df.sparkSession
        if spark.catalog.tableExists(spec.full_table_name):
            self._client.write_table(name=spec.full_table_name, df=df, mode="merge")
        else:
            self._client.create_table(
                name=spec.full_table_name,
                primary_keys=list(spec.primary_keys),
                df=df,
                description=f"Feature table gerenciada pela feature engineering job ({spec.full_table_name}).",
            )
        logger.info(
            "Tabela de features atualizada (Unity Catalog)",
            extra={"extra_fields": {"table_name": spec.full_table_name}},
        )

    def read_table(self, table_name: str) -> DataFrame:
        return self._client.read_table(name=table_name)

    def build_training_set(
        self, labels_df: DataFrame, spec: FeatureStoreSettings, label_column: str
    ) -> TrainingSetResult:
        from databricks.feature_engineering import FeatureLookup

        lookups = [
            FeatureLookup(
                table_name=spec.full_table_name,
                lookup_key=list(spec.primary_keys),
                feature_names=list(spec.feature_names),
            )
        ]
        training_set = self._client.create_training_set(
            df=labels_df.select(*spec.primary_keys, label_column),
            feature_lookups=lookups,
            label=label_column,
        )
        self._last_training_set = training_set
        return TrainingSetResult(
            dataframe=training_set.load_df(),
            table_name=spec.full_table_name,
            feature_names=spec.feature_names,
            primary_keys=spec.primary_keys,
        )

    def log_model_with_lookup(self, pipeline_model, registered_model_name: str | None = None):
        import mlflow

        training_set = getattr(self, "_last_training_set", None)
        if training_set is None:
            raise FeatureEngineeringError(
                "build_training_set precisa ser chamado antes de log_model_with_lookup "
                "(o log de modelo com feature lookup depende do training_set gerado)."
            )
        self._client.log_model(
            model=pipeline_model,
            artifact_path="model",
            flavor=mlflow.spark,
            training_set=training_set,
            registered_model_name=registered_model_name,
        )

    def score_batch(self, model_uri: str, keys_df: DataFrame) -> DataFrame:
        return self._client.score_batch(model_uri=model_uri, df=keys_df)

    def supports_online_model_lookup(self) -> bool:
        return True


def get_feature_store(config: AppConfig, spark: SparkSession):
    """Retorna a implementação de feature store adequada ao ambiente configurado."""
    if config.environment in ("hml", "prd"):
        return DatabricksFeatureStore()
    return LocalFeatureStore(spark, config.artifacts_dir())
