"""Carregamento de configuração em camadas (base + ambiente) com resolução de variáveis.

``conf/base.yml`` contém os valores comuns entre ambientes. ``conf/{dev,hml,prd}.yml``
sobrescreve apenas o que muda por ambiente (merge profundo). Placeholders no formato
``${VAR_NAME}`` ou ``${VAR_NAME:-default}`` são resolvidos a partir de variáveis de
ambiente no momento da carga -- nenhum valor real de catálogo/host é fixado no YAML.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from churn_model.exceptions import ConfigurationError

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


def _resolve_placeholders(value: Any) -> Any:
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            var_name, _, default = match.groups()
            return os.environ.get(var_name, default if default is not None else "")

        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class SparkSettings:
    app_name: str = "churn_model"
    master: str | None = None
    shuffle_partitions: int = 8


@dataclass(frozen=True)
class QualitySettings:
    max_null_percentage: float = 0.05
    max_duplicate_percentage: float = 0.01
    min_rows: int = 1
    max_rows: int = 100_000_000
    max_volume_change_percentage: float = 0.5


@dataclass(frozen=True)
class DataSettings:
    source: str = "local"
    raw_path: str = ""
    raw_table: str = ""
    catalog: str = ""
    schema: str = ""
    warehouse_dir: str = "artifacts/warehouse"
    target_column: str = "churn"
    id_column: str = "customer_id"
    required_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    quality: QualitySettings = field(default_factory=QualitySettings)


@dataclass(frozen=True)
class FeaturesSettings:
    version: str = "1"
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    label_column: str = "churn"


@dataclass(frozen=True)
class ModelThresholds:
    minimum_accuracy: float = 0.6
    minimum_precision: float = 0.55
    minimum_recall: float = 0.55
    minimum_f1_score: float = 0.55
    minimum_roc_auc: float = 0.6
    maximum_metric_regression: float = 0.01
    minimum_improvement: float = 0.0


@dataclass(frozen=True)
class ModelSettings:
    type: str = "spark_ml"
    algorithm: str = "logistic_regression"
    primary_metric: str = "f1_score"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    thresholds: ModelThresholds = field(default_factory=ModelThresholds)


@dataclass(frozen=True)
class MLflowSettings:
    tracking_uri: str = "file:./mlruns"
    registry_uri: str = "file:./mlruns"
    experiment_name: str = "/churn_model/experiment"
    registered_model_name: str = "churn_model"
    registered_model_alias_champion: str = "champion"
    registered_model_alias_challenger: str = "challenger"
    registered_model_alias_previous_champion: str = "previous_champion"
    registry_mode: str = "shared_uc"


@dataclass(frozen=True)
class TrainTestSplitSettings:
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    environment: str
    project_name: str
    package_name: str
    spark: SparkSettings
    data: DataSettings
    features: FeaturesSettings
    model: ModelSettings
    mlflow: MLflowSettings
    train_test_split: TrainTestSplitSettings
    raw: dict[str, Any]

    def artifacts_dir(self) -> Path:
        return Path(self.raw.get("artifacts_dir", "artifacts"))


def _build_data_settings(raw: dict[str, Any]) -> DataSettings:
    quality_raw = raw.get("quality", {})
    return DataSettings(
        source=raw.get("source", "local"),
        raw_path=raw.get("raw_path", ""),
        raw_table=raw.get("raw_table", ""),
        catalog=raw.get("catalog", ""),
        schema=raw.get("schema", ""),
        warehouse_dir=raw.get("warehouse_dir", "artifacts/warehouse"),
        target_column=raw.get("target_column", "churn"),
        id_column=raw.get("id_column", "customer_id"),
        required_columns=tuple(raw.get("required_columns", [])),
        categorical_columns=tuple(raw.get("categorical_columns", [])),
        numeric_columns=tuple(raw.get("numeric_columns", [])),
        quality=QualitySettings(
            max_null_percentage=quality_raw.get("max_null_percentage", 0.05),
            max_duplicate_percentage=quality_raw.get("max_duplicate_percentage", 0.01),
            min_rows=quality_raw.get("min_rows", 1),
            max_rows=quality_raw.get("max_rows", 100_000_000),
            max_volume_change_percentage=quality_raw.get("max_volume_change_percentage", 0.5),
        ),
    )


def _build_model_settings(raw: dict[str, Any]) -> ModelSettings:
    thresholds_raw = raw.get("thresholds", {})
    return ModelSettings(
        type=raw.get("type", "spark_ml"),
        algorithm=raw.get("algorithm", "logistic_regression"),
        primary_metric=raw.get("primary_metric", "f1_score"),
        hyperparameters=dict(raw.get("hyperparameters", {})),
        thresholds=ModelThresholds(
            minimum_accuracy=thresholds_raw.get("minimum_accuracy", 0.6),
            minimum_precision=thresholds_raw.get("minimum_precision", 0.55),
            minimum_recall=thresholds_raw.get("minimum_recall", 0.55),
            minimum_f1_score=thresholds_raw.get("minimum_f1_score", 0.55),
            minimum_roc_auc=thresholds_raw.get("minimum_roc_auc", 0.6),
            maximum_metric_regression=thresholds_raw.get("maximum_metric_regression", 0.01),
            minimum_improvement=thresholds_raw.get("minimum_improvement", 0.0),
        ),
    )


def load_config(config_path: str | Path) -> AppConfig:
    """Carrega ``conf/base.yml`` mesclado com o arquivo de ambiente informado.

    Args:
        config_path: caminho para o YAML de ambiente, ex.: ``conf/dev.yml``.
            O arquivo ``base.yml`` deve estar no mesmo diretório.

    Raises:
        ConfigurationError: quando algum arquivo obrigatório está ausente ou inválido.
    """
    env_path = Path(config_path)
    if not env_path.exists():
        raise ConfigurationError(f"Arquivo de configuração não encontrado: {env_path}")

    base_path = env_path.parent / "base.yml"
    base_raw: dict[str, Any] = {}
    if base_path.exists():
        with base_path.open("r", encoding="utf-8") as fh:
            base_raw = yaml.safe_load(fh) or {}

    with env_path.open("r", encoding="utf-8") as fh:
        env_raw = yaml.safe_load(fh) or {}

    merged = _resolve_placeholders(_deep_merge(base_raw, env_raw))

    if "environment" not in merged:
        raise ConfigurationError(f"Campo 'environment' ausente em {env_path}")

    try:
        return AppConfig(
            environment=merged["environment"],
            project_name=merged.get("project_name", "ml-project"),
            package_name=merged.get("package_name", "package"),
            spark=SparkSettings(**merged.get("spark", {})),
            data=_build_data_settings(merged.get("data", {})),
            features=FeaturesSettings(
                version=str(merged.get("features", {}).get("version", "1")),
                numeric_columns=tuple(merged.get("features", {}).get("numeric_columns", [])),
                categorical_columns=tuple(merged.get("features", {}).get("categorical_columns", [])),
                label_column=merged.get("features", {}).get("label_column", "churn"),
            ),
            model=_build_model_settings(merged.get("model", {})),
            mlflow=MLflowSettings(**merged.get("mlflow", {})),
            train_test_split=TrainTestSplitSettings(**merged.get("train_test_split", {})),
            raw=merged,
        )
    except TypeError as exc:
        raise ConfigurationError(f"Configuração inválida em {env_path}: {exc}") from exc
