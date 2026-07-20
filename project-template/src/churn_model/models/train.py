"""Treinamento do modelo candidato (Challenger)."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from churn_model.config import AppConfig
from churn_model.exceptions import ModelTrainingError
from churn_model.features.pipeline import build_full_pipeline, feature_config_dict
from churn_model.features.transformations import (
    cast_target_to_double,
    handle_missing_values,
    split_train_val_test,
)
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    pipeline_model: PipelineModel
    features_column: str
    train_df: DataFrame
    validation_df: DataFrame
    test_df: DataFrame
    feature_config: dict


def prepare_datasets(df: DataFrame, config: AppConfig) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Aplica imputação determinística e realiza o split treino/validação/teste."""
    prepared = handle_missing_values(
        df, list(config.features.numeric_columns), list(config.features.categorical_columns)
    )
    prepared = cast_target_to_double(prepared, config.features.label_column)
    return split_train_val_test(
        prepared,
        config.train_test_split.train_fraction,
        config.train_test_split.validation_fraction,
        config.train_test_split.test_fraction,
        config.train_test_split.seed,
    )


def prepare_datasets_from_feature_store(
    spark, config: AppConfig, labels_df: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Monta o dataset de treino a partir da feature store, em vez de features
    calculadas ad-hoc a partir dos dados brutos.

    ``labels_df`` só precisa conter as chaves primárias e a coluna-alvo -- as
    demais colunas (features) vêm do ``build_training_set``, que faz o lookup
    na tabela publicada por ``cli/run_feature_engineering.py``.
    """
    from churn_model.features.feature_store import get_feature_store

    store = get_feature_store(config, spark)
    training_set = store.build_training_set(labels_df, config.feature_store, config.features.label_column)
    return prepare_datasets(training_set.dataframe, config)


def train_candidate(
    train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame, config: AppConfig
) -> TrainingResult:
    """Treina o pipeline completo (pré-processamento + estimador) apenas no conjunto de treino."""
    if train_df.rdd.isEmpty():
        raise ModelTrainingError("Conjunto de treino vazio -- não é possível treinar o modelo.")

    pipeline, features_column = build_full_pipeline(
        numeric_columns=list(config.features.numeric_columns),
        categorical_columns=list(config.features.categorical_columns),
        label_column=config.features.label_column,
        algorithm=config.model.algorithm,
        hyperparameters=config.model.hyperparameters,
    )

    try:
        pipeline_model = pipeline.fit(train_df)
    except Exception as exc:  # noqa: BLE001
        raise ModelTrainingError(f"Falha ao treinar o pipeline: {exc}") from exc

    feature_config = feature_config_dict(
        numeric_columns=list(config.features.numeric_columns),
        categorical_columns=list(config.features.categorical_columns),
        label_column=config.features.label_column,
        version=config.features.version,
        algorithm=config.model.algorithm,
        hyperparameters=config.model.hyperparameters,
    )

    logger.info(
        "Treinamento do modelo candidato concluído",
        extra={
            "extra_fields": {
                "algorithm": config.model.algorithm,
                "train_rows": train_df.count(),
            }
        },
    )

    return TrainingResult(
        pipeline_model=pipeline_model,
        features_column=features_column,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_config=feature_config,
    )
