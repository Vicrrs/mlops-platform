"""Transformações de dados e split treino/validação/teste sem vazamento de dados.

O split acontece ANTES de qualquer ajuste de transformador (StringIndexer,
scaler etc.). Os estágios de feature engineering só são ajustados (``fit``)
no conjunto de treino; validação e teste apenas os aplicam (``transform``),
prevenindo data leakage. A mesma função de imputação é usada em treino e
inferência para garantir consistência.
"""

from __future__ import annotations

from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_model.exceptions import FeatureEngineeringError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)

MISSING_CATEGORY_LABEL = "unknown"


def handle_missing_values(
    df: DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> DataFrame:
    """Imputação determinística: numéricos -> 0, categóricos -> 'unknown'.

    Aplicada de forma idêntica no treino e na inferência (chamada pelos dois
    fluxos), o que evita a divergência de transformação proibida pela plataforma.
    """
    result = df
    numeric_fill = {c: 0.0 for c in numeric_columns if c in df.columns}
    categorical_fill = {c: MISSING_CATEGORY_LABEL for c in categorical_columns if c in df.columns}
    if numeric_fill:
        result = result.fillna(numeric_fill)
    if categorical_fill:
        result = result.fillna(categorical_fill)
    return result


def split_train_val_test(
    df: DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Divide o dataset em treino/validação/teste antes de qualquer fit de transformador."""
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise FeatureEngineeringError(f"As frações de split devem somar 1.0, obtido {total}.")
    train_df, val_df, test_df = df.randomSplit(
        [train_fraction, validation_fraction, test_fraction], seed=seed
    )
    logger.info(
        "Split treino/validação/teste realizado",
        extra={
            "extra_fields": {
                "train_rows": train_df.count(),
                "validation_rows": val_df.count(),
                "test_rows": test_df.count(),
                "seed": seed,
            }
        },
    )
    return train_df, val_df, test_df


def build_feature_stages(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_features: bool = True,
) -> tuple[list, str]:
    """Constrói os estágios Spark ML: StringIndexer -> OneHotEncoder -> VectorAssembler (-> Scaler).

    Returns:
        Tupla (lista de estágios não ajustados, nome da coluna final de features).
    """
    stages: list = []
    encoded_columns: list[str] = []

    for column in categorical_columns:
        indexed_col = f"{column}_idx"
        encoded_col = f"{column}_ohe"
        stages.append(StringIndexer(inputCol=column, outputCol=indexed_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=indexed_col, outputCol=encoded_col))
        encoded_columns.append(encoded_col)

    assembler_inputs = list(numeric_columns) + encoded_columns
    if not assembler_inputs:
        raise FeatureEngineeringError("Nenhuma coluna numérica ou categórica configurada para features.")

    raw_features_col = "features_raw" if scale_features else "features"
    stages.append(
        VectorAssembler(inputCols=assembler_inputs, outputCol=raw_features_col, handleInvalid="keep")
    )

    final_col = raw_features_col
    if scale_features:
        final_col = "features"
        stages.append(
            StandardScaler(inputCol=raw_features_col, outputCol=final_col, withMean=True, withStd=True)
        )

    return stages, final_col


def cast_target_to_double(df: DataFrame, target_column: str) -> DataFrame:
    return df.withColumn(target_column, F.col(target_column).cast("double"))
