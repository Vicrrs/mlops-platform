"""Schema explícito dos dados brutos de churn de clientes.

Usado por leitura (evita inferência em produção), validação e geração de dados
sintéticos de exemplo -- é a única fonte de verdade sobre nomes/tipos de coluna.
"""

from __future__ import annotations

from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

CUSTOMER_ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"

RAW_CUSTOMER_CHURN_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), nullable=False),
        StructField("tenure_months", IntegerType(), nullable=False),
        StructField("monthly_charges", DoubleType(), nullable=False),
        StructField("total_charges", DoubleType(), nullable=True),
        StructField("contract_type", StringType(), nullable=False),
        StructField("internet_service", StringType(), nullable=False),
        StructField("support_calls", IntegerType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=False),
        StructField("signup_date", DateType(), nullable=True),
        StructField("churn", IntegerType(), nullable=False),
    ]
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(f.name for f in RAW_CUSTOMER_CHURN_SCHEMA.fields)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "contract_type",
    "internet_service",
)

ALLOWED_CONTRACT_TYPES: tuple[str, ...] = ("month-to-month", "one-year", "two-year")
ALLOWED_INTERNET_SERVICES: tuple[str, ...] = ("dsl", "fiber", "none")
ALLOWED_TARGET_VALUES: tuple[int, ...] = (0, 1)

PREDICTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "prediction",
    "probability_churn",
    "model_version",
    "mlflow_run_id",
    "scoring_timestamp",
    "git_commit",
)
