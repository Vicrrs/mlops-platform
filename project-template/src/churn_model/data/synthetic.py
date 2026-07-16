"""Geração determinística de dados sintéticos de churn de clientes.

Usada para popular ``data/sample/`` (exemplo versionado no repositório) e como
fixture nos testes Spark/integração -- garante que o exemplo funcional roda
sem qualquer fonte de dados externa ou credencial.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession

from churn_model.data.schemas import RAW_CUSTOMER_CHURN_SCHEMA

_CONTRACT_TYPES = ("month-to-month", "one-year", "two-year")
_INTERNET_SERVICES = ("dsl", "fiber", "none")


def _generate_rows(n_rows: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    rows = []
    base_date = date(2023, 1, 1)
    for i in range(n_rows):
        contract_type = rng.choice(_CONTRACT_TYPES)
        internet_service = rng.choice(_INTERNET_SERVICES)
        tenure_months = rng.randint(0, 72)
        monthly_charges = round(rng.uniform(20.0, 150.0), 2)
        total_charges = round(monthly_charges * max(tenure_months, 1) * rng.uniform(0.9, 1.05), 2)
        support_calls = rng.randint(0, 10)
        is_active = rng.random() > 0.15
        signup_date = base_date + timedelta(days=rng.randint(0, 900))

        # Sinal determinístico para o modelo aprender: contratos mensais,
        # muitas chamadas de suporte e tenure curto aumentam a chance de churn.
        churn_score = (
            (0.35 if contract_type == "month-to-month" else 0.0)
            + (0.05 * support_calls)
            - (0.01 * tenure_months)
            + (0.10 if internet_service == "fiber" else 0.0)
            + rng.uniform(-0.2, 0.2)
        )
        churn = 1 if churn_score > 0.55 else 0

        rows.append(
            (
                f"CUST-{i:06d}",
                tenure_months,
                monthly_charges,
                total_charges,
                contract_type,
                internet_service,
                support_calls,
                is_active,
                signup_date,
                churn,
            )
        )
    return rows


def generate_customers_churn_df(spark: SparkSession, n_rows: int = 2000, seed: int = 42) -> DataFrame:
    rows = _generate_rows(n_rows, seed)
    return spark.createDataFrame(rows, schema=RAW_CUSTOMER_CHURN_SCHEMA)
