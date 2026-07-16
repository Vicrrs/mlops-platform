from __future__ import annotations

import json

import pytest

from churn_model.data.eda import run_eda, write_eda_artifacts
from churn_model.exceptions import SchemaValidationError


def test_run_eda_produces_all_sections(sample_customers_df):
    summary = run_eda(
        sample_customers_df,
        required_columns=("customer_id", "churn"),
        numeric_columns=("tenure_months", "monthly_charges"),
        categorical_columns=("contract_type", "internet_service"),
        target_column="churn",
        date_columns=("signup_date",),
        dataset_name="test",
    )
    assert summary["row_count"] == sample_customers_df.count()
    assert "tenure_months" in summary["numeric_statistics"]
    assert "contract_type" in summary["categorical_statistics"]
    assert summary["target_distribution"]["total"] == summary["row_count"]
    assert "tenure_months__monthly_charges" in summary["correlations"]


def test_run_eda_fails_on_missing_required_column(sample_customers_df):
    with pytest.raises(SchemaValidationError):
        run_eda(
            sample_customers_df,
            required_columns=("does_not_exist",),
            numeric_columns=(),
            categorical_columns=(),
            target_column="churn",
        )


def test_run_eda_does_not_fail_on_missing_optional_date_column(sample_customers_df):
    summary = run_eda(
        sample_customers_df,
        required_columns=("customer_id",),
        numeric_columns=("tenure_months",),
        categorical_columns=(),
        target_column="churn",
        date_columns=("optional_column_that_does_not_exist",),
    )
    assert summary["temporal"] == {}


def test_write_eda_artifacts(sample_customers_df, tmp_path):
    summary = run_eda(
        sample_customers_df,
        required_columns=("customer_id",),
        numeric_columns=("tenure_months",),
        categorical_columns=("contract_type",),
        target_column="churn",
    )
    output_dir = write_eda_artifacts(summary, tmp_path / "eda")
    for filename in (
        "eda_summary.json",
        "schema.json",
        "null_report.json",
        "duplicate_report.json",
        "target_distribution.json",
        "numeric_statistics.json",
        "categorical_statistics.json",
    ):
        assert json.loads((output_dir / filename).read_text(encoding="utf-8")) is not None
    assert (output_dir / "data_profile.md").exists()


def test_categorical_statistics_respects_max_categories(sample_customers_df):
    summary = run_eda(
        sample_customers_df,
        required_columns=("customer_id",),
        numeric_columns=(),
        categorical_columns=("contract_type",),
        target_column="churn",
        max_categories=1,
    )
    stats = summary["categorical_statistics"]["contract_type"]
    assert stats["high_cardinality"] is True
    assert len(stats["top_categories"]) == 1
