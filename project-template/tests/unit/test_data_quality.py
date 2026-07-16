from __future__ import annotations

import json

import pytest

from churn_model.data.quality import (
    compute_basic_data_drift,
    enforce_quality_gate,
    run_data_quality_checks,
    write_data_quality_artifacts,
)
from churn_model.exceptions import DataQualityError


def test_run_data_quality_checks_passes_on_clean_data(sample_customers_df, test_config):
    report = run_data_quality_checks(sample_customers_df, test_config.data)
    assert report.passed
    assert report.row_count == sample_customers_df.count()
    assert not report.blocking_failures()


def test_run_data_quality_checks_fails_on_missing_required_column(sample_customers_df, test_config):
    broken_df = sample_customers_df.drop("churn")
    report = run_data_quality_checks(broken_df, test_config.data)
    assert not report.passed
    assert any(r.name == "required_columns" for r in report.blocking_failures())


def test_enforce_quality_gate_raises_on_blocking_failure(sample_customers_df, test_config):
    broken_df = sample_customers_df.drop("churn")
    report = run_data_quality_checks(broken_df, test_config.data)
    with pytest.raises(DataQualityError):
        enforce_quality_gate(report)


def test_write_data_quality_artifacts(sample_customers_df, test_config, tmp_path):
    report = run_data_quality_checks(sample_customers_df, test_config.data)
    output_dir = write_data_quality_artifacts(report, tmp_path / "dq")

    for filename in (
        "data_quality_report.json",
        "failed_rules.json",
        "schema_comparison.json",
        "data_drift_report.json",
    ):
        content = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        assert content is not None


def test_compute_basic_data_drift(sample_customers_df):
    drift = compute_basic_data_drift(
        sample_customers_df, sample_customers_df, ("tenure_months", "monthly_charges")
    )
    assert drift["tenure_months"]["relative_mean_shift"] == 0.0
