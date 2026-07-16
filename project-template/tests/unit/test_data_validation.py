from __future__ import annotations

from churn_model.data import validation as v


def test_check_required_columns_detects_missing(sample_customers_df):
    result = v.check_required_columns(sample_customers_df, ("customer_id", "does_not_exist"))
    assert not result.passed
    assert result.severity == v.Severity.CRITICAL
    assert "does_not_exist" in result.observed_value


def test_check_required_columns_passes(sample_customers_df):
    result = v.check_required_columns(sample_customers_df, ("customer_id", "churn"))
    assert result.passed


def test_check_duplicates(spark, sample_customers_df):
    duplicated = sample_customers_df.union(sample_customers_df.limit(10))
    result = v.check_duplicates(duplicated, ["customer_id"], max_duplicate_percentage=0.0)
    assert not result.passed
    assert result.observed_value > 0


def test_check_null_percentage(spark, sample_customers_df):
    result = v.check_null_percentage(sample_customers_df, "monthly_charges", max_null_percentage=0.0)
    assert result.passed  # dados sintéticos não têm nulos em monthly_charges


def test_check_value_range_flags_out_of_range(sample_customers_df):
    result = v.check_value_range(sample_customers_df, "tenure_months", min_value=0, max_value=1)
    assert not result.passed
    assert result.observed_value > 0


def test_check_allowed_domain(sample_customers_df):
    result = v.check_allowed_domain(sample_customers_df, "contract_type", ("month-to-month",))
    assert not result.passed


def test_check_row_count(sample_customers_df):
    result = v.check_row_count(sample_customers_df, min_rows=10_000, max_rows=20_000)
    assert not result.passed


def test_check_schema_drift_detects_change():
    result = v.check_schema_drift({"a": 1}, {"a": 1})
    assert result.passed
    result_changed = v.check_schema_drift({"a": 2}, {"a": 1})
    assert not result_changed.passed


def test_rule_result_is_blocking_failure_only_for_error_and_critical():
    warning_fail = v._result("x", "d", v.Severity.WARNING, False, 1, 0)
    error_fail = v._result("x", "d", v.Severity.ERROR, False, 1, 0)
    assert not warning_fail.is_blocking_failure
    assert error_fail.is_blocking_failure
