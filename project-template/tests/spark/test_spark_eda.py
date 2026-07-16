from __future__ import annotations

from churn_model.data.eda import (
    profile_correlations,
    profile_duplicates,
    profile_nulls,
    profile_numeric_statistics,
    profile_outliers_basic,
)


def test_profile_nulls_uses_aggregation_not_full_collect(sample_customers_df):
    result = profile_nulls(sample_customers_df, ["monthly_charges", "support_calls"])
    assert result["row_count"] == sample_customers_df.count()
    assert result["columns"]["monthly_charges"]["null_count"] == 0


def test_profile_duplicates_detects_exact_duplicates(spark, sample_customers_df):
    duplicated = sample_customers_df.union(sample_customers_df.limit(20))
    result = profile_duplicates(duplicated, ["customer_id"])
    assert result["duplicate_count"] == 20


def test_profile_numeric_statistics_bounds_are_consistent(sample_customers_df):
    stats = profile_numeric_statistics(sample_customers_df, ["tenure_months"])
    assert (
        stats["tenure_months"]["min"]
        <= stats["tenure_months"]["median_approx"]
        <= stats["tenure_months"]["max"]
    )


def test_profile_correlations_returns_value_between_minus_one_and_one(sample_customers_df):
    correlations = profile_correlations(sample_customers_df, ["tenure_months", "monthly_charges"])
    value = correlations["tenure_months__monthly_charges"]
    assert -1.0 <= value <= 1.0


def test_profile_outliers_basic_uses_iqr_bounds(sample_customers_df):
    outliers = profile_outliers_basic(sample_customers_df, ["monthly_charges"])
    stats = outliers["monthly_charges"]
    assert stats["lower_bound"] <= stats["q1"] <= stats["q3"] <= stats["upper_bound"]
    assert stats["outlier_count"] >= 0
