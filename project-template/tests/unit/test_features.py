from __future__ import annotations

import pytest

from churn_model.exceptions import FeatureEngineeringError
from churn_model.features.pipeline import (
    build_full_pipeline,
    feature_config_dict,
    load_feature_config,
    save_feature_config,
)
from churn_model.features.transformations import (
    handle_missing_values,
    split_train_val_test,
)


def test_handle_missing_values_fills_defaults(spark):
    df = spark.createDataFrame([(None, None), (1.0, "fiber")], schema=["monthly_charges", "internet_service"])
    filled = handle_missing_values(df, ["monthly_charges"], ["internet_service"])
    rows = filled.collect()
    assert rows[0]["monthly_charges"] == 0.0
    assert rows[0]["internet_service"] == "unknown"


def test_split_train_val_test_respects_fractions(sample_customers_df):
    train_df, val_df, test_df = split_train_val_test(sample_customers_df, 0.6, 0.2, 0.2, seed=42)
    total = sample_customers_df.count()
    assert train_df.count() + val_df.count() + test_df.count() == total
    assert train_df.count() > val_df.count()


def test_split_train_val_test_rejects_invalid_fractions(sample_customers_df):
    with pytest.raises(FeatureEngineeringError):
        split_train_val_test(sample_customers_df, 0.5, 0.3, 0.3, seed=42)


def test_build_full_pipeline_fits_and_transforms(sample_customers_df):
    prepared = handle_missing_values(
        sample_customers_df, ["tenure_months", "monthly_charges"], ["contract_type", "internet_service"]
    )
    pipeline, features_column = build_full_pipeline(
        numeric_columns=["tenure_months", "monthly_charges"],
        categorical_columns=["contract_type", "internet_service"],
        label_column="churn",
        algorithm="logistic_regression",
        hyperparameters={"max_iter": 5},
    )
    model = pipeline.fit(prepared)
    result = model.transform(prepared)
    assert features_column == "features"
    assert "prediction" in result.columns


def test_build_full_pipeline_rejects_unsupported_algorithm():
    with pytest.raises(FeatureEngineeringError):
        build_full_pipeline(["a"], [], "churn", "unknown_algo", {})


def test_feature_config_roundtrip(tmp_path):
    config = feature_config_dict(["a"], ["b"], "churn", "1", "logistic_regression", {"max_iter": 10})
    path = save_feature_config(config, tmp_path / "feature_config.json")
    loaded = load_feature_config(path)
    assert loaded == config
