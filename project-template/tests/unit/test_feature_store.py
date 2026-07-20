from __future__ import annotations

import pytest

from churn_model.exceptions import FeatureEngineeringError
from churn_model.features.feature_store import LocalFeatureStore, compute_customer_features, get_feature_store


def test_compute_customer_features_adds_derived_columns(sample_customers_df):
    result = compute_customer_features(sample_customers_df)
    for column in ("avg_monthly_charge_ratio", "support_calls_per_tenure_month", "feature_timestamp"):
        assert column in result.columns
    assert result.count() == sample_customers_df.count()


def test_compute_customer_features_no_division_by_zero_for_new_customers(spark):
    df = spark.createDataFrame(
        [("CUST-1", 0, 50.0, 0.0, 2)],
        schema=["customer_id", "tenure_months", "monthly_charges", "total_charges", "support_calls"],
    )
    result = compute_customer_features(df)
    row = result.collect()[0]
    # tenure_months=0 -> usa monthly_charges como proxy, não divide por zero.
    assert row["avg_monthly_charge_ratio"] == 50.0
    assert row["support_calls_per_tenure_month"] == 2.0  # support_calls / (0 + 1)


def test_get_feature_store_returns_local_for_dev(spark, test_config):
    store = get_feature_store(test_config, spark)
    assert isinstance(store, LocalFeatureStore)
    assert store.supports_online_model_lookup() is False


def test_local_feature_store_create_and_read_round_trip(spark, sample_customers_df, test_config):
    store = get_feature_store(test_config, spark)
    features_df = compute_customer_features(sample_customers_df)

    store.create_or_update_table(features_df, test_config.feature_store)

    read_back = store.read_table(test_config.feature_store.full_table_name)
    assert read_back.count() == sample_customers_df.count()
    assert "avg_monthly_charge_ratio" in read_back.columns


def test_local_feature_store_read_unknown_table_raises(spark, test_config):
    store = get_feature_store(test_config, spark)
    with pytest.raises(FeatureEngineeringError):
        store.read_table("does_not_exist")


def test_local_feature_store_build_training_set_joins_by_primary_key(spark, sample_customers_df, test_config):
    store = get_feature_store(test_config, spark)
    features_df = compute_customer_features(sample_customers_df)
    store.create_or_update_table(features_df, test_config.feature_store)

    labels_df = sample_customers_df.select("customer_id", "churn")
    training_set = store.build_training_set(labels_df, test_config.feature_store, "churn")

    assert training_set.table_name == test_config.feature_store.full_table_name
    assert training_set.dataframe.count() == sample_customers_df.count()
    for feature in test_config.feature_store.feature_names:
        assert feature in training_set.dataframe.columns
    assert "churn" in training_set.dataframe.columns


def test_local_feature_store_write_is_upsert_not_duplicate(spark, sample_customers_df, test_config):
    store = get_feature_store(test_config, spark)
    features_df = compute_customer_features(sample_customers_df)

    store.create_or_update_table(features_df, test_config.feature_store)
    store.create_or_update_table(features_df, test_config.feature_store)  # roda de novo, mesmas chaves

    read_back = store.read_table(test_config.feature_store.full_table_name)
    assert read_back.count() == sample_customers_df.count()


def test_local_feature_store_score_batch_joins_features(spark, sample_customers_df, test_config):
    store = get_feature_store(test_config, spark)
    features_df = compute_customer_features(sample_customers_df)
    store.create_or_update_table(features_df, test_config.feature_store)

    keys_df = sample_customers_df.select("customer_id").limit(10)
    scored_input = store.score_batch(keys_df, test_config.feature_store)

    assert scored_input.count() == 10
    assert "avg_monthly_charge_ratio" in scored_input.columns
