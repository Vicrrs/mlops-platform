from __future__ import annotations

import pytest

from churn_model.exceptions import ModelTrainingError
from churn_model.models.train import prepare_datasets, train_candidate


def test_prepare_datasets_splits_and_casts_target(sample_customers_df, test_config):
    train_df, val_df, test_df = prepare_datasets(sample_customers_df, test_config)
    total = sample_customers_df.count()
    assert train_df.count() + val_df.count() + test_df.count() == total
    assert dict(train_df.dtypes)["churn"] == "double"


def test_train_candidate_produces_fitted_pipeline(sample_customers_df, test_config):
    train_df, val_df, test_df = prepare_datasets(sample_customers_df, test_config)
    result = train_candidate(train_df, val_df, test_df, test_config)
    assert result.pipeline_model is not None
    predictions = result.pipeline_model.transform(val_df)
    assert "prediction" in predictions.columns
    assert result.feature_config["algorithm"] == "logistic_regression"


def test_train_candidate_rejects_empty_train_set(spark, sample_customers_df, test_config):
    empty_train = sample_customers_df.limit(0)
    with pytest.raises(ModelTrainingError):
        train_candidate(empty_train, sample_customers_df, sample_customers_df, test_config)
