from __future__ import annotations

from churn_model.features.pipeline import build_estimator, build_full_pipeline
from churn_model.features.transformations import cast_target_to_double, handle_missing_values


def test_build_full_pipeline_random_forest_fits(sample_customers_df):
    prepared = handle_missing_values(
        sample_customers_df, ["tenure_months"], ["contract_type", "internet_service"]
    )
    prepared = cast_target_to_double(prepared, "churn")
    pipeline, features_column = build_full_pipeline(
        numeric_columns=["tenure_months"],
        categorical_columns=["contract_type", "internet_service"],
        label_column="churn",
        algorithm="random_forest",
        hyperparameters={"num_trees": 5, "max_depth": 3, "seed": 1},
    )
    model = pipeline.fit(prepared)
    predictions = model.transform(prepared)
    assert "prediction" in predictions.columns
    assert len(model.stages) == len(pipeline.getStages())


def test_build_estimator_logistic_regression_uses_hyperparameters():
    estimator = build_estimator("logistic_regression", "churn", "features", {"max_iter": 7, "reg_param": 0.2})
    assert estimator.getMaxIter() == 7
    assert estimator.getRegParam() == 0.2


def test_pipeline_transform_is_deterministic_between_train_and_inference(sample_customers_df):
    """A mesma PipelineModel deve produzir o mesmo resultado ao ser aplicada duas vezes
    -- garante que não há divergência entre a transformação usada no treino e na inferência."""
    prepared = handle_missing_values(
        sample_customers_df, ["tenure_months"], ["contract_type", "internet_service"]
    )
    prepared = cast_target_to_double(prepared, "churn")
    pipeline, _ = build_full_pipeline(
        numeric_columns=["tenure_months"],
        categorical_columns=["contract_type", "internet_service"],
        label_column="churn",
        algorithm="logistic_regression",
        hyperparameters={"max_iter": 5},
    )
    model = pipeline.fit(prepared)

    first_predictions = model.transform(prepared).select("customer_id", "prediction").collect()
    second_predictions = model.transform(prepared).select("customer_id", "prediction").collect()
    assert first_predictions == second_predictions
