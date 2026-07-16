from __future__ import annotations

from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

from churn_model.features.transformations import build_feature_stages, handle_missing_values


def test_build_feature_stages_produces_expected_stage_types(spark):
    stages, final_col = build_feature_stages(
        numeric_columns=["tenure_months", "monthly_charges"],
        categorical_columns=["contract_type"],
    )
    stage_types = [type(s) for s in stages]
    assert StringIndexer in stage_types
    assert OneHotEncoder in stage_types
    assert VectorAssembler in stage_types
    assert StandardScaler in stage_types
    assert final_col == "features"


def test_build_feature_stages_without_scaling_assembles_directly_into_features(spark):
    stages, final_col = build_feature_stages(
        numeric_columns=["tenure_months"], categorical_columns=[], scale_features=False
    )
    # Sem scaler, o VectorAssembler grava direto na coluna final "features" (não há
    # estágio intermediário "features_raw" a renomear).
    assert final_col == "features"
    assert not any(isinstance(s, StandardScaler) for s in stages)
    assert isinstance(stages[-1], VectorAssembler)
    assert stages[-1].getOutputCol() == "features"


def test_handle_missing_values_preserves_row_count(sample_customers_df):
    result = handle_missing_values(sample_customers_df, ["tenure_months"], ["contract_type"])
    assert result.count() == sample_customers_df.count()


def test_feature_stages_fit_transform_produces_vector_column(spark, sample_customers_df):
    stages, final_col = build_feature_stages(
        numeric_columns=["tenure_months", "monthly_charges"], categorical_columns=["contract_type"]
    )
    from pyspark.ml import Pipeline

    model = Pipeline(stages=stages).fit(sample_customers_df)
    result = model.transform(sample_customers_df)
    assert final_col in result.columns
    vector_type = dict(result.dtypes)[final_col]
    assert "vector" in vector_type.lower()
