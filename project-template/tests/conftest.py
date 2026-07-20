"""Fixtures compartilhadas por toda a suíte de testes (unit/spark/integration/smoke)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PYSPARK_PYTHON", os.path.join(os.getcwd(), ".venv", "bin", "python"))

from churn_model.config import (
    AppConfig,
    DataSettings,
    FeaturesSettings,
    FeatureStoreSettings,
    MLflowSettings,
    ModelSettings,
    ModelThresholds,
    QualitySettings,
    SparkSettings,
    TrainTestSplitSettings,
)
from churn_model.data.synthetic import generate_customers_churn_df
from churn_model.spark import get_spark_session, stop_spark_session


@pytest.fixture(scope="session")
def spark():
    session = get_spark_session(app_name="churn_model-tests", environment="local", shuffle_partitions=2)
    yield session
    stop_spark_session()


@pytest.fixture
def sample_customers_df(spark):
    return generate_customers_churn_df(spark, n_rows=300, seed=7)


@pytest.fixture(scope="module")
def module_customers_df(spark):
    """Mesmos dados, mas de escopo 'module' -- para testes que treinam/registram
    um modelo e reaproveitam o resultado entre vários casos de teste do mesmo
    arquivo, evitando retreinar a cada função de teste."""
    return generate_customers_churn_df(spark, n_rows=300, seed=7)


@pytest.fixture(scope="module")
def module_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("module-scoped")


@pytest.fixture(scope="module")
def module_config(module_tmp_path) -> AppConfig:
    return _build_test_config(module_tmp_path)


@pytest.fixture
def test_config(tmp_path) -> AppConfig:
    return _build_test_config(tmp_path)


def _build_test_config(tmp_path) -> AppConfig:
    return AppConfig(
        environment="dev",
        project_name="ml-churn-test",
        package_name="churn_model",
        spark=SparkSettings(app_name="churn_model-tests", master="local[2]", shuffle_partitions=2),
        data=DataSettings(
            source="local",
            raw_path="data/sample/customers_churn.csv",
            target_column="churn",
            id_column="customer_id",
            required_columns=(
                "customer_id",
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "contract_type",
                "internet_service",
                "support_calls",
                "is_active",
                "signup_date",
                "churn",
            ),
            categorical_columns=("contract_type", "internet_service"),
            numeric_columns=("tenure_months", "monthly_charges", "total_charges", "support_calls"),
            quality=QualitySettings(
                max_null_percentage=0.05,
                max_duplicate_percentage=0.01,
                min_rows=1,
                max_rows=1_000_000,
                max_volume_change_percentage=0.5,
            ),
        ),
        features=FeaturesSettings(
            version="1",
            numeric_columns=("tenure_months", "monthly_charges", "total_charges", "support_calls"),
            categorical_columns=("contract_type", "internet_service"),
            label_column="churn",
        ),
        model=ModelSettings(
            type="spark_ml",
            algorithm="logistic_regression",
            primary_metric="f1_score",
            hyperparameters={"max_iter": 10, "reg_param": 0.01, "elastic_net_param": 0.0},
            thresholds=ModelThresholds(
                minimum_accuracy=0.5,
                minimum_precision=0.5,
                minimum_recall=0.5,
                minimum_f1_score=0.5,
                minimum_roc_auc=0.5,
                maximum_metric_regression=0.05,
                minimum_improvement=0.0,
            ),
        ),
        mlflow=MLflowSettings(
            tracking_uri=f"file:{tmp_path}/mlruns",
            registry_uri=f"file:{tmp_path}/mlruns",
            experiment_name="/tests/ml-churn",
            registered_model_name="test_catalog.churn.churn_model",
        ),
        train_test_split=TrainTestSplitSettings(
            train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=42
        ),
        feature_store=FeatureStoreSettings(
            enabled=True,
            catalog="",
            schema="",
            table_name="churn_features_test",
            primary_keys=("customer_id",),
            feature_names=(
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "support_calls",
                "contract_type",
                "internet_service",
                "avg_monthly_charge_ratio",
                "support_calls_per_tenure_month",
            ),
        ),
        raw={"artifacts_dir": str(tmp_path / "artifacts")},
    )
