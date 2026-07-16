from __future__ import annotations

import pytest

from churn_model.config import load_config
from churn_model.exceptions import ConfigurationError


def test_load_dev_config_merges_base_and_env():
    config = load_config("conf/dev.yml")
    assert config.environment == "dev"
    assert config.project_name == "ml-churn"
    assert config.data.raw_path == "data/sample/customers_churn.csv"
    # Vem de base.yml, deve sobreviver ao merge com dev.yml:
    assert "customer_id" in config.data.required_columns
    assert config.model.algorithm == "logistic_regression"


def test_load_config_missing_file_raises():
    with pytest.raises(ConfigurationError):
        load_config("conf/does-not-exist.yml")


def test_load_hml_config_resolves_env_placeholders(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABRICKS_CATALOG", "hml_catalog")
    monkeypatch.setenv("DATABRICKS_SCHEMA", "churn")
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "hml_catalog.churn.churn_model")

    config = load_config("conf/hml.yml")
    assert config.data.catalog == "hml_catalog"
    assert config.data.schema == "churn"
    assert config.mlflow.registered_model_name == "hml_catalog.churn.churn_model"


def test_load_hml_config_without_env_uses_default_placeholder(monkeypatch):
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    config = load_config("conf/hml.yml")
    assert config.mlflow.experiment_name == "/Shared/ml-churn/hml"
