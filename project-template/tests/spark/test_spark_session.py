from __future__ import annotations

from churn_model.spark import get_spark_session, is_running_on_databricks


def test_get_spark_session_returns_local_session(spark):
    assert spark is not None
    assert "local" in spark.sparkContext.master


def test_get_spark_session_reuses_active_session(spark):
    second = get_spark_session(app_name="ignored-because-reused", environment="local")
    assert second is spark


def test_is_running_on_databricks_is_false_locally(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("DB_HOME", raising=False)
    assert is_running_on_databricks() is False


def test_is_running_on_databricks_true_when_env_set(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    assert is_running_on_databricks() is True


def test_spark_session_has_delta_extension_configured(spark):
    extensions = spark.conf.get("spark.sql.extensions", "")
    assert "delta" in extensions.lower()
