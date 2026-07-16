from __future__ import annotations

from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets, train_candidate
from churn_model.monitoring.data_monitoring import monitor_incoming_data, write_data_monitoring_report
from churn_model.monitoring.model_monitoring import (
    monitor_model_performance,
    write_alerts,
    write_model_performance_report,
)
from churn_model.monitoring.prediction_monitoring import (
    monitor_predictions,
    write_prediction_monitoring_report,
)


def test_monitor_incoming_data_generates_no_alerts_on_clean_data(sample_customers_df, test_config):
    payload, alerts = monitor_incoming_data(
        sample_customers_df, test_config, reference_df=None, model_version="1"
    )
    assert payload["quality_report"]["passed"] is True
    assert alerts == []


def test_monitor_incoming_data_flags_missing_required_column(sample_customers_df, test_config):
    broken_df = sample_customers_df.drop("churn")
    payload, alerts = monitor_incoming_data(broken_df, test_config, reference_df=None, model_version="1")
    assert payload["quality_report"]["passed"] is False
    assert any(a.rule == "required_columns" for a in alerts)


def test_write_data_monitoring_report(sample_customers_df, test_config, tmp_path):
    payload, _ = monitor_incoming_data(sample_customers_df, test_config, reference_df=None, model_version="1")
    path = write_data_monitoring_report(payload, tmp_path)
    assert path.exists()


def test_monitor_predictions_flags_high_error_rate(spark, test_config):
    df = spark.createDataFrame([(1.0,), (None,), (None,)], schema=["prediction"])
    payload, alerts = monitor_predictions(df, test_config, model_version="1")
    assert payload["error_rate"] > 0.01
    assert any(a.rule == "prediction_error_rate" for a in alerts)


def test_monitor_predictions_writes_report(spark, test_config, tmp_path):
    df = spark.createDataFrame([(1.0,), (0.0,)], schema=["prediction"])
    payload, _ = monitor_predictions(df, test_config, model_version="1")
    path = write_prediction_monitoring_report(payload, tmp_path)
    assert path.exists()


def test_monitor_model_performance_end_to_end(sample_customers_df, test_config, tmp_path):
    train_df, val_df, test_df = prepare_datasets(sample_customers_df, test_config)
    result = train_candidate(train_df, val_df, test_df, test_config)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)

    client = configure_mlflow(test_config)
    run_id = log_training_run(
        client=client,
        config=test_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="v1",
    )
    version = register_model_version(client, test_config.mlflow.registered_model_name, run_id)
    set_alias(client, test_config.mlflow.registered_model_name, "champion", version)

    payload, alerts = monitor_model_performance(client, test_config, test_df)
    assert payload["evaluated"] is True
    assert "f1_score" in payload["metrics"]

    path = write_model_performance_report(payload, tmp_path)
    assert path.exists()
    alerts_path = write_alerts(alerts, tmp_path)
    assert alerts_path.exists()


def test_monitor_model_performance_without_champion_returns_not_evaluated(sample_customers_df, test_config):
    client = configure_mlflow(test_config)
    payload, alerts = monitor_model_performance(client, test_config, sample_customers_df)
    assert payload["evaluated"] is False
    assert alerts == []
