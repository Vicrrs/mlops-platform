"""Fluxo completo: publicar a feature store -> treinar a partir dela -> registrar no MLflow."""

from __future__ import annotations

from churn_model.features.feature_store import compute_customer_features, get_feature_store
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets_from_feature_store, train_candidate


def test_train_from_feature_store_end_to_end(spark, sample_customers_df, test_config):
    # 1) job de feature engineering: calcula e publica a tabela de features.
    store = get_feature_store(test_config, spark)
    features_df = compute_customer_features(sample_customers_df)
    store.create_or_update_table(features_df, test_config.feature_store)

    # 2) treino consome só chaves + rótulo, e busca as features na feature store.
    labels_df = sample_customers_df.select("customer_id", "churn")
    train_df, val_df, test_df = prepare_datasets_from_feature_store(spark, test_config, labels_df)

    assert train_df.count() + val_df.count() + test_df.count() == sample_customers_df.count()
    for feature in test_config.feature_store.feature_names:
        assert feature in train_df.columns

    # 3) o restante do fluxo (treino + registro) é idêntico ao caminho sem feature store.
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
        dataset_version="feature-store-v1",
    )
    version = register_model_version(client, test_config.mlflow.registered_model_name, run_id)
    set_alias(client, test_config.mlflow.registered_model_name, "challenger", version)

    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "f1_score" in run.data.metrics
