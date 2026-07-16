"""Fluxo completo: treinar -> registrar como Champion -> inferência em um lote novo."""

from __future__ import annotations

from churn_model.data.synthetic import generate_customers_churn_df
from churn_model.models import inference
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets, train_candidate


def test_inference_on_previously_unseen_batch(spark, sample_customers_df, test_config):
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

    # Lote completamente novo, gerado com uma seed diferente da usada no treino.
    unseen_batch = generate_customers_churn_df(spark, n_rows=50, seed=999)

    scored = inference.score(test_config, unseen_batch)
    assert scored.count() == 50
    assert scored.select("model_version").distinct().collect()[0]["model_version"] == "unknown"

    scored_with_version = inference.score(
        test_config, unseen_batch, model_alias="champion", model_version=str(version)
    )
    row = scored_with_version.select("model_version").distinct().collect()[0]
    assert row["model_version"] == str(version)
