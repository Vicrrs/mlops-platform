# Databricks notebook source
# MAGIC %md
# MAGIC # Experimentação de modelo - ml-churn
# MAGIC Notebook de apoio para explorar rapidamente hiperparâmetros e ver métricas.
# MAGIC Não substitui `churn_model.cli.train_model` (usado pela pipeline) -- toda a
# MAGIC lógica de treino/avaliação vem de `src/churn_model/models/{train,evaluate}.py`.

# COMMAND ----------

try:
    dbutils.widgets.text("config_path", "conf/dev.yml", "Config path")
    config_path = dbutils.widgets.get("config_path")
except NameError:
    config_path = "conf/dev.yml"

# COMMAND ----------

from churn_model.config import load_config
from churn_model.data.loader import load_raw_customers
from churn_model.models.evaluate import evaluate_model
from churn_model.models.train import prepare_datasets, train_candidate
from churn_model.spark import get_spark_session, stop_spark_session

config = load_config(config_path)
spark = get_spark_session(app_name="ml-churn-experimentation-notebook", environment=config.environment)

# COMMAND ----------

raw_df = load_raw_customers(spark, config)
train_df, validation_df, test_df = prepare_datasets(raw_df, config)

# COMMAND ----------

# MAGIC %md
# MAGIC Ajuste os hiperparâmetros aqui apenas para experimentação manual. Para
# MAGIC persistir uma mudança de hiperparâmetro, edite `conf/base.yml` (fonte da
# MAGIC verdade usada pela pipeline), não este notebook.
experimental_hyperparameters = dict(config.model.hyperparameters)
experimental_hyperparameters["max_iter"] = 100

config_with_experiment = config.__class__(
    **{**config.__dict__, "model": config.model.__class__(
        type=config.model.type,
        algorithm=config.model.algorithm,
        primary_metric=config.model.primary_metric,
        hyperparameters=experimental_hyperparameters,
        thresholds=config.model.thresholds,
    )}
)

result = train_candidate(train_df, validation_df, test_df, config_with_experiment)
metrics = evaluate_model(result.pipeline_model, validation_df, config.features.label_column)

print("Métricas de validação (experimental):", metrics)

# COMMAND ----------

stop_spark_session()
