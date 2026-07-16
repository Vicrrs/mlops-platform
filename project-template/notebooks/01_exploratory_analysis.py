# Databricks notebook source
# MAGIC %md
# MAGIC # EDA - ml-churn
# MAGIC Este notebook **não contém lógica de produção**. Ele apenas chama funções
# MAGIC testadas do pacote `churn_model`, recebe parâmetros via widgets e exibe os
# MAGIC resultados para apoiar a exploração manual. Toda a lógica reutilizável está
# MAGIC em `src/churn_model/data/eda.py` (com testes em `tests/spark/test_spark_eda.py`).

# COMMAND ----------

try:
    dbutils.widgets.text("config_path", "conf/dev.yml", "Config path")
    dbutils.widgets.text("output_dir", "artifacts/eda", "Output dir")
    config_path = dbutils.widgets.get("config_path")
    output_dir = dbutils.widgets.get("output_dir")
except NameError:
    # Executando fora do Databricks (ex.: teste local via `python notebooks/...py`).
    config_path = "conf/dev.yml"
    output_dir = "artifacts/eda"

# COMMAND ----------

from churn_model.config import load_config
from churn_model.data.eda import run_eda, write_eda_artifacts
from churn_model.data.loader import load_raw_customers
from churn_model.spark import get_spark_session, stop_spark_session

config = load_config(config_path)
spark = get_spark_session(app_name="ml-churn-eda-notebook", environment=config.environment)

# COMMAND ----------

raw_df = load_raw_customers(spark, config)
try:
    display(raw_df.limit(20))  # noqa: F821 - `display` só existe no runtime Databricks
except NameError:
    raw_df.limit(20).show(truncate=False)

# COMMAND ----------

summary = run_eda(
    raw_df,
    required_columns=config.data.required_columns,
    numeric_columns=config.features.numeric_columns,
    categorical_columns=config.features.categorical_columns,
    target_column=config.features.label_column,
    date_columns=("signup_date",),
    dataset_name=config.project_name,
)
write_eda_artifacts(summary, output_dir)

print(f"Linhas: {summary['row_count']}, Colunas: {summary['column_count']}")
print("Distribuição do alvo:", summary["target_distribution"])

# COMMAND ----------

stop_spark_session()
