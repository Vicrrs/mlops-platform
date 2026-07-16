# Changelog — ml-churn (project-template)

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

## [0.1.0] - 2026-07-15

### Adicionado
- Estrutura inicial do projeto sobre os padrões da `mlops-platform`.
- Pacote `churn_model` completo: config, Spark session, I/O, qualidade de
  dados, EDA automatizada, features (Spark ML Pipeline), treino, avaliação,
  registro MLflow, Champion/Challenger, inferência e monitoramento.
- Dataset sintético determinístico em `data/sample/customers_churn.csv`.
- Databricks Asset Bundle (`databricks.yml` + `resources/*.job.yml`) com os 7
  jobs: qualidade de dados, EDA, treino, validação de modelo, inferência em
  lote, monitoramento e smoke test.
- Suíte de testes unit/spark/integration/smoke.
