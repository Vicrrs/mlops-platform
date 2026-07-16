# Changelog — mlops-platform

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).
Projetos consumidores devem fixar `ref: refs/tags/vX.Y.Z` -- nunca
`refs/heads/main`.

## [1.0.0] - 2026-07-16

### Adicionado

- `pipelines/templates/ml-project-pipeline.yml`: template principal com os
  22 stages do fluxo Repo → HML → aprovação manual → PRD.
- 14 templates auxiliares (`ci.yml`, `build.yml`, `deploy-hml.yml`,
  `spark-tests.yml`, `data-quality.yml`, `train-candidate.yml`,
  `register-challenger.yml`, `compare-models.yml`, `model-quality-gate.yml`,
  `approval-prd.yml`, `deploy-prd.yml`, `promote-champion.yml`,
  `smoke-tests.yml`, `rollback.yml`) e 10 steps reutilizáveis.
- 10 scripts de governança em `scripts/` (validação de estrutura, ambiente,
  qualidade de dados, métricas, artefato, comparação Champion/Challenger,
  promoção, verificação de metadados de promoção, smoke test, rollback),
  todos independentes de PySpark e cobertos por testes reais em `tests/`.
- `project-template/`: projeto ML completo e funcional (`churn_model`,
  classificação binária de churn) com PySpark + Spark ML + MLflow +
  Databricks Asset Bundle + 107 testes reais (unit/spark/integration/smoke).
- 10 documentos em `docs/` cobrindo arquitetura, padrões (projeto, Spark,
  MLflow), Champion/Challenger, onboarding, processo de release, variáveis
  obrigatórias, segurança e rollback.

### Decisões técnicas registradas

- Scripts centrais não dependem de PySpark (rodam em qualquer agente).
- Aliases (`champion`/`challenger`/`previous_champion`) em vez dos estágios
  legados do MLflow.
- Promoção e rollback sempre em dry-run por padrão, exigindo `--confirm`.
- `retrainInProduction` com padrão `false`; exceção precisa ser explícita.
