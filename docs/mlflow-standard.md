# Padrão MLflow Tracking e Model Registry

## Tracking

Toda execução de treino (`models/registry.log_training_run`) registra, numa
única run real do MLflow (nunca um `run_id` inventado):

- parâmetros: algoritmo, versão das features, hiperparâmetros (`hp_*`);
- métricas: todas as calculadas por `models/evaluate.evaluate_model`
  (accuracy, precision, recall, f1_score, roc_auc, latency_ms,
  model_size_bytes, invalid_prediction_percentage, error_rate);
- tags obrigatórias: `project_name`, `environment`, `git_commit`,
  `git_branch`, `azure_build_id`, `azure_build_number`, `package_version`,
  `dataset_version`, `model_name`, `model_framework`, `pipeline_version`,
  `training_timestamp`, `model_alias`;
- artefatos: `feature_config.json` sempre; EDA/qualidade de dados quando
  fornecidos via `extra_artifacts`;
- o modelo (`mlflow.spark.log_model`) com `input_example` (amostra real do
  treino) e `signature` inferida (`infer_signature`), garantindo que o
  contrato de entrada/saída do modelo é conhecido e validável.

Localmente (fora do Azure DevOps), as tags de build usam o valor literal
`"local"` -- nunca um valor fabricado tentando parecer uma execução real de
CI. Em Azure DevOps, vêm de `BUILD_SOURCEVERSION`, `BUILD_SOURCEBRANCH`,
`BUILD_BUILDID`, `BUILD_BUILDNUMBER`.

## Model Registry -- somente aliases

Nomes de modelo seguem sempre `<catalog>.<schema>.<model_name>`,
parametrizados via `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`, `MODEL_NAME` --
nunca fixados no código (ver `conf/hml.yml`/`conf/prd.yml` no
`project-template`).

Os únicos três aliases usados são `champion`, `challenger` e
`previous_champion` (nomes configuráveis via
`championAlias`/`challengerAlias`/`previousChampionAlias` nos parâmetros da
pipeline). **Nunca** use os estágios legados `Staging`/`Production`/
`Archived` -- eles não compõem bem com múltiplos ambientes compartilhando um
registry e estão sendo descontinuados pelo MLflow.

## Carregar e validar o modelo após registrar

Depois de `register_model_version` + `set_alias`, o fluxo de treino do
projeto (`cli/train_model.py` seguido por `cli/evaluate_model.py`) sempre
recarrega o modelo via `models.registry.load_model_by_alias` antes de
qualquer promoção -- se o carregamento falhar, a run é tratada como inválida
e a promoção não avança. Isso garante que nunca se promove uma versão que
não consegue ser servida.

## `registryMode`: `shared_uc` vs. `separate_uc`

Ver `docs/champion-challenger.md`, seção "Modos de registry", para os passos
completos de cada modo -- a diferença central é que `separate_uc` exige
verificação de SHA-256 do artefato transferido entre metastores antes de
mover qualquer alias.

## Versão do MLflow

A plataforma foi validada com `mlflow==2.17.2` (compatível com Unity
Catalog Model Registry e aliases). Ao atualizar a versão, confirme que:

- `MlflowClient.set_registered_model_alias` / `get_model_version_by_alias`
  continuam disponíveis (API estável desde MLflow 2.9);
- o flavor `mlflow.spark` continua serializando `PipelineModel` corretamente
  (testado em `tests/integration/test_mlflow_tracking.py` e
  `tests/unit/test_registry.py`).
