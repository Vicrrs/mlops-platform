# Feature Store

## O que muda em relação à engenharia de features "normal"

Antes desta adição, features eram calculadas **dentro** do `Pipeline` de treino
(`features/transformations.py`), ajustadas junto com o modelo, e descartadas
depois — cada treino recalculava tudo do zero a partir dos dados brutos.

A feature store separa duas coisas que antes estavam juntas:

- **Cálculo de features** (`cli/run_feature_engineering.py`): roda como job
  próprio, escreve numa tabela nomeada e versionada.
- **Consumo de features** (treino e inferência): busca por nome/chave, sem
  saber como a feature foi calculada nem recalculá-la.

Isso é o que permite reaproveitar a mesma tabela de features entre execuções
de treino diferentes e, numa organização real, entre modelos diferentes que
olham para o mesmo cliente (ex.: o modelo de churn e um futuro modelo de
propensão a upgrade podem consumir a mesma tabela `churn_features`).

## Duas implementações, mesma interface

Mesmo padrão já usado por `spark.py` (sessão local vs. Databricks) e
`models/registry.py` (MLflow local vs. Unity Catalog): `get_feature_store(config, spark)`
devolve a implementação certa para o ambiente, sem o chamador precisar saber qual é.

| | `dev`/local | `hml`/`prd` |
|---|---|---|
| Implementação | `LocalFeatureStore` | `DatabricksFeatureStore` |
| Armazenamento | Delta local em `artifacts/feature_store/tables/` | Tabela gerenciada do Unity Catalog |
| Catálogo de tabelas | `artifacts/feature_store/registry.json` | Metastore do Unity Catalog |
| Lookup automático na inferência (online) | Não -- `score_batch` faz join manual | Sim -- `fe.score_batch` resolve pela assinatura do modelo |
| Dependência extra | Nenhuma (reaproveita `io.writers.merge_delta`) | `databricks-feature-engineering` (extra `feature-store` do `pyproject.toml`) |

Isso é testado de verdade em `tests/unit/test_feature_store.py` (round-trip de
escrita/leitura, upsert idempotente, join por chave primária) e em
`tests/integration/test_feature_store_training.py` (publica a tabela → monta
o training set → treina → registra no MLflow, ponta a ponta, com Spark e
MLflow reais).

## Como usar num projeto

1. **Habilitar** em `conf/<ambiente>.yml`:
   ```yaml
   feature_store:
     enabled: true
     table_name: churn_features
     primary_keys: [customer_id]
     feature_names: [tenure_months, monthly_charges, ..., avg_monthly_charge_ratio]
   ```
   Em `hml`/`prd`, `catalog`/`schema` vêm de `${DATABRICKS_CATALOG}`/`${DATABRICKS_SCHEMA}`
   (mesmas variáveis já usadas pelo resto do projeto -- ver `docs/required-variables.md`).

2. **Calcular e publicar as features** (job separado, roda antes do treino):
   ```bash
   python -m churn_model.cli.run_feature_engineering --config conf/dev.yml
   ```
   No Databricks, isso é `resources/feature-engineering.job.yml` (agendado, pausado por padrão).

3. **Treinar a partir da feature store** (em vez de calcular features ad-hoc):
   ```bash
   python -m churn_model.cli.train_model --config conf/dev.yml --use-feature-store
   ```
   Por baixo, `models/train.prepare_datasets_from_feature_store` recebe só
   chaves + rótulo, faz o lookup na tabela publicada, e segue o mesmo fluxo de
   sempre (split, treino, avaliação, registro).

## Adicionando uma feature nova

Só em dois lugares:

1. `features/feature_store.compute_customer_features` -- calcula a coluna nova.
2. `conf/base.yml` -- adiciona o nome na lista `feature_store.feature_names`
   (e em `features.numeric_columns`/`categorical_columns`, se quiser que o
   modelo de fato a utilize no `VectorAssembler`).

Não precisa mexer no treino, na inferência, nem nos scripts centrais -- eles
não sabem (nem precisam saber) como cada feature é calculada.

## Limitações conhecidas

- `LocalFeatureStore` não tem serving online: `score_batch` faz um join
  manual, então só serve para lotes onde as chaves já existem na tabela de
  features. Em produção real (`DatabricksFeatureStore`), o
  `fe.score_batch`/`fe.log_model` registram a proveniência da feature no
  próprio modelo, permitindo lookup automático a partir só das chaves.
- O client `databricks-feature-engineering` (testado na versão `0.6.0`) ainda
  importa `pkg_resources`, removido do `setuptools` a partir da v81 -- se o
  ambiente falhar ao importar, prenda `setuptools<81` (o Databricks Runtime
  já traz uma versão compatível, isso só afeta ambientes locais/CI).
- Nunca testado contra um Unity Catalog real nesta sessão (sem workspace
  disponível) -- `DatabricksFeatureStore.create_or_update_table` foi validado
  localmente apenas até o ponto em que o Unity Catalog rejeita o catálogo
  inexistente (`Catalog 'dev_catalog' does not exist in the metastore`),
  confirmando que o client é chamado corretamente.
