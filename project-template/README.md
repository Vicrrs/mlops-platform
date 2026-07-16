# ml-churn (template de projeto sobre a mlops-platform)

Exemplo funcional de projeto de Machine Learning construído sobre os padrões
do repositório central [`mlops-platform`](../README.md): classificação binária
de churn de clientes com PySpark + Spark ML + MLflow, incluindo EDA
automatizada, qualidade de dados, estratégia Champion/Challenger e
monitoramento.

Este diretório serve dois propósitos:

1. É o **template** que o time de MLOps distribui para novos projetos (`docs/onboarding.md` no repositório central explica como instanciá-lo).
2. É um **exemplo real e executável**, com dados sintéticos incluídos, para provar que o padrão funciona de ponta a ponta sem depender de nenhum workspace Databricks.

## Estrutura

```
churn_model/
├── config.py            # Configuração em camadas (base + ambiente)
├── spark.py              # SparkSession local ou Databricks, com Delta Lake
├── io/                   # Leitura/escrita padronizada (Delta, Parquet, CSV, JSON, JDBC)
├── data/                 # Schema, validação, qualidade de dados, EDA
├── features/             # Transformações e Spark ML Pipeline
├── models/               # Treino, avaliação, registry MLflow, Champion/Challenger, inferência
├── monitoring/           # Monitoramento de dados, previsões e performance
└── cli/                  # Um entry point por job Databricks (resources/*.job.yml)
```

## Pré-requisitos locais

- Python 3.11/3.12
- Java 17 (necessário para o PySpark). Se não houver JDK instalado:
  ```bash
  curl -sSL -o /tmp/jdk17.tar.gz \
    "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk"
  mkdir -p ~/.local/opt/jdk17 && tar -xzf /tmp/jdk17.tar.gz -C ~/.local/opt/jdk17 --strip-components=1
  export JAVA_HOME=~/.local/opt/jdk17
  export PATH="$JAVA_HOME/bin:$PATH"
  ```
- A primeira execução baixa os jars do Delta Lake via Maven (necessita rede); execuções seguintes usam o cache local (`~/.ivy2`).

## Execução local

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,spark,mlflow]"

# Testes
pytest -v
pytest --cov=churn_model --cov-report=term-missing --cov-report=xml

# Pipeline funcional completa (equivalente aos jobs do Databricks Asset Bundle)
python -m churn_model.cli.run_data_quality --config conf/dev.yml
python -m churn_model.cli.run_eda --config conf/dev.yml
python -m churn_model.cli.train_model --config conf/dev.yml
python -m churn_model.cli.evaluate_model --config conf/dev.yml --enforce
python -m churn_model.cli.run_inference --config conf/dev.yml --model-alias challenger

# MLflow UI (para inspecionar runs, métricas e o modelo registrado)
mlflow ui --backend-store-uri ./mlruns
```

Por padrão, `conf/dev.yml` usa `data/sample/customers_churn.csv` (dados
sintéticos determinísticos, gerados por `churn_model.data.synthetic`) e um
MLflow local baseado em arquivo (`file:./mlruns`) -- nada disso depende de
credenciais ou de um workspace Databricks.

### Promovendo o Challenger para Champion localmente

A promoção real (mudança de alias) é feita pelos scripts do repositório
central, não pelo projeto -- isso mantém a governança centralizada. Exemplo
(usando os scripts em `../scripts/` a partir daqui):

```bash
# 1) Um aprovador registra a decisão (normalmente isso vem do gate manual do
#    Azure DevOps Environment; aqui simulamos localmente):
cat > artifacts/approval.json <<'JSON'
{"approver": "seu-usuario@empresa.com", "run_id": "<mlflow_run_id do challenger>", "model_version": <versão do challenger>}
JSON

# 2) Verifica consistência entre recomendação técnica, metadados e aprovação:
python ../scripts/verify_promotion_metadata.py \
  --promotion-recommendation-path artifacts/model_comparison/promotion_recommendation.json \
  --run-metadata-path artifacts/training/run_metadata.json \
  --approval-path artifacts/approval.json

# 3) Promove de fato (dry-run por padrão; use --confirm para aplicar):
python ../scripts/promote_model.py \
  --tracking-uri file:./mlruns --registry-uri file:./mlruns \
  --registered-model-name dev_catalog.churn.churn_model \
  --challenger-version <versão do challenger> \
  --promotion-recommendation-path artifacts/model_comparison/promotion_recommendation.json \
  --run-metadata-path artifacts/training/run_metadata.json \
  --approval-path artifacts/approval.json \
  --approver "seu-usuario@empresa.com" --build-id local-demo --confirm

# 4) Smoke test do novo Champion:
python -m churn_model.cli.run_smoke_test --config conf/dev.yml
```

## Databricks Asset Bundle

```bash
databricks bundle validate -t dev   # requer autenticação com um workspace real
databricks bundle deploy -t hml     # NÃO execute sem revisão -- fora do escopo desta entrega
```

Este repositório nunca executa `bundle deploy` automaticamente; isso é feito
pela pipeline central após todos os quality gates (ver
`../docs/release-process.md`).

## Variáveis de ambiente externas (nunca fixadas no código)

Ver `../docs/required-variables.md` para a lista completa. Resumo: quando
`data.source: unity_catalog` (ambientes `hml`/`prd`), são necessárias
`DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`, `MLFLOW_EXPERIMENT_NAME` e
`MLFLOW_REGISTERED_MODEL_NAME`, além da autenticação do cluster (Service
Principal / Workload Identity Federation -- nunca PAT em texto).
