# Onboarding: criando um novo projeto de ML

Passos para um time criar um projeto novo (ex.: `ml-risco-credito`) sobre a
plataforma.

## 1. Criar o repositório e copiar o template

```bash
git clone <AZURE_DEVOPS_PROJECT>/ml-risco-credito
cp -r mlops-platform/project-template/* ml-risco-credito/
cp -r mlops-platform/project-template/.gitignore ml-risco-credito/ 2>/dev/null || true
cd ml-risco-credito
```

Renomeie o pacote (`src/churn_model` → `src/risco_credito_model`) e ajuste:

- `pyproject.toml`: `name`, `[project.scripts]` (entry points), imports;
- `databricks.yml`: `bundle.name`, variáveis `schema`/`model_name`;
- `resources/*.job.yml`: `package_name`/`entry_point` de cada
  `python_wheel_task`;
- `conf/base.yml`: `project_name`, `package_name`, colunas/schema do seu
  domínio;
- `azure-pipelines.yml`: `projectName`, `packageName`, `artifactName`, e os
  limites de qualidade/métricas do seu problema.

## 2. Adaptar o domínio

Dentro de `src/<pacote>/`:

- `data/schemas.py`: schema real dos seus dados brutos;
- `data/synthetic.py` (opcional): gerador de dados sintéticos para
  desenvolvimento local sem depender de dados reais;
- `features/transformations.py` / `features/pipeline.py`: features e
  algoritmo do seu problema;
- `models/train.py` / `models/evaluate.py`: ajuste apenas se as métricas de
  avaliação padrão (accuracy/precision/recall/F1/ROC AUC) não fizerem
  sentido para seu problema (ex.: regressão usaria RMSE/MAE -- adapte
  `models/evaluate.py` mantendo a mesma interface de retorno usada por
  `champion_challenger.py`).

Não precisa tocar em `spark.py`, `io/`, `models/registry.py`,
`models/champion_challenger.py` nem `monitoring/` -- são genéricos.

## 3. Configurar o Azure DevOps

Peça ao time de MLOps (ou, se você tiver permissão, configure):

- Environments `databricks-hml` e `databricks-prd`, com approval check no
  segundo;
- Variable Groups `vg-databricks-hml` e `vg-databricks-prd` (ver
  `docs/required-variables.md`);
- Branch policy na `main` exigindo PR + build validation
  (ver `docs/security.md`, seção Governança).

## 4. Rodar localmente antes do primeiro push

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,spark,mlflow]"
pytest -v
python -m <pacote>.cli.run_data_quality --config conf/dev.yml
python -m <pacote>.cli.train_model --config conf/dev.yml
python -m <pacote>.cli.evaluate_model --config conf/dev.yml --enforce
```

Só faça o primeiro commit depois que os testes e o fluxo local passarem --
ver `project-template/README.md` para o passo a passo completo.

## 5. Primeiro PR

O primeiro PR deve conter só a estrutura adaptada (sem alterar
`pipelines/` do repositório central). A pipeline vai rodar
`ValidateProject`/`Lint`/`UnitTests`/`SparkTests` automaticamente; PRs nunca
disparam deploy em HML ou PRD (ver `docs/release-process.md`).

## 6. Pedir revisão do time de MLOps quando necessário

Mudanças em `azure-pipelines.yml`, `databricks.yml`, `resources/**`,
`conf/**`, `src/**/io/**`, `src/**/models/registry.py` ou
`src/**/models/champion_challenger.py` exigem revisão do time de MLOps (ver
`docs/security.md`).
