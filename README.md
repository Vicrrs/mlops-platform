# mlops-platform

Repositório central de MLOps: templates de pipeline Azure DevOps, scripts de
governança (validação, comparação Champion/Challenger, promoção, rollback) e
o template de projeto (`project-template/`) que qualquer time de ML usa como
ponto de partida. Ver `docs/architecture.md` para a visão completa.

## Estrutura

```
mlops-platform/
├── azure-pipelines.yml       # CI deste próprio repositório
├── pipelines/
│   ├── templates/             # 15 templates (importados via `extends`)
│   └── steps/                 # 10 steps reutilizáveis
├── scripts/                   # 10 scripts de governança (sem PySpark)
├── project-template/          # projeto ML completo e funcional (churn_model)
├── tests/                     # testes dos scripts de governança
├── docs/                      # 10 documentos (arquitetura, padrões, segurança...)
├── requirements-dev.txt
└── pyproject.toml
```

## O que este repositório NÃO faz

- Não recebe cópias do código de modelo dos times (isso vive nos
  repositórios consumidores).
- Não faz deploy real nem promove modelos automaticamente -- toda promoção
  para PRD depende de aprovação manual (Azure DevOps Environment).
- Não versiona nenhuma credencial. Ver `docs/security.md` e
  `docs/required-variables.md`.

## Rodando os testes deste repositório localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Lint + testes dos scripts de governança (rápido, sem Spark):
ruff check scripts tests
pytest tests -v

# Validação de sintaxe de todos os templates/steps:
python - <<'PY'
import glob, yaml
for f in sorted(glob.glob("pipelines/steps/*.yml")) + sorted(glob.glob("pipelines/templates/*.yml")):
    yaml.safe_load(open(f))
    print("OK", f)
PY
```

Para rodar o exemplo funcional completo (PySpark + MLflow reais), veja
`project-template/README.md` -- ele treina um modelo real, registra no
MLflow, compara Champion/Challenger e roda inferência, tudo localmente sem
nenhum workspace Databricks.

## Como um projeto consumidor usa este repositório

```yaml
# azure-pipelines.yml do projeto (ex.: ml-fraude), completo:
resources:
  repositories:
    - repository: mlops
      type: git
      name: <AZURE_DEVOPS_PROJECT>/mlops-platform
      ref: refs/tags/v1.0.0   # sempre uma tag, nunca refs/heads/main

extends:
  template: pipelines/templates/ml-project-pipeline.yml@mlops
  parameters:
    projectName: ml-fraude
    packageName: fraude_model
    # ... (ver project-template/azure-pipelines.yml para o exemplo completo)
```

Ver `docs/onboarding.md` para o passo a passo de criar um projeto novo.

## Documentação

| Documento | Conteúdo |
|---|---|
| `docs/architecture.md` | Visão geral, por que scripts centrais não usam Spark, aliases vs. estágios |
| `docs/project-standard.md` | Estrutura obrigatória de projeto, regras de notebook fino |
| `docs/spark-standard.md` | Padrões de SparkSession, I/O, EDA, Spark ML Pipeline |
| `docs/mlflow-standard.md` | Tags/params/metrics obrigatórios, aliases, registry |
| `docs/champion-challenger.md` | Fluxo completo, regras de aprovação, `shared_uc` vs. `separate_uc` |
| `docs/onboarding.md` | Como criar um projeto novo |
| `docs/release-process.md` | Os 22 stages, comportamento PR/main/PRD, `retrainInProduction` |
| `docs/required-variables.md` | Toda variável/secret necessária, por Variable Group |
| `docs/security.md` | Autenticação, segredos, governança de mudanças |
| `docs/rollback.md` | Como e quando rodar rollback, dry-run, auditoria |

## Versionamento

Este repositório é consumido por tag (`refs/tags/vX.Y.Z`). Ver
`CHANGELOG.md` para o histórico de versões do template.
