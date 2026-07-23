# Plataforma MLOps — guia da esteira

> Guia de onboarding preparado para **Guilherme Faval** e para os times que
> desenvolvem, publicam e operam modelos de Machine Learning.

Este repositório é a plataforma central de MLOps. Ele padroniza como um
projeto de ML é validado, empacotado, implantado no Databricks, registrado no
MLflow, comparado com o modelo atual e promovido para produção com aprovação
humana e rastreabilidade.

Em uma frase: o cientista desenvolve o modelo no repositório do projeto; esta
plataforma fornece a esteira e as regras para levar esse modelo de um Pull
Request até produção com segurança.

## Sumário

- [O que é MLOps](#o-que-é-mlops)
- [O papel deste repositório](#o-papel-deste-repositório)
- [Do que a plataforma é composta](#do-que-a-plataforma-é-composta)
- [Arquitetura](#arquitetura)
- [Como a pipeline funciona](#como-a-pipeline-funciona)
- [Feature Store](#feature-store)
- [MLflow e Champion/Challenger](#mlflow-e-championchallenger)
- [Como começar um projeto](#como-começar-um-projeto)
- [Tutorial de comandos](#tutorial-de-comandos)
- [Configuração no Azure DevOps](#configuração-no-azure-devops)
- [Operação e solução de problemas](#operação-e-solução-de-problemas)

## O que é MLOps

MLOps aplica práticas de engenharia e operação ao ciclo de vida de Machine
Learning. Um modelo não é apenas um arquivo treinado: ele depende de código,
dados, features, parâmetros, métricas, ambiente e uma versão reproduzível.

Nesta plataforma, MLOps cobre:

- validação do código e da estrutura do projeto;
- testes Python, Spark e de integração;
- qualidade e exploração dos dados;
- geração e reutilização de features;
- treinamento, avaliação e registro de modelos;
- comparação do candidato com o modelo em produção;
- aprovação, deploy e rollback controlados;
- monitoramento e vínculo entre commit, build, dados, run e modelo.

## O papel deste repositório

Existem dois tipos de repositório:

1. **`mlops-platform` (este repositório):** contém templates compartilhados,
   scripts de governança, documentação e um projeto de exemplo.
2. **Repositório consumidor (por exemplo, `ml-churn` ou `ml-fraude`):**
   contém o código e as configurações do modelo do time.

O código dos modelos **não é copiado para `mlops-platform`**. O projeto
consumidor importa a pipeline central por uma tag imutável:

```yaml
resources:
  repositories:
    - repository: mlops
      type: git
      name: <PROJETO_AZURE_DEVOPS>/mlops-platform
      ref: refs/tags/v1.0.0

extends:
  template: pipelines/templates/ml-project-pipeline.yml@mlops
  parameters:
    projectName: ml-churn
    packageName: churn_model
    artifactName: ml-churn
```

Usar uma tag, em vez de `main`, impede que uma alteração central mude a
esteira de todos os projetos sem revisão. A atualização é intencional:
troca-se a tag no Pull Request do projeto consumidor.

## Do que a plataforma é composta

```text
mlops-platform/
├── azure-pipelines.yml       # CI da própria plataforma
├── pipelines/
│   ├── templates/            # stages reutilizáveis da esteira
│   └── steps/                # passos menores reutilizáveis
├── scripts/                  # validações, promoção e rollback
├── project-template/         # projeto ML funcional de referência
├── tests/                    # testes dos scripts centrais
├── docs/                     # padrões e documentação detalhada
├── pyproject.toml
└── requirements-dev.txt
```

Os principais componentes são:

| Componente | Responsabilidade |
|---|---|
| Azure DevOps Pipelines | Orquestra CI, HML, aprovação e PRD |
| Databricks Asset Bundles | Define e implanta jobs e recursos por ambiente |
| Apache Spark / Spark ML | Processa dados, cria features, treina e pontua |
| MLflow | Registra runs, parâmetros, métricas, artefatos e versões do modelo |
| Unity Catalog | Governa tabelas, Feature Store e Model Registry em HML/PRD |
| Feature Store | Publica features versionadas e reutilizáveis |
| Scripts centrais | Aplicam quality gates, promoção, auditoria e rollback |
| Pytest, Ruff e pip-audit | Testes, qualidade de código e segurança |

Os scripts centrais são Python puro e não executam scoring com Spark. O
processamento pesado ocorre nos jobs Databricks do projeto; os scripts leem
os JSONs gerados pelos jobs e aplicam as regras de governança. Assim, a
governança roda em um agente comum do Azure DevOps e não depende de cluster.

## Arquitetura

```text
Desenvolvedor
     │ push / Pull Request
     ▼
Repositório do modelo ── importa por tag ──► mlops-platform
     │                                      (templates + regras)
     ▼
Azure DevOps Pipeline
     │
     ├── CI: estrutura, lint, testes, segurança e build
     ├── HML: deploy, dados, features, treino e comparação
     ├── Gate técnico: métricas e Champion/Challenger
     ├── Gate humano: aprovação no Environment de PRD
     └── PRD: mesmo artefato, promoção, smoke test e monitoramento
                 │
                 ├── Databricks / Spark
                 ├── Unity Catalog / Feature Store
                 └── MLflow Model Registry
```

O artefato de build contém a wheel do projeto, configurações, recursos do
bundle, `commit_sha.txt` e `artifact_sha256.txt`. A mesma unidade testada em
HML é implantada em PRD. Por padrão, não existe novo treinamento em produção
(`retrainInProduction: false`).

## Como a pipeline funciona

### Visão ponta a ponta

1. O desenvolvedor abre um Pull Request.
2. A CI valida estrutura, estilo, testes, dependências e bundle.
3. Depois do merge, o bundle é implantado em HML.
4. A pipeline valida dados, executa EDA e treina um candidato.
5. O candidato vira `challenger` no MLflow.
6. Challenger e Champion são avaliados sobre a mesma referência.
7. Os limites absolutos e a regressão máxima permitida são verificados.
8. Um aprovador humano autoriza a passagem para PRD.
9. A pipeline verifica a identidade do artefato e faz o deploy em PRD.
10. Os aliases são atualizados, o smoke test roda e o monitoramento é iniciado.

### Os 22 stages

| # | Stage | O que faz |
|---:|---|---|
| 1 | `ValidateProject` | Confere a estrutura obrigatória do projeto |
| 2 | `Lint` | Executa Ruff |
| 3 | `UnitTests` | Executa testes unitários e publica resultados |
| 4 | `SparkTests` | Testa sessão, transformações, I/O e pipeline Spark |
| 5 | `SecurityScan` | Audita dependências com `pip-audit` |
| 6 | `BuildArtifact` | Gera wheel e artefato imutável |
| 7 | `ValidateBundle` | Valida o Databricks Asset Bundle |
| 8 | `DeployHML` | Implanta o bundle em homologação |
| 9 | `DataQualityHML` | Verifica nulos, duplicidades e regras dos dados |
| 10 | `AutomatedEDAHML` | Gera análise exploratória automatizada |
| 11 | `TrainCandidateHML` | Treina o modelo candidato em HML |
| 12 | `EvaluateCandidateHML` | Aplica limites mínimos de métricas |
| 13 | `RegisterChallenger` | Registra a versão com alias `challenger` |
| 14 | `CompareChampionChallenger` | Compara candidato e modelo vigente |
| 15 | `IntegrationTestsHML` | Testa integrações reais do fluxo |
| 16 | `ModelQualityGate` | Bloqueia promoção sem qualidade suficiente |
| 17 | `ApprovalPRD` | Aguarda aprovação humana no Azure DevOps |
| 18 | `VerifyPromotionArtifact` | Confere commit, run e versão aprovados |
| 19 | `DeployPRD` | Implanta em produção o artefato testado |
| 20 | `PromoteChampion` | Atualiza os aliases no registry |
| 21 | `SmokeTestsPRD` | Confirma que o Champion responde em PRD |
| 22 | `StartMonitoring` | Inicia ou reativa o job de monitoramento |

Stages opcionais são controlados por parâmetros como `runSparkTests`,
`runIntegrationTests`, `runSecurityScan`, `runDataQuality`, `runEDA` e
`runSmokeTests`.

### O que roda em cada evento

| Evento | Comportamento |
|---|---|
| Pull Request para `main` | CI, testes, build e validação; nunca faz deploy em HML/PRD |
| Merge em `main` | Executa HML, treino, comparação e gates |
| Branch `release/*` | Pode chegar a HML, mas PRD continua restrito à `main` |
| Aprovação de PRD | Libera verificação, deploy, promoção e smoke test |
| Falha em qualquer gate | Interrompe as etapas dependentes |

### Quality gates

O projeto define limites no seu `azure-pipelines.yml`, por exemplo:

```yaml
minimumAccuracy: 0.60
minimumPrecision: 0.55
minimumRecall: 0.55
minimumF1Score: 0.55
minimumRocAuc: 0.60
maximumMetricRegression: 0.01
maximumNullPercentage: 0.05
maximumDuplicatePercentage: 0.01
```

Esses valores são exemplos do projeto de churn, não limites universais. Cada
time deve defini-los de acordo com o problema e registrar a justificativa no
Pull Request.

## Feature Store

A Feature Store separa o cálculo das features do treinamento do modelo.
Em vez de recalcular tudo dentro de cada treino, um job publica uma tabela
nomeada; treino e inferência consultam as features por chave.

Isso melhora:

- reutilização de features entre execuções e modelos;
- consistência entre treino e inferência;
- governança e linhagem;
- atualizações idempotentes por chave;
- redução de lógica duplicada.

### Implementações por ambiente

| Ambiente | Implementação | Armazenamento |
|---|---|---|
| `dev` local | `LocalFeatureStore` | Delta local + `registry.json` |
| `hml` / `prd` | `DatabricksFeatureStore` | Tabela gerenciada no Unity Catalog |

### Como habilitar

Em `conf/base.yml` ou `conf/<ambiente>.yml`:

```yaml
feature_store:
  enabled: true
  table_name: churn_features
  primary_keys: [customer_id]
  feature_names:
    - tenure_months
    - monthly_charges
    - total_charges
    - avg_monthly_charge_ratio
```

Publicar e depois treinar usando a Feature Store:

```bash
python -m churn_model.cli.run_feature_engineering --config conf/dev.yml
python -m churn_model.cli.train_model \
  --config conf/dev.yml \
  --use-feature-store
```

Em HML/PRD, catálogo e schema vêm de `DATABRICKS_CATALOG` e
`DATABRICKS_SCHEMA`. Para criar uma feature, implemente o cálculo em
`features/feature_store.py` e inclua seu nome na configuração. O treino não
deve conhecer os detalhes desse cálculo.

## MLflow e Champion/Challenger

O MLflow Tracking registra cada execução com parâmetros, métricas, tags e
artefatos. O Model Registry cria versões imutáveis do modelo.

A plataforma usa aliases:

- `challenger`: candidato recém-treinado;
- `champion`: versão ativa e aprovada;
- `previous_champion`: versão anterior, preservada para rollback.

Fluxo de promoção:

```text
novo treino → challenger → comparação → gate técnico → aprovação humana
           → previous_champion recebe o Champion atual
           → challenger passa a champion
```

No primeiro deploy, quando ainda não existe Champion, o candidato precisa
passar pelos limites absolutos e pela aprovação. A ausência de Champion não
elimina os gates.

A comparação gera artefatos em `artifacts/model_comparison/`, e a promoção
gera um `promotion_record.json`. O registro relaciona versão, run, commit,
build, aprovador e horário.

## Como começar um projeto

### 1. Copiar o template

Partindo do diretório que contém os dois repositórios:

```bash
cp -R mlops-platform/project-template/. ml-meu-modelo/
cd ml-meu-modelo
```

Renomeie `src/churn_model` para o pacote do projeto e ajuste:

- `pyproject.toml`: nome, pacote e comandos;
- `conf/*.yml`: fonte, colunas, métricas e ambientes;
- `databricks.yml`: bundle, catálogo, schema e modelo;
- `resources/*.job.yml`: pacote e entry points;
- `azure-pipelines.yml`: projeto, pacote, artefato e limites;
- testes e código em `src/<pacote>/`.

### 2. Configurar a pipeline consumidora

Use [`project-template/azure-pipelines.yml`](project-template/azure-pipelines.yml)
como referência. Mantenha o arquivo pequeno: ele deve declarar parâmetros e
importar o template central, não copiar os stages.

### 3. Validar antes do primeiro push

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,spark,mlflow]"

ruff check src tests
pytest tests/unit -v
pytest tests/spark -v
pytest tests/integration -v
```

### 4. Abrir o Pull Request

```bash
git checkout -b feature/primeiro-modelo
git add .
git commit -m "feat: adiciona primeiro fluxo do modelo"
git push -u origin feature/primeiro-modelo
```

Abra o PR para `main`, acompanhe os logs e corrija qualquer gate. Deploy e
promoção não devem ser feitos manualmente para contornar uma falha.

## Tutorial de comandos

Os exemplos abaixo assumem que o terminal está na raiz de
`project-template/` ou de um projeto criado a partir dele.

### Preparar o ambiente local

Pré-requisitos: Python 3.11/3.12 e Java 17.

```bash
python --version
java -version

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,spark,mlflow]"
```

Se o projeto usar o client da Feature Store Databricks localmente:

```bash
pip install -e ".[dev,spark,mlflow,feature-store]"
```

### Rodar validações rápidas

```bash
# Estilo e erros estáticos
ruff check src tests
ruff format --check src tests

# Testes separados por tipo
pytest tests/unit -v
pytest tests/spark -v
pytest tests/integration -v
pytest tests/smoke -v

# Tudo com cobertura
pytest -v \
  --cov=churn_model \
  --cov-report=term-missing \
  --cov-report=xml
```

Ao renomear o template, substitua `churn_model` pelo nome real do pacote.

### Executar o fluxo ML local

```bash
# 1. Validar a qualidade dos dados
python -m churn_model.cli.run_data_quality --config conf/dev.yml

# 2. Gerar a EDA
python -m churn_model.cli.run_eda --config conf/dev.yml

# 3. Publicar features, quando Feature Store estiver habilitada
python -m churn_model.cli.run_feature_engineering --config conf/dev.yml

# 4A. Treinar pelo fluxo tradicional
python -m churn_model.cli.train_model --config conf/dev.yml

# 4B. Ou treinar consumindo a Feature Store
python -m churn_model.cli.train_model \
  --config conf/dev.yml \
  --use-feature-store

# 5. Avaliar e falhar se as métricas não atingirem os limites
python -m churn_model.cli.evaluate_model \
  --config conf/dev.yml \
  --enforce

# 6. Executar inferência com o candidato
python -m churn_model.cli.run_inference \
  --config conf/dev.yml \
  --model-alias challenger

# 7. Monitorar dados/modelo
python -m churn_model.cli.monitor_model --config conf/dev.yml
```

### Inspecionar runs no MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Depois, abra `http://127.0.0.1:5000`. Em desenvolvimento, `conf/dev.yml`
usa dados de exemplo e MLflow baseado em arquivos, sem workspace Databricks.

### Validar um Databricks Asset Bundle

```bash
databricks auth profiles
databricks bundle validate -t dev
databricks bundle validate -t hml
```

Os comandos a seguir alteram ambiente remoto. Execute somente com autorização
e no alvo correto:

```bash
databricks bundle deploy -t hml
databricks bundle run -t hml feature_engineering_job
databricks bundle run -t hml training_job
databricks bundle run -t hml batch_inference_job
```

Em operação normal, o Azure DevOps executa deploy e jobs; o uso manual é para
diagnóstico controlado, nunca para pular gates.

### Testar a própria plataforma central

Na raiz de `mlops-platform`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

ruff check scripts tests
ruff format --check scripts tests
pytest tests -v

python scripts/validate_project_structure.py \
  --project-path project-template
```

Para testar o projeto-modelo completo:

```bash
cd project-template
pip install -e ".[dev,spark,mlflow]"
pytest tests/unit tests/spark -v
```

### Rollback

Rollback troca o alias `champion` por uma versão anterior; não recompila o
código. Consulte primeiro o modo de simulação:

```bash
python scripts/rollback.py --help
```

Rode sem confirmação para revisar o plano e use `--confirm` somente após
validar modelo, ambiente, versão-alvo e motivo. O processo gera
`rollback_report.json` para auditoria. O procedimento completo está em
[`docs/rollback.md`](docs/rollback.md).

## Configuração no Azure DevOps

Antes da primeira execução remota, são necessários:

- acesso do projeto consumidor ao repositório `mlops-platform`;
- Environments `databricks-hml` e `databricks-prd`;
- approval check obrigatório em `databricks-prd`;
- Variable Groups `vg-databricks-hml` e `vg-databricks-prd`;
- branch policy em `main`, com PR e build validation;
- autenticação por Service Principal ou Workload Identity Federation.

Variáveis principais:

| Variável | Uso |
|---|---|
| `DATABRICKS_HOST` | URL do workspace |
| `DATABRICKS_CLIENT_ID` | Identidade da automação |
| `DATABRICKS_CLIENT_SECRET` | Segredo, quando aplicável |
| `DATABRICKS_CATALOG` | Catálogo do Unity Catalog |
| `DATABRICKS_SCHEMA` | Schema do projeto |
| `MLFLOW_EXPERIMENT_NAME` | Experimento do projeto |
| `MLFLOW_REGISTERED_MODEL_NAME` | Nome completo do modelo registrado |

Nunca grave credenciais em YAML, código, `.env` versionado ou logs. A lista
completa e os métodos de autenticação estão em
[`docs/required-variables.md`](docs/required-variables.md) e
[`docs/security.md`](docs/security.md).

## Operação e solução de problemas

### “A pipeline parou em um stage”

Abra o primeiro stage que falhou, não o último que foi cancelado. Os stages
posteriores são dependentes e ficam bloqueados quando um gate anterior falha.

### “O deploy de PRD está aguardando”

Esse é o comportamento esperado do `ApprovalPRD`. Um aprovador autorizado
deve revisar métricas, comparação, commit, artefato e mudança antes de liberar
o Environment `databricks-prd`.

### “Meu modelo não virou Champion”

Verifique, nesta ordem:

1. limites absolutos do candidato;
2. `promotion_recommendation.json`;
3. regressão máxima frente ao Champion;
4. resultado dos testes de integração;
5. aprovação de PRD;
6. consistência de `run_metadata.json`;
7. smoke test.

### “A Feature Store não encontra a tabela”

Confira `feature_store.enabled`, nome da tabela, chaves primárias, catálogo,
schema, permissões no Unity Catalog e se o job de engenharia de features
executou antes do treino.

### “Spark não inicia localmente”

Confirme `java -version` (Java 17), `JAVA_HOME`, versão do Python e conexão
na primeira execução, pois os jars Delta podem ser baixados via Maven.

## Regras importantes

- nunca referencie `main` da plataforma; use uma tag;
- nunca promova um modelo ignorando os quality gates;
- nunca treine novamente em PRD sem exceção explícita e aprovada;
- nunca versione credenciais;
- mantenha notebooks finos; lógica testável deve ficar em `src/`;
- preserve os metadados de treino, promoção e rollback;
- alterações na plataforma central afetam vários times e exigem revisão.

## Documentação complementar

| Documento | Assunto |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Arquitetura e decisões técnicas |
| [`docs/onboarding.md`](docs/onboarding.md) | Criação de um projeto |
| [`docs/release-process.md`](docs/release-process.md) | Detalhes dos 22 stages |
| [`docs/feature-store.md`](docs/feature-store.md) | Feature Store local e Databricks |
| [`docs/mlflow-standard.md`](docs/mlflow-standard.md) | Padrão de tracking e registry |
| [`docs/champion-challenger.md`](docs/champion-challenger.md) | Comparação e promoção |
| [`docs/project-standard.md`](docs/project-standard.md) | Estrutura obrigatória |
| [`docs/spark-standard.md`](docs/spark-standard.md) | Padrões Spark |
| [`docs/required-variables.md`](docs/required-variables.md) | Variáveis e secrets |
| [`docs/security.md`](docs/security.md) | Segurança e governança |
| [`docs/rollback.md`](docs/rollback.md) | Procedimento de rollback |

## Checklist rápido para o Guilherme

- [ ] Entendi a diferença entre plataforma central e projeto consumidor.
- [ ] Criei o ambiente local e rodei os testes.
- [ ] Executei qualidade, EDA, treino, avaliação e inferência local.
- [ ] Sei onde consultar runs e modelos no MLflow.
- [ ] Entendi `challenger`, `champion` e `previous_champion`.
- [ ] Sei publicar e consumir features pela Feature Store.
- [ ] Configurei a pipeline consumidora usando uma tag da plataforma.
- [ ] Sei que PRD exige aprovação e recebe o artefato testado em HML.
- [ ] Sei localizar os artefatos de comparação, promoção e rollback.
