# Variáveis obrigatórias (Variable Groups / ambiente)

Nenhum valor real aparece nos exemplos abaixo -- todos são placeholders.
Configure-os no Azure DevOps (Pipelines → Library → Variable Groups) e/ou no
Databricks Job (Job parameters), nunca no código ou no YAML versionado.

## `vg-databricks-hml` / `vg-databricks-prd`

| Variável | Descrição | Secreta? |
|---|---|---|
| `DATABRICKS_HOST` | URL do workspace Databricks do ambiente (ex.: `https://<workspace>.azuredatabricks.net`) | Não |
| `DATABRICKS_CLIENT_ID` | Application ID do Service Principal (auth via OIDC/Workload Identity Federation) | Não |
| `DATABRICKS_CLIENT_SECRET` | Secret do Service Principal, **somente** se OIDC/WIF não estiver disponível no tenant | **Sim** |
| `AZURE_CLIENT_ID` | Client ID usado na federação de identidade Azure DevOps ↔ Entra ID | Não |
| `AZURE_TENANT_ID` | Tenant ID do Entra ID | Não |
| `DATABRICKS_CATALOG` | Catálogo Unity Catalog do ambiente (ex.: `hml_catalog`, `prd_catalog`) | Não |
| `DATABRICKS_SCHEMA` | Schema do projeto dentro do catálogo | Não |
| `MLFLOW_EXPERIMENT_NAME` | Caminho do experimento MLflow (ex.: `/Shared/ml-churn/hml`) | Não |
| `MLFLOW_REGISTERED_MODEL_NAME` | `<catalog>.<schema>.<model_name>` completo | Não |

## Variáveis injetadas automaticamente pelo Azure DevOps

Não precisam ser configuradas -- já existem em toda execução:

| Variável | Uso na plataforma |
|---|---|
| `BUILD_SOURCEVERSION` | Vira `git_commit` nas tags do MLflow e em `run_metadata.json` |
| `BUILD_SOURCEBRANCH` | Vira `git_branch` |
| `BUILD_BUILDID` / `BUILD_BUILDNUMBER` | Vira `azure_build_id` / `azure_build_number` |
| `SYSTEM_ACCESSTOKEN` | Usado apenas se a pipeline precisar chamar a REST API do Azure DevOps (ex.: para ler status de outro pipeline); habilite `System.AccessToken` explicitamente no job que precisar |

## Quando rodar localmente (ambiente `dev`)

Nenhuma das variáveis acima é necessária. `conf/dev.yml` usa:

- `data.source: local` (lê `data/sample/*.csv`);
- `mlflow.tracking_uri: file:./mlruns` (sem servidor MLflow);
- `git_commit=local`, `azure_build_id=local` etc. (ver
  `models.registry.GitContext.from_environment`).

## JDBC (quando aplicável)

| Variável | Descrição |
|---|---|
| `JDBC_URL` | String de conexão sem credenciais embutidas |
| `JDBC_USER` | Usuário (via secret) |
| `JDBC_PASSWORD` | Senha (via secret) |

`io/readers.read_jdbc` e `io/writers.write_jdbc` recusam-se a rodar
(`DataReadError`/`DataWriteError`) se `user`/`password` vierem vazios --
força que o chamador resolva essas variáveis a partir de um secret scope
antes de invocar a função, nunca como literal.

## Databricks Secret Scopes (recomendado para segredos usados DENTRO do job)

Dentro de um job Databricks, prefira `dbutils.secrets.get(scope, key)` para
qualquer segredo consumido pelo próprio cluster (ex.: credenciais JDBC de uma
fonte de dados on-premises), em vez de variáveis de ambiente do Azure
DevOps -- o Secret Scope é o mecanismo nativo do workspace e evita que o
segredo trafegue pelo agente de CI.
