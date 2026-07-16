# Segurança e governança

## Segredos -- o que nunca aparece no código ou no Git

PAT, token Databricks, senha, client secret, chave privada, credenciais
JDBC, host sensível ou connection string completa. Ver
`docs/required-variables.md` para onde cada valor deve viver.

## Autenticação recomendada

Em ordem de preferência:

1. **Workload Identity Federation / OIDC** entre Azure DevOps e Entra ID --
   nenhum secret de longa duração armazenado em lugar nenhum.
2. **Service Principal** com secret rotacionado, armazenado como variável
   secreta do Variable Group (nunca em texto no YAML).
3. **Managed Identity** para recursos Azure que suportam (ex.: acesso a
   Storage Account a partir de um cluster Databricks).
4. **Databricks Secret Scopes** para segredos consumidos de dentro do
   cluster (JDBC, APIs externas).

PAT (Personal Access Token) é o último recurso e deve ter data de expiração
curta e escopo mínimo quando inevitável.

## Detecção de segredos e dependências vulneráveis

- `SecurityScan` (dentro de `templates/ci.yml`) roda `pip-audit` sobre as
  dependências resolvidas do projeto e registra um aviso (não bloqueia por
  padrão -- ajuste `failOnFinding` no seu processo se sua política exigir
  bloqueio automático).
- Nenhum dos scripts centrais (`scripts/*.py`) loga valores de variáveis de
  ambiente que possam ser segredos -- apenas nomes de variáveis ausentes
  (`validate_environment.py`).

## Governança de mudanças

- **Pull Request obrigatório** e **proibição de push direto na `main`** em
  todo repositório (branch policy no Azure Repos).
- **Build validation** (a pipeline de CI) obrigatória antes do merge.
- **Comentários resolvidos** obrigatórios antes do merge.
- **Aprovação do time responsável** pelo domínio do modelo.
- **Aprovação adicional do time de MLOps** para mudanças nos seguintes
  caminhos (configurar via `CODEOWNERS` ou branch policy com revisores
  obrigatórios por path):

  ```
  azure-pipelines.yml
  databricks.yml
  resources/**
  conf/**
  infrastructure/**
  src/**/io/**
  src/**/models/registry.py
  src/**/models/champion_challenger.py
  ```

  Esses caminhos controlam onde os dados são lidos/gravados e como o
  Model Registry é manipulado -- exigem revisão de quem entende os
  controles de produção da plataforma, não só do domínio do modelo.

- **Versionamento do template central** por tag Git (`refs/tags/vX.Y.Z`).
  Nenhum projeto consumidor deve importar `refs/heads/main` do
  `mlops-platform` -- isso tornaria uma mudança não revisada no template
  imediatamente ativa em todos os projetos. Atualizações de versão são um
  PR deliberado no repositório consumidor, alterando o `ref:` do recurso
  `mlops`.
- **Rastreabilidade código-dado-modelo**: toda promoção/rollback registra
  `git_commit`, `dataset_version`, `mlflow_run_id` e `model_version`
  simultaneamente (ver `docs/release-process.md`, seção Rastreabilidade).

## O que a plataforma explicitamente recusa fazer

- Promover um modelo sem aprovação manual (`ApprovalPRD` é sempre um
  Environment check, nunca automatizado).
- Mudar o alias `champion` antes do quality gate (`ModelQualityGate`
  sempre precede `ApprovalPRD` na ordem de dependências dos stages).
- Reter (`--confirm` ausente) qualquer alteração de `promote_model.py` ou
  `rollback.py` -- ambos são dry-run por padrão.
- Apagar uma versão de modelo durante rollback -- apenas reatribui aliases.
