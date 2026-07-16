# Rollback

## Quando usar

Quando o Champion atual em produção apresenta regressão de performance,
erro operacional ou qualquer comportamento inesperado detectado pelo
monitoramento (`monitoring.job.yml` → `alerts.json` com severidade
`critical`), e a decisão é voltar para uma versão anterior conhecida como
boa -- normalmente a que está (ou esteve) sob o alias `previous_champion`.

## Como funciona `scripts/rollback.py`

1. Valida que `--target-version` existe de fato no registry
   (`client.get_model_version`) -- falha explicitamente se não existir.
2. Recupera a versão atualmente sob `champion-alias` (se houver).
3. **Dry-run por padrão**: sem `--confirm`, apenas grava
   `rollback_report.json` com o plano (`from_version`, `to_version`) e não
   altera nada no registry.
4. Com `--confirm`:
   a. Se `--smoke-test-command` foi passado, executa esse comando (tipo-
      icamente disparando `resources/smoke-test.job.yml` contra a versão
      alvo); se ele falhar, o rollback é abortado ANTES de qualquer troca
      de alias.
   b. Move o alias `champion` atual para `previous-champion-alias` (permite
      rollback do rollback, se necessário).
   c. Atribui `champion-alias` à `--target-version`.
5. Grava `rollback_report.json` com autor, motivo, build, timestamps e
   (quando aplicável) se o smoke test passou.

Nenhuma versão é apagada em nenhum momento -- rollback é sempre reatribuição
de alias.

## Uso via CLI (local ou dentro de uma pipeline dedicada)

```bash
# 1) Sempre comece em dry-run (padrão):
python scripts/rollback.py \
  --project ml-fraude \
  --environment prd \
  --tracking-uri databricks --registry-uri databricks-uc \
  --registered-model-name prd_catalog.fraude.modelo_fraude \
  --target-version 12 \
  --author "jane.doe@empresa.com" \
  --reason "Regressão de recall detectada em produção (alerta crítico #482)" \
  --build-id 4821

# 2) Revise artifacts/rollback/rollback_report.json -- confirme from_version/to_version.

# 3) Execute de fato, com smoke test bloqueante:
python scripts/rollback.py \
  --project ml-fraude --environment prd \
  --tracking-uri databricks --registry-uri databricks-uc \
  --registered-model-name prd_catalog.fraude.modelo_fraude \
  --target-version 12 \
  --author "jane.doe@empresa.com" \
  --reason "Regressão de recall detectada em produção (alerta crítico #482)" \
  --build-id 4821 \
  --smoke-test-command "databricks bundle run smoke_test_job -t prd" \
  --confirm
```

## Via pipeline dedicada (`pipelines/templates/rollback.yml`)

`rollback.yml` **não** faz parte do fluxo automático de
`ml-project-pipeline.yml` -- rollback é uma ação deliberada, não algo que
deveria disparar sozinho a partir de um push. Crie uma pipeline separada
(ex.: `rollback-pipeline.yml` no repositório consumidor) com um parâmetro
manual para a versão-alvo, importando:

```yaml
extends:
  template: pipelines/templates/rollback.yml@mlops
  parameters:
    projectName: ml-fraude
    environment: prd
    registeredModelName: prd_catalog.fraude.modelo_fraude
    targetVersion: "12"
    author: "$(Build.RequestedFor)"
    reason: "Preencher o motivo ao disparar manualmente"
    confirm: false   # true apenas depois de revisar o dry-run
```

Rode primeiro com `confirm: false`, revise o artefato
`rollback-report` publicado pela pipeline, e só então rode de novo com
`confirm: true`.

## Auditoria

Todo `rollback_report.json` contém `project`, `environment`,
`registered_model_name`, `from_version`, `to_version`, `to_run_id`,
`author`, `reason`, `build_id`, `dry_run` e `timestamp` -- suficiente para
reconstruir quem decidiu o rollback, por quê, e exatamente quais versões
foram trocadas.
