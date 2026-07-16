# Estratégia Champion/Challenger

## Definições

- **Champion**: modelo atualmente aprovado e servindo em produção (alias
  `champion`).
- **Challenger**: candidato treinado em HML, ainda não aprovado para
  produção (alias `challenger`).
- **Previous Champion**: versão anterior do Champion, preservada para
  rollback rápido (alias `previous_champion`).

## Fluxo (stages `TrainCandidateHML` → `PromoteChampion`)

1. **Treino** (`resources/training.job.yml` → `churn_model.cli.train_model`):
   treina o candidato, registra a run e a versão no MLflow, atribui o alias
   `challenger`. Não altera o Champion.
2. **Avaliação absoluta** (stage `EvaluateCandidateHML` →
   `scripts/validate_model_metrics.py`): confere que o Challenger atinge os
   limites mínimos absolutos (`minimumAccuracy`, `minimumPrecision`,
   `minimumRecall`, `minimumF1Score`, `minimumRocAuc`) antes de prosseguir.
3. **Comparação** (`resources/model-validation.job.yml` →
   `churn_model.models.champion_challenger.run_champion_challenger_comparison`):
   carrega o Champion atual (se existir) e o Challenger, pontua **os dois no
   mesmo DataFrame de teste** (mesmos registros, mesma definição de métrica) e
   grava `champion_metrics.json` / `challenger_metrics.json`.
4. **Recomendação centralizada** (stage `CompareChampionChallenger` →
   `scripts/compare_champion_challenger.py`): recalcula, de forma
   independente e sem Spark, a aprovação técnica a partir dos dois JSONs de
   métricas + os limites parametrizados na pipeline. Produz
   `comparison.json`, `comparison_report.md` e `promotion_recommendation.json`.
5. **Quality gate** (`scripts/validate_model_metrics.py` dentro de
   `model-quality-gate.yml`): interrompe a pipeline se
   `technical_approval=false`.
6. **Aprovação manual** (`ApprovalPRD`, Azure DevOps Environment): só depois
   disso o `approval.json` é gerado e a promoção pode prosseguir.
7. **Promoção** (`scripts/promote_model.py`): reverifica tudo
   (`verify_promotion_metadata.py`), move `champion` atual para
   `previous_champion` e atribui `champion` à versão do Challenger. Nunca
   treina de novo.

## Quando não existe Champion (primeiro modelo)

`run_champion_challenger_comparison` detecta a ausência de alias `champion`
(`get_model_version_by_alias` retorna `None`) e trata o Challenger como
primeiro candidato: `is_first_model=true`, `champion_metric=null`,
`absolute_improvement=null`. Neste caso **todos** os limites absolutos ainda
se aplicam (não há bônus por ser o primeiro), e a aprovação manual continua
obrigatória.

## Regras de aprovação técnica

```
technical_approval =
    (challenger.accuracy  >= minimumAccuracy)  AND
    (challenger.precision >= minimumPrecision) AND
    (challenger.recall    >= minimumRecall)    AND
    (challenger.f1_score  >= minimumF1Score)   AND
    (challenger.roc_auc   >= minimumRocAuc)    AND
    (champion is None OR
     challenger[primaryMetric] >= champion[primaryMetric] - maximumMetricRegression)
```

`primaryMetric` (padrão `f1_score`) é a métrica usada na regressão relativa;
as demais métricas ainda precisam bater os limites absolutos. Um requisito
opcional de melhoria mínima (`minimum_improvement`) está disponível em
`ModelThresholds` para times que exigem `challenger > champion + X`, não
apenas "não regrediu".

## Artefatos gerados (por execução, em `artifacts/model_comparison/`)

| Arquivo | Conteúdo |
|---|---|
| `champion_metrics.json` | Métricas reais do Champion no dataset de teste (ou ausente se primeiro modelo) |
| `challenger_metrics.json` | Métricas reais do Challenger no MESMO dataset de teste |
| `comparison.json` | Os dois anteriores + o contexto da recomendação |
| `comparison_report.md` | Relatório legível por humanos (tabela lado a lado) |
| `promotion_recommendation.json` | `technical_approval`, `recommendation` (`promote`/`reject`), métricas, run_ids, versões, commit, dataset_version, timestamp |

Nenhum valor em `promotion_recommendation.json` é escrito manualmente --
todos vêm da execução real do MLflow e da comparação no dataset de teste.

## Modos de registry

### `shared_uc`

HML e PRD compartilham o mesmo Model Registry / metastore Unity Catalog. A
versão registrada em HML já é a versão final -- `promote_model.py` apenas
reatribui aliases (`champion` ← versão do challenger, `previous_champion` ←
Champion anterior). Nenhuma cópia de artefato é necessária.

### `separate_uc`

HML e PRD têm metastores/contas isoladas (comum entre assinaturas Azure
distintas). Antes de `promote_model.py` poder mover o alias `champion` no
registry de PRD, é necessário:

1. Baixar o artefato imutável validado em HML (o wheel + o modelo MLflow);
2. Calcular o SHA-256 (`scripts/validate_model_artifact.py` já faz isso);
3. Transferir o artefato por um mecanismo aprovado pela organização (ex.:
   Storage Account compartilhada com acesso via Managed Identity, nunca uma
   cópia manual);
4. Registrar o MESMO artefato no registry de destino (nova versão no
   metastore de PRD, mas apontando para o binário idêntico);
5. Validar que o SHA-256 no destino bate com o de origem;
6. Preservar nos metadados da nova versão: `run_id` de origem, `commit_sha`,
   `dataset_version`, assinatura e dependências (não recriar o
   `input_example`/`signature` -- copiar os já validados em HML);
7. Só então `promote_model.py` reatribui o alias `champion` no registry de
   PRD.

`promote_model.py` recebe `--registry-mode separate_uc` para deixar
explícito no log/relatório qual caminho foi seguido; a transferência do
artefato (passo 3) é responsabilidade de um step específico do projeto
(não genérico o suficiente para viver no repositório central, pois depende
de como cada organização conecta suas contas Azure/Databricks).

**Nunca assuma que aliases ou versões atravessam automaticamente workspaces
ou contas isoladas** -- isso só é verdade em `shared_uc`.
