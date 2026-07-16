# Processo de release

## Os 22 stages de `ml-project-pipeline.yml`

| # | Stage | Quando roda | Template/script |
|---|---|---|---|
| 1 | ValidateProject | Sempre | `templates/ci.yml` → `steps/validate-structure.yml` |
| 2 | Lint | Sempre | `templates/ci.yml` → `steps/run-lint.yml` |
| 3 | UnitTests | Sempre | `templates/ci.yml` → `steps/run-unit-tests.yml` |
| 4 | SparkTests | Se `runSparkTests` | `templates/spark-tests.yml` |
| 5 | SecurityScan | Se `runSecurityScan` | `templates/ci.yml` (pip-audit) |
| 6 | BuildArtifact | Sempre | `templates/build.yml` |
| 7 | ValidateBundle | Sempre | `steps/validate-bundle.yml` |
| 8 | DeployHML | Não em PR | `templates/deploy-hml.yml` |
| 9 | DataQualityHML | Se `runDataQuality`, não em PR | `templates/data-quality.yml` |
| 10 | AutomatedEDAHML | Se `runEDA`, não em PR | `templates/data-quality.yml` |
| 11 | TrainCandidateHML | Não em PR | `templates/train-candidate.yml` |
| 12 | EvaluateCandidateHML | Não em PR | `scripts/validate_model_metrics.py` |
| 13 | RegisterChallenger | — | `templates/register-challenger.yml` |
| 14 | CompareChampionChallenger | — | `templates/compare-models.yml` |
| 15 | IntegrationTestsHML | Se `runIntegrationTests` | `pytest tests/integration` |
| 16 | ModelQualityGate | — | `templates/model-quality-gate.yml` |
| 17 | ApprovalPRD | Só na `main` | Azure DevOps Environment (`databricks-prd`) + `templates/approval-prd.yml` |
| 18 | VerifyPromotionArtifact | — | `scripts/validate_model_artifact.py` |
| 19 | DeployPRD | — | `templates/deploy-prd.yml` |
| 20 | PromoteChampion | — | `templates/promote-champion.yml` → `scripts/promote_model.py` |
| 21 | SmokeTestsPRD | Se `runSmokeTests` | `templates/smoke-tests.yml` |
| 22 | StartMonitoring | — | `databricks bundle run monitoring_job` |

## Comportamento por evento

- **Pull Request** (`Build.Reason = PullRequest`): executa stages 1-5 (e 15,
  se o PR já tiver progredido até lá em um pipeline manual) -- valida
  estrutura, lint, testes unitários e Spark, security scan. **Nunca** chega a
  `DeployHML` nem `DeployPRD` (`condition: eq(variables.isPullRequest, false)`
  bloqueia explicitamente).
- **Merge na `main`**: além do CI, executa `DeployHML` e toda a cadeia de
  treino/comparação/quality gate em HML. Só chega em `ApprovalPRD` (e,
  portanto, em PRD) quando a branch de origem é `main`
  (`condition: eq(variables.isMainBranch, true)`).
- **PRD**: só ocorre após um aprovador humano aceitar o check de aprovação
  no Environment `databricks-prd`. Antes disso, `ApprovalPRD` fica pendente
  indefinidamente (não há timeout automático nesta plataforma -- configure um
  se sua política de segurança exigir).

## O artefato que chega em PRD é sempre o testado em HML

`VerifyPromotionArtifact` (stage 18) roda `scripts/validate_model_artifact.py`
comparando:

- `git_commit` do `run_metadata.json` vs. `$(Build.SourceVersion)` do build
  atual;
- `mlflow_run_id` do `run_metadata.json` vs. o `run_id` real gravado durante
  o treino (nunca um valor recalculado);
- `model_version` do `run_metadata.json` vs. a versão registrada.

Qualquer divergência falha a pipeline antes do deploy em PRD. Isso é o que
garante "a mesma versão testada em HML chega ao PRD" (critério de aceite #17
do escopo da plataforma).

## `retrainInProduction`

Valor padrão: `false`. A pipeline central **nunca** treina em PRD por
padrão -- `DeployPRD` (stage 19) apenas faz `databricks bundle deploy` do
mesmo artefato. A única forma de retreinar em PRD é setar
`retrainInProduction: true` explicitamente nos parâmetros do
`azure-pipelines.yml` do projeto, o que:

- gera um `##vso[task.logissue type=warning]` visível no log da pipeline;
- documenta a exceção no próprio YAML do projeto (visível em toda revisão
  de PR daquele repositório);
- ainda assim não pula nenhum quality gate -- o modelo retreinado em PRD
  passaria pelos MESMOS stages de avaliação antes de qualquer promoção.

Toda exceção ao comportamento padrão deve ficar documentada no
`azure-pipelines.yml` do projeto (comentário explicando o motivo) e
aprovada pelo time de MLOps na revisão (ver `docs/security.md`).

## Rastreabilidade

Cada `promotion_record.json` (gravado por `scripts/promote_model.py`) e
`rollback_report.json` (gravado por `scripts/rollback.py`) contém: projeto,
ambiente, modelo, versões envolvidas, `run_id`, `git_commit`, `build_id`,
aprovador/autor, motivo (rollback) e timestamp -- o vínculo completo entre
código, dados (via `dataset_version`) e modelo exigido pela governança da
plataforma.
