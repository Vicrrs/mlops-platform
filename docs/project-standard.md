# Padrão de estrutura de projeto ML

Todo repositório consumidor (ex.: `ml-fraude`, `ml-risco-credito`, `ml-churn`,
`ml-previsao-demanda`) deve seguir exatamente a estrutura abaixo. Ela é
validada automaticamente pelo stage `ValidateProject` via
`scripts/validate_project_structure.py` -- a pipeline falha com uma lista
explícita de arquivos/pastas ausentes quando a estrutura diverge.

```
ml-nome-projeto/
├── azure-pipelines.yml        # pequeno, apenas `extends` o template central
├── databricks.yml
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── conf/{base,dev,hml,prd}.yml
├── resources/*.job.yml        # 7 jobs: data-quality, eda, training,
│                               #   model-validation, batch-inference,
│                               #   monitoring, smoke-test
├── src/<nome_pacote>/
│   ├── config.py, exceptions.py, logging_config.py, spark.py
│   ├── io/{readers,writers}.py
│   ├── data/{schemas,validation,quality,eda}.py
│   ├── features/{transformations,pipeline}.py
│   ├── models/{train,evaluate,registry,champion_challenger,inference}.py
│   ├── monitoring/{data_monitoring,prediction_monitoring,model_monitoring}.py
│   └── cli/                   # um módulo por job Databricks
├── notebooks/                 # finos: só chamam o pacote (ver abaixo)
├── data/{sample,schemas}/
├── tests/{unit,spark,integration,smoke}/
└── artifacts/.gitkeep
```

Use `project-template/` (neste repositório) como ponto de partida real e
executável -- não um esqueleto: ele contém um caso funcional completo
(classificação binária de churn) que você adapta ao seu domínio.

## Regras não negociáveis

- **Notebooks são finos.** Só podem: importar funções do pacote, ler
  widgets/parâmetros, chamar código testável, exibir resultados
  (`display()`/gráficos). Nenhuma lógica de negócio, transformação de dados
  ou treino pode existir *apenas* em um notebook -- se existe, tem que estar
  em `src/` com teste correspondente.
- **Mesma transformação em treino e inferência.** O `PipelineModel` ajustado
  no treino (`models/train.py`) é o mesmo objeto logado no MLflow e carregado
  na inferência (`models/inference.py`). Não implemente uma segunda versão
  "simplificada" da transformação para servir predições.
- **Sem segredo no código.** Toda credencial vem de Variable Group / Secret
  Scope / variável de ambiente (ver `docs/security.md` e
  `docs/required-variables.md`), nunca de um literal em `.py` ou `.yml`.
- **`conf/base.yml` é a fonte da verdade** para hiperparâmetros e limites;
  `conf/{dev,hml,prd}.yml` só sobrescrevem o que muda por ambiente.

## Testes obrigatórios

| Camada | O que cobre | Depende de |
|---|---|---|
| `tests/unit` | Lógica pura e com SparkSession local (config, regras de qualidade, features, treino, avaliação, registry, Champion/Challenger, inferência, monitoramento) | Nada externo |
| `tests/spark` | Mecânica Spark (sessão, leitura/escrita Delta, transformações, EDA, Pipeline) | SparkSession local |
| `tests/integration` | Fluxos completos (treino→registry, Champion/Challenger com champion real, integridade de artefato, configuração do bundle) | SparkSession local + MLflow file store |
| `tests/smoke` | Contrato do smoke test de produção via subprocess (CLI real) | SparkSession local + MLflow file store |

Nenhum teste deve usar métricas fixas nem mockar o MLflow/Spark de forma que
o teste sempre passe independentemente do código -- os testes desta
plataforma (ver `project-template/tests/`) treinam modelos reais pequenos e
verificam valores reais.
