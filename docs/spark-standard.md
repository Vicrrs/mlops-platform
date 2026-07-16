# Padrão PySpark / Spark SQL

## SparkSession

Use sempre `<pacote>.spark.get_spark_session(app_name, environment,
shuffle_partitions)`. Ela:

- reutiliza a sessão do cluster quando roda no Databricks
  (`DATABRICKS_RUNTIME_VERSION` presente);
- cria uma sessão `local[*]` com Delta Lake habilitado (via
  `delta.configure_spark_with_delta_pip`) fora do Databricks -- necessário
  para dev e para `tests/spark`/`tests/integration` rodarem sem cluster;
- nunca cria uma segunda sessão enquanto a primeira estiver ativa (singleton
  por processo).

Não instancie `SparkSession.builder` diretamente em código de produção.

## Leitura (`io/readers.py`)

- Schema explícito é obrigatório fora de `dev`
  (`allow_schema_inference=True` só é aceito em desenvolvimento). Isso evita
  que uma mudança silenciosa de schema na fonte vire um bug sutil em produção.
- Toda função de leitura loga origem, formato e `row_count`.
- JDBC nunca recebe usuário/senha literais -- `properties` vem de secrets
  resolvidos pelo chamador (Databricks Secret Scope / variável de ambiente).
- Leitura incremental usa Change Data Feed (`readChangeFeed`) quando a
  tabela de origem tiver `delta.enableChangeDataFeed = true`.

## Escrita (`io/writers.py`)

- `overwrite` em ambiente `prd` é bloqueado por padrão
  (`DataWriteError`) a menos que `allow_prd_overwrite=True` seja passado
  explicitamente pelo chamador -- previne perda de dados por engano.
- Toda escrita valida DataFrame vazio antes de gravar.
- `dry_run=True` loga o que seria escrito (destino, modo, `row_count`) sem
  gravar -- use isso para validar pipelines antes de rodar de verdade.
- `merge_delta` faz upsert real via `DeltaTable.merge` quando a tabela já
  existe; cria a tabela (modo `errorifexists`) na primeira execução.

## SQL

Consultas Spark SQL devem:

- registrar views temporárias (`createOrReplaceTempView`) quando precisarem
  compor múltiplas fontes;
- nunca concatenar entrada não confiável (nomes de coluna/tabela vindos de
  configuração são OK; valores vindos de usuário final, nunca) diretamente em
  uma string SQL -- prefira a API de DataFrame (`.filter()`, `.where()`) ou
  parâmetros ligados, evitando injeção de SQL;
- validar o schema do resultado quando ele alimenta uma etapa downstream
  crítica (usar `io.readers.validate_schema`).

## EDA e agregações -- nunca `collect()` indiscriminado

`data/eda.py` só materializa localmente:

- agregações escalares (`min`/`max`/`mean`/`stddev` via `df.select(...).first()`);
- quantis aproximados (`approxQuantile`, nunca ordenação exata de dataset
  completo);
- top-N de colunas categóricas, com `max_categories` configurável
  (`groupBy().orderBy().limit(N).collect()` -- limitado, não a tabela toda);
- contagens de outliers via filtro + `count()`, nunca linha a linha.

Nunca chame `.toPandas()` sobre o dataset completo. Quando uma amostra for
necessária (ex.: `input_example` do MLflow), documente o tamanho e a
estratégia de amostragem no código e no `eda_summary.json`.

## Machine Learning (Spark ML Pipeline)

- Pré-processamento (`StringIndexer` → `OneHotEncoder` → `VectorAssembler` →
  `StandardScaler`) e o estimador (`LogisticRegression`/
  `RandomForestClassifier` por padrão, mas a abstração em
  `features/pipeline.py` aceita outros algoritmos) vivem no MESMO
  `pyspark.ml.Pipeline`.
- O split treino/validação/teste (`features/transformations.split_train_val_test`)
  acontece **antes** de qualquer `fit()` -- os estágios de feature engineering
  só veem o conjunto de treino durante o ajuste, prevenindo vazamento de dados.
- `handle_missing_values` (imputação determinística) é chamada tanto no
  treino quanto na inferência, garantindo que a transformação nunca diverge
  entre os dois fluxos.
