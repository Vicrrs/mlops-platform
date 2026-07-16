"""EDA (Análise Exploratória de Dados) automatizada, baseada em agregações Spark.

Nunca converte o dataset inteiro para pandas nem executa ``collect()``
indiscriminado -- estatísticas numéricas usam agregações/``approxQuantile``,
frequências categóricas usam ``groupBy().limit(N)`` com limite configurável de
cardinalidade, e outliers são contados via agregação (não coletados linha a linha).
"""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_model.exceptions import SchemaValidationError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CATEGORIES = 50
DEFAULT_QUANTILES = (0.25, 0.5, 0.75)
DEFAULT_QUANTILE_RELATIVE_ERROR = 0.01


def profile_schema(df: DataFrame) -> dict:
    return {
        "fields": [
            {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable}
            for f in df.schema.fields
        ]
    }


def profile_nulls(df: DataFrame, columns: list[str]) -> dict:
    total = df.count()
    if total == 0 or not columns:
        return {"row_count": total, "columns": {}}
    agg_row = df.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns]).first()
    result = {}
    for column in columns:
        null_count = agg_row[column] or 0
        result[column] = {
            "null_count": null_count,
            "null_percentage": round(null_count / total, 4),
        }
    return {"row_count": total, "columns": result}


def profile_duplicates(df: DataFrame, keys: list[str]) -> dict:
    total = df.count()
    distinct = df.select(*keys).distinct().count() if keys else df.distinct().count()
    duplicate_count = total - distinct
    return {
        "row_count": total,
        "distinct_count": distinct,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(duplicate_count / total, 4) if total else 0.0,
        "keys": keys,
    }


def profile_numeric_statistics(
    df: DataFrame,
    numeric_columns: list[str],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict:
    if not numeric_columns:
        return {}
    agg_exprs = []
    for column in numeric_columns:
        agg_exprs += [
            F.min(column).alias(f"{column}__min"),
            F.max(column).alias(f"{column}__max"),
            F.mean(column).alias(f"{column}__mean"),
            F.stddev(column).alias(f"{column}__stddev"),
        ]
    agg_row = df.select(*agg_exprs).first()

    result: dict[str, dict] = {}
    for column in numeric_columns:
        quantile_values = df.approxQuantile(column, list(quantiles), DEFAULT_QUANTILE_RELATIVE_ERROR)
        result[column] = {
            "min": agg_row[f"{column}__min"],
            "max": agg_row[f"{column}__max"],
            "mean": agg_row[f"{column}__mean"],
            "median_approx": quantile_values[len(quantile_values) // 2] if quantile_values else None,
            "stddev": agg_row[f"{column}__stddev"],
            "quantiles": dict(zip((str(q) for q in quantiles), quantile_values, strict=True)),
        }
    return result


def profile_categorical_statistics(
    df: DataFrame,
    categorical_columns: list[str],
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> dict:
    result: dict[str, dict] = {}
    for column in categorical_columns:
        distinct_count = df.select(column).distinct().count()
        top_rows = df.groupBy(column).count().orderBy(F.desc("count")).limit(max_categories).collect()
        result[column] = {
            "distinct_count": distinct_count,
            "high_cardinality": distinct_count > max_categories,
            "top_categories": [{"value": row[column], "count": row["count"]} for row in top_rows],
        }
    return result


def profile_target_distribution(df: DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return {}
    total = df.count()
    counts = {row[target_column]: row["count"] for row in df.groupBy(target_column).count().collect()}
    positive = counts.get(1, 0)
    negative = counts.get(0, 0)
    return {
        "total": total,
        "class_counts": {str(k): v for k, v in counts.items()},
        "positive_ratio": round(positive / total, 4) if total else 0.0,
        "negative_ratio": round(negative / total, 4) if total else 0.0,
        "imbalance_ratio": round(negative / positive, 4) if positive else None,
    }


def profile_correlations(df: DataFrame, numeric_columns: list[str]) -> dict:
    correlations: dict[str, float] = {}
    for i, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[i + 1 :]:
            value = df.stat.corr(col_a, col_b)
            correlations[f"{col_a}__{col_b}"] = round(value, 4) if value is not None else None
    return correlations


def profile_outliers_basic(df: DataFrame, numeric_columns: list[str]) -> dict:
    result: dict[str, dict] = {}
    for column in numeric_columns:
        q1, q3 = df.approxQuantile(column, [0.25, 0.75], DEFAULT_QUANTILE_RELATIVE_ERROR)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = df.where((F.col(column) < lower) | (F.col(column) > upper)).count()
        result[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count,
        }
    return result


def profile_temporal(df: DataFrame, date_columns: list[str]) -> dict:
    result: dict[str, dict] = {}
    for column in date_columns:
        if column not in df.columns:
            continue
        row = df.select(F.min(column).alias("min_date"), F.max(column).alias("max_date")).first()
        monthly = (
            df.select(F.date_format(column, "yyyy-MM").alias("month"))
            .where(F.col("month").isNotNull())
            .groupBy("month")
            .count()
            .orderBy("month")
            .collect()
        )
        result[column] = {
            "min_date": str(row["min_date"]) if row["min_date"] else None,
            "max_date": str(row["max_date"]) if row["max_date"] else None,
            "monthly_distribution": [{"month": r["month"], "count": r["count"]} for r in monthly],
        }
    return result


def render_markdown_profile(summary: dict) -> str:
    lines = [
        f"# Data Profile — {summary.get('dataset_name', 'dataset')}",
        "",
        f"- Linhas: **{summary['row_count']}**",
        f"- Colunas: **{summary['column_count']}**",
        f"- Estratégia de amostragem: **{summary.get('sampling_strategy', 'full_dataset')}**",
        "",
        "## Nulos",
        "",
        "| Coluna | % Nulos |",
        "|---|---|",
    ]
    for column, stats in summary.get("null_report", {}).get("columns", {}).items():
        lines.append(f"| {column} | {stats['null_percentage']:.2%} |")

    lines += [
        "",
        "## Estatísticas numéricas",
        "",
        "| Coluna | Min | Max | Média | Mediana~ | Desvio |",
        "|---|---|---|---|---|---|",
    ]
    for column, stats in summary.get("numeric_statistics", {}).items():
        lines.append(
            f"| {column} | {stats['min']} | {stats['max']} | {stats['mean']:.2f} | "
            f"{stats['median_approx']:.2f} | {stats['stddev']:.2f} |"
        )

    lines += ["", "## Distribuição da variável alvo", ""]
    target = summary.get("target_distribution", {})
    if target:
        lines.append(f"- Positivos: {target['positive_ratio']:.2%}")
        lines.append(f"- Negativos: {target['negative_ratio']:.2%}")

    return "\n".join(lines)


def run_eda(
    df: DataFrame,
    required_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    target_column: str,
    date_columns: tuple[str, ...] = (),
    dataset_name: str = "dataset",
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> dict:
    """Executa a EDA completa e retorna um dicionário-resumo com todas as seções.

    Raises:
        SchemaValidationError: quando alguma coluna obrigatória está ausente.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"EDA: colunas obrigatórias ausentes: {missing}")

    row_count = df.count()
    schema = profile_schema(df)
    null_report = profile_nulls(df, list(df.columns))
    duplicate_report = profile_duplicates(df, [c for c in ("customer_id",) if c in df.columns])
    numeric_stats = profile_numeric_statistics(df, list(numeric_columns))
    categorical_stats = profile_categorical_statistics(df, list(categorical_columns), max_categories)
    target_distribution = profile_target_distribution(df, target_column)
    correlations = profile_correlations(df, list(numeric_columns))
    outliers = profile_outliers_basic(df, list(numeric_columns))
    temporal = profile_temporal(df, list(date_columns))

    summary = {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "column_count": len(df.columns),
        "sampling_strategy": "full_dataset",
        "schema": schema,
        "null_report": null_report,
        "duplicate_report": duplicate_report,
        "numeric_statistics": numeric_stats,
        "categorical_statistics": categorical_stats,
        "target_distribution": target_distribution,
        "correlations": correlations,
        "outliers": outliers,
        "temporal": temporal,
    }
    logger.info(
        "EDA concluída",
        extra={"extra_fields": {"dataset_name": dataset_name, "row_count": row_count}},
    )
    return summary


def write_eda_artifacts(summary: dict, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (output_path / "schema.json").write_text(
        json.dumps(summary["schema"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "null_report.json").write_text(
        json.dumps(summary["null_report"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "duplicate_report.json").write_text(
        json.dumps(summary["duplicate_report"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "target_distribution.json").write_text(
        json.dumps(summary["target_distribution"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "numeric_statistics.json").write_text(
        json.dumps(summary["numeric_statistics"], indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (output_path / "categorical_statistics.json").write_text(
        json.dumps(summary["categorical_statistics"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "data_profile.md").write_text(render_markdown_profile(summary), encoding="utf-8")

    logger.info("Artefatos de EDA gravados", extra={"extra_fields": {"path": str(output_path)}})
    return output_path
