"""Orquestração das regras de qualidade de dados e geração dos artefatos.

Produz, sob ``artifacts/data_quality/``:
``data_quality_report.json``, ``failed_rules.json``, ``schema_comparison.json``
e ``data_drift_report.json``, conforme exigido pela plataforma.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from churn_model.config import DataSettings
from churn_model.data import validation as v
from churn_model.data.schemas import ALLOWED_CONTRACT_TYPES, ALLOWED_INTERNET_SERVICES, TARGET_COLUMN
from churn_model.exceptions import DataQualityError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    rules: list[v.RuleResult]
    passed: bool
    row_count: int
    timestamp: str

    def failed_rules(self) -> list[v.RuleResult]:
        return [r for r in self.rules if not r.passed]

    def blocking_failures(self) -> list[v.RuleResult]:
        return [r for r in self.rules if r.is_blocking_failure]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "row_count": self.row_count,
            "timestamp": self.timestamp,
            "rules": [r.to_dict() for r in self.rules],
        }


def run_data_quality_checks(
    df: DataFrame,
    settings: DataSettings,
    previous_row_count: int | None = None,
    reference_schema_json: dict | None = None,
) -> DataQualityReport:
    """Executa o conjunto de regras de qualidade sobre ``df`` e agrega o resultado.

    Regras com severidade ``error`` ou ``critical`` que falharem tornam o
    relatório reprovado (``passed=False``); o chamador decide, via
    ``DataQualityError``, se isso interrompe a pipeline.
    """
    rules: list[v.RuleResult] = []

    rules.append(v.check_required_columns(df, settings.required_columns))
    if rules[-1].passed:
        rules.append(v.check_extra_columns(df, settings.required_columns))
        rules.append(v.check_row_count(df, settings.quality.min_rows, settings.quality.max_rows))
        rules.append(
            v.check_volume_change(
                df.count(), previous_row_count, settings.quality.max_volume_change_percentage
            )
        )
        rules.append(v.check_duplicates(df, [settings.id_column], settings.quality.max_duplicate_percentage))

        for column in settings.required_columns:
            rules.append(v.check_null_percentage(df, column, settings.quality.max_null_percentage))

        if "contract_type" in df.columns:
            rules.append(v.check_allowed_domain(df, "contract_type", ALLOWED_CONTRACT_TYPES))
        if "internet_service" in df.columns:
            rules.append(v.check_allowed_domain(df, "internet_service", ALLOWED_INTERNET_SERVICES))
        if "tenure_months" in df.columns:
            rules.append(v.check_value_range(df, "tenure_months", 0, 600))
        if "monthly_charges" in df.columns:
            rules.append(v.check_value_range(df, "monthly_charges", 0, 100_000))
        if TARGET_COLUMN in df.columns:
            rules.append(v.check_target_distribution(df, TARGET_COLUMN))

        if reference_schema_json is not None:
            actual_schema_json = json.loads(df.schema.json())
            rules.append(v.check_schema_drift(actual_schema_json, reference_schema_json))

    blocking_failures = [r for r in rules if r.is_blocking_failure]
    report = DataQualityReport(
        rules=rules,
        passed=not blocking_failures,
        row_count=df.count(),
        timestamp=datetime.now(UTC).isoformat(),
    )
    logger.info(
        "Verificação de qualidade de dados concluída",
        extra={
            "extra_fields": {
                "passed": report.passed,
                "total_rules": len(rules),
                "failed_rules": len(report.failed_rules()),
            }
        },
    )
    return report


def compute_basic_data_drift(
    current_df: DataFrame, reference_df: DataFrame, numeric_columns: tuple[str, ...]
) -> dict:
    """Compara média e desvio padrão das colunas numéricas entre dataset atual e referência."""
    drift: dict[str, dict] = {}
    if not numeric_columns:
        return drift
    current_stats = current_df.select(
        *[F.mean(c).alias(f"{c}_mean") for c in numeric_columns],
        *[F.stddev(c).alias(f"{c}_stddev") for c in numeric_columns],
    ).first()
    reference_stats = reference_df.select(
        *[F.mean(c).alias(f"{c}_mean") for c in numeric_columns],
        *[F.stddev(c).alias(f"{c}_stddev") for c in numeric_columns],
    ).first()
    for column in numeric_columns:
        current_mean = current_stats[f"{column}_mean"]
        reference_mean = reference_stats[f"{column}_mean"]
        relative_shift = None
        if reference_mean not in (None, 0):
            relative_shift = round(abs((current_mean or 0) - reference_mean) / abs(reference_mean), 4)
        drift[column] = {
            "current_mean": current_mean,
            "reference_mean": reference_mean,
            "current_stddev": current_stats[f"{column}_stddev"],
            "reference_stddev": reference_stats[f"{column}_stddev"],
            "relative_mean_shift": relative_shift,
        }
    return drift


def write_data_quality_artifacts(
    report: DataQualityReport,
    output_dir: str | Path,
    schema_comparison: dict | None = None,
    data_drift: dict | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "data_quality_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "failed_rules.json").write_text(
        json.dumps([r.to_dict() for r in report.failed_rules()], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "schema_comparison.json").write_text(
        json.dumps(schema_comparison or {}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "data_drift_report.json").write_text(
        json.dumps(data_drift or {}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Artefatos de qualidade de dados gravados", extra={"extra_fields": {"path": str(output_path)}}
    )
    return output_path


def enforce_quality_gate(report: DataQualityReport) -> None:
    """Interrompe a pipeline quando houver falha bloqueante (severidade error/critical)."""
    blocking = report.blocking_failures()
    if blocking:
        names = [r.name for r in blocking]
        raise DataQualityError(
            f"Quality gate de dados reprovado: {len(blocking)} regra(s) bloqueante(s) falharam: {names}",
            failed_rules=names,
        )
