"""Motor de regras de qualidade de dados.

Cada verificação retorna um :class:`RuleResult` com nome, descrição, severidade,
valor observado/esperado, status e timestamp -- o formato exigido pelos
relatórios de qualidade da plataforma. Nenhuma verificação usa ``collect()``
sobre o dataset completo; todas usam agregações Spark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


BLOCKING_SEVERITIES = {Severity.ERROR, Severity.CRITICAL}


@dataclass
class RuleResult:
    name: str
    description: str
    severity: Severity
    status: str  # "pass" | "fail"
    observed_value: object
    expected_value: object
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def is_blocking_failure(self) -> bool:
        return not self.passed and self.severity in BLOCKING_SEVERITIES


def _result(
    name: str,
    description: str,
    severity: Severity,
    passed: bool,
    observed: object,
    expected: object,
) -> RuleResult:
    return RuleResult(
        name=name,
        description=description,
        severity=severity,
        status="pass" if passed else "fail",
        observed_value=observed,
        expected_value=expected,
    )


def check_required_columns(df: DataFrame, required: tuple[str, ...]) -> RuleResult:
    missing = [c for c in required if c not in df.columns]
    return _result(
        name="required_columns",
        description="Todas as colunas obrigatórias devem estar presentes.",
        severity=Severity.CRITICAL,
        passed=not missing,
        observed=missing,
        expected=list(required),
    )


def check_extra_columns(df: DataFrame, allowed: tuple[str, ...]) -> RuleResult:
    extra = [c for c in df.columns if c not in allowed]
    return _result(
        name="extra_columns",
        description="Não deve haver colunas fora do schema esperado.",
        severity=Severity.WARNING,
        passed=not extra,
        observed=extra,
        expected=[],
    )


def check_row_count(df: DataFrame, min_rows: int, max_rows: int) -> RuleResult:
    count = df.count()
    passed = min_rows <= count <= max_rows
    return _result(
        name="row_count",
        description=f"Volume de registros deve estar entre {min_rows} e {max_rows}.",
        severity=Severity.CRITICAL,
        passed=passed,
        observed=count,
        expected=f"[{min_rows}, {max_rows}]",
    )


def check_volume_change(
    current_count: int, previous_count: int | None, max_change_percentage: float
) -> RuleResult:
    if previous_count is None or previous_count == 0:
        return _result(
            name="volume_change",
            description="Variação de volume em relação à execução anterior.",
            severity=Severity.INFO,
            passed=True,
            observed=current_count,
            expected="sem baseline anterior",
        )
    change = abs(current_count - previous_count) / previous_count
    return _result(
        name="volume_change",
        description="Variação de volume em relação à execução anterior.",
        severity=Severity.WARNING,
        passed=change <= max_change_percentage,
        observed=round(change, 4),
        expected=f"<= {max_change_percentage}",
    )


def check_null_percentage(df: DataFrame, column: str, max_null_percentage: float) -> RuleResult:
    total = df.count()
    if total == 0:
        null_pct = 0.0
    else:
        nulls = df.where(F.col(column).isNull()).count()
        null_pct = nulls / total
    return _result(
        name=f"null_percentage[{column}]",
        description=f"Percentual de nulos em '{column}' deve ser <= {max_null_percentage}.",
        severity=Severity.ERROR,
        passed=null_pct <= max_null_percentage,
        observed=round(null_pct, 4),
        expected=f"<= {max_null_percentage}",
    )


def check_duplicates(df: DataFrame, keys: list[str], max_duplicate_percentage: float) -> RuleResult:
    total = df.count()
    if total == 0:
        dup_pct = 0.0
    else:
        distinct = df.select(*keys).distinct().count()
        dup_pct = (total - distinct) / total
    return _result(
        name=f"duplicates[{','.join(keys)}]",
        description=f"Percentual de duplicidade em {keys} deve ser <= {max_duplicate_percentage}.",
        severity=Severity.ERROR,
        passed=dup_pct <= max_duplicate_percentage,
        observed=round(dup_pct, 4),
        expected=f"<= {max_duplicate_percentage}",
    )


def check_value_range(df: DataFrame, column: str, min_value: float, max_value: float) -> RuleResult:
    out_of_range = df.where((F.col(column) < min_value) | (F.col(column) > max_value)).count()
    return _result(
        name=f"value_range[{column}]",
        description=f"Valores de '{column}' devem estar entre {min_value} e {max_value}.",
        severity=Severity.ERROR,
        passed=out_of_range == 0,
        observed=out_of_range,
        expected=0,
    )


def check_allowed_domain(df: DataFrame, column: str, allowed_values: tuple[str, ...]) -> RuleResult:
    invalid = df.where(~F.col(column).isin(list(allowed_values)) & F.col(column).isNotNull()).count()
    return _result(
        name=f"allowed_domain[{column}]",
        description=f"Valores de '{column}' devem pertencer a {allowed_values}.",
        severity=Severity.ERROR,
        passed=invalid == 0,
        observed=invalid,
        expected=0,
    )


def check_cardinality(df: DataFrame, column: str, max_distinct: int) -> RuleResult:
    distinct = df.select(column).distinct().count()
    return _result(
        name=f"cardinality[{column}]",
        description=f"Cardinalidade de '{column}' deve ser <= {max_distinct}.",
        severity=Severity.WARNING,
        passed=distinct <= max_distinct,
        observed=distinct,
        expected=f"<= {max_distinct}",
    )


def check_target_distribution(
    df: DataFrame,
    target_column: str,
    min_positive_ratio: float = 0.01,
    max_positive_ratio: float = 0.99,
) -> RuleResult:
    total = df.count()
    if total == 0:
        ratio = 0.0
    else:
        positives = df.where(F.col(target_column) == 1).count()
        ratio = positives / total
    passed = min_positive_ratio <= ratio <= max_positive_ratio
    return _result(
        name=f"target_distribution[{target_column}]",
        description="Proporção da classe positiva deve estar dentro de limites plausíveis.",
        severity=Severity.WARNING,
        passed=passed,
        observed=round(ratio, 4),
        expected=f"[{min_positive_ratio}, {max_positive_ratio}]",
    )


def check_referential_integrity(
    df: DataFrame, column: str, valid_values_df: DataFrame, valid_column: str
) -> RuleResult:
    invalid = (
        df.select(column)
        .distinct()
        .join(
            valid_values_df.select(valid_column).distinct(),
            df[column] == valid_values_df[valid_column],
            "left_anti",
        )
        .count()
    )
    return _result(
        name=f"referential_integrity[{column}]",
        description=f"Valores de '{column}' devem existir na dimensão de referência.",
        severity=Severity.ERROR,
        passed=invalid == 0,
        observed=invalid,
        expected=0,
    )


def check_schema_drift(actual_schema_json: dict, expected_schema_json: dict) -> RuleResult:
    passed = actual_schema_json == expected_schema_json
    return _result(
        name="schema_drift",
        description="O schema atual deve corresponder ao schema de referência registrado.",
        severity=Severity.CRITICAL,
        passed=passed,
        observed=actual_schema_json if not passed else "unchanged",
        expected="unchanged" if passed else expected_schema_json,
    )
