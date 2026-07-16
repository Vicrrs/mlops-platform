#!/usr/bin/env python3
"""Valida o relatório de qualidade de dados produzido pelo job de qualidade do projeto.

Uso:
    python scripts/validate_data_quality.py --report-path artifacts/data_quality/data_quality_report.json
"""

from __future__ import annotations

import argparse
import sys

from _lib import fail, get_logger, read_json, succeed

logger = get_logger(__name__)

BLOCKING_SEVERITIES = {"error", "critical"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args(argv)

    try:
        report = read_json(args.report_path)
    except FileNotFoundError:
        return fail(logger, "Relatório de qualidade de dados não encontrado", report_path=args.report_path)

    blocking_failures = [
        rule
        for rule in report.get("rules", [])
        if rule.get("status") == "fail" and rule.get("severity") in BLOCKING_SEVERITIES
    ]

    if blocking_failures or not report.get("passed", False):
        names = [r["name"] for r in blocking_failures]
        return fail(
            logger,
            "Quality gate de dados reprovado",
            row_count=report.get("row_count"),
            failed_rules=names,
        )

    return succeed(logger, "Quality gate de dados aprovado", row_count=report.get("row_count"))


if __name__ == "__main__":
    sys.exit(main())
