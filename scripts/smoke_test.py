#!/usr/bin/env python3
"""Dispara o smoke test do projeto (executado no Databricks) e valida o relatório resultante.

A lógica de inferência (Spark) vive no projeto (``churn_model.cli.run_smoke_test``,
acionado via ``resources/smoke-test.job.yml``). Este script central é
agnóstico de projeto: executa o comando informado em ``--command`` (que
dispara o job no projeto/Databricks) e valida o ``smoke_test_report.json``
resultante, garantindo o mesmo contrato para qualquer projeto.

Uso:
    python scripts/smoke_test.py \
        --command ".venv/bin/python -m churn_model.cli.run_smoke_test --config conf/dev.yml --model-alias champion" \
        --report-path artifacts/smoke_test/smoke_test_report.json
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from _lib import fail, get_logger, read_json, succeed

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        help="Comando que executa o smoke test do projeto (tipicamente dispara o job Databricks).",
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--cwd", default=None)
    args = parser.parse_args(argv)

    result = subprocess.run(shlex.split(args.command), cwd=args.cwd, check=False)
    if result.returncode != 0:
        return fail(logger, "Comando de smoke test retornou código de erro", exit_code=result.returncode)

    try:
        report = read_json(args.report_path)
    except FileNotFoundError:
        return fail(logger, "Relatório de smoke test não encontrado", report_path=args.report_path)

    if not report.get("passed", False):
        return fail(logger, "Smoke test reprovado pelo relatório do projeto", report=report)

    return succeed(
        logger,
        "Smoke test aprovado",
        model_alias=report.get("model_alias"),
        row_count=report.get("row_count"),
    )


if __name__ == "__main__":
    sys.exit(main())
