#!/usr/bin/env python3
"""Valida que as variáveis de ambiente obrigatórias estão definidas (nunca imprime valores).

Uso:
    python scripts/validate_environment.py --required-vars DATABRICKS_HOST DATABRICKS_CLIENT_ID
    python scripts/validate_environment.py --required-vars-file required-vars.txt
"""

from __future__ import annotations

import argparse
import os
import sys

from _lib import fail, get_logger, succeed

logger = get_logger(__name__)

DEFAULT_REQUIRED_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_CLIENT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "DATABRICKS_CATALOG",
    "DATABRICKS_SCHEMA",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-vars", nargs="*", default=None)
    parser.add_argument("--required-vars-file", default=None)
    args = parser.parse_args(argv)

    required_vars = args.required_vars
    if required_vars is None and args.required_vars_file:
        with open(args.required_vars_file, encoding="utf-8") as fh:
            required_vars = [line.strip() for line in fh if line.strip()]
    if required_vars is None:
        required_vars = DEFAULT_REQUIRED_VARS

    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        return fail(logger, "Variáveis de ambiente obrigatórias ausentes", missing_variables=missing)

    return succeed(logger, "Todas as variáveis de ambiente obrigatórias estão definidas", checked=len(required_vars))


if __name__ == "__main__":
    sys.exit(main())
