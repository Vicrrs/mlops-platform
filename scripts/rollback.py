#!/usr/bin/env python3
"""Executa rollback do Champion para uma versão anterior conhecida como boa.

Sempre roda em modo dry-run (nenhuma alteração de alias) a menos que
``--confirm`` seja passado explicitamente. Nunca apaga versões -- apenas
reatribui os aliases ``champion``/``previous_champion``. Executa um smoke
test (via ``--smoke-test-command``) antes de trocar o alias, quando informado.

Uso:
    python scripts/rollback.py \
        --project ml-fraude \
        --environment prd \
        --tracking-uri databricks --registry-uri databricks-uc \
        --registered-model-name prd_catalog.fraude.modelo_fraude \
        --target-version 12 \
        --author "jane.doe@empresa.com" --reason "Regressão de recall detectada em produção" \
        --build-id 4821 \
        --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import UTC, datetime

from _lib import fail, get_logger, get_mlflow_client, succeed, write_json
from mlflow.exceptions import MlflowException

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--registry-uri", default=None)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--champion-alias", default="champion")
    parser.add_argument("--previous-champion-alias", default="previous_champion")
    parser.add_argument("--author", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument(
        "--smoke-test-command",
        default=None,
        help="Comando que executa o smoke test do projeto contra a versão alvo antes de trocar o alias.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicita a intenção de dry-run (comportamento padrão; ver --confirm).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Executa o rollback de fato. Sem esta flag, nenhuma alteração é aplicada.",
    )
    parser.add_argument("--output-path", default="artifacts/rollback/rollback_report.json")
    args = parser.parse_args(argv)

    execute = args.confirm and not args.dry_run

    client = get_mlflow_client(args.tracking_uri, args.registry_uri)
    model_name = args.registered_model_name

    try:
        target_version_info = client.get_model_version(model_name, args.target_version)
    except MlflowException as exc:
        return fail(logger, "Versão alvo do rollback não existe no registry", target_version=args.target_version, error=str(exc))

    try:
        current_champion = client.get_model_version_by_alias(model_name, args.champion_alias)
        current_champion_version = current_champion.version
    except MlflowException:
        current_champion_version = None

    report = {
        "project": args.project,
        "environment": args.environment,
        "registered_model_name": model_name,
        "from_version": current_champion_version,
        "to_version": args.target_version,
        "to_run_id": target_version_info.run_id,
        "author": args.author,
        "reason": args.reason,
        "build_id": args.build_id,
        "dry_run": not execute,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if not execute:
        write_json(args.output_path, report)
        return succeed(logger, "[dry-run] Rollback simulado -- use --confirm para aplicar", **report)

    if args.smoke_test_command:
        result = subprocess.run(shlex.split(args.smoke_test_command), check=False)
        if result.returncode != 0:
            report["smoke_test_passed"] = False
            write_json(args.output_path, report)
            return fail(logger, "Smoke test falhou -- rollback abortado antes de trocar o alias")
        report["smoke_test_passed"] = True

    if current_champion_version is not None:
        client.set_registered_model_alias(model_name, args.previous_champion_alias, current_champion_version)

    client.set_registered_model_alias(model_name, args.champion_alias, args.target_version)

    write_json(args.output_path, report)
    return succeed(
        logger,
        "Rollback aplicado com sucesso",
        registered_model_name=model_name,
        from_version=current_champion_version,
        to_version=args.target_version,
        author=args.author,
    )


if __name__ == "__main__":
    sys.exit(main())
