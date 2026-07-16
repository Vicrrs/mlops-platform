#!/usr/bin/env python3
"""Promove a versão Challenger validada para Champion (somente após aprovação manual).

Não treina um novo modelo -- promove exatamente a versão testada em HML. Sempre
executa em modo dry-run a menos que ``--confirm`` seja passado explicitamente.

Modos de registry (ver docs/champion-challenger.md):
  - shared_uc:   HML e PRD compartilham o mesmo Model Registry/metastore. A
                 mesma versão registrada recebe o alias 'champion' diretamente.
  - separate_uc: HML e PRD têm registries isolados. É necessário que a versão
                 já tenha sido replicada para o registry de destino por um
                 mecanismo aprovado (ex.: exportação/importação de modelo) e
                 que o SHA-256 do artefato tenha sido validado antes desta
                 chamada (ver validate_model_artifact.py). Este script então
                 apenas move os aliases no registry de DESTINO.

Uso (dry-run por padrão):
    python scripts/promote_model.py \
        --tracking-uri file:./mlruns --registry-uri file:./mlruns \
        --registered-model-name dev_catalog.churn.churn_model \
        --challenger-version 3 \
        --promotion-recommendation-path artifacts/model_comparison/promotion_recommendation.json \
        --run-metadata-path artifacts/training/run_metadata.json \
        --approval-path artifacts/approval.json \
        --approver "jane.doe@empresa.com" --build-id 12345 \
        --confirm
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _lib import fail, get_logger, get_mlflow_client, read_json, succeed, write_json

logger = get_logger(__name__)


def run_verification(args: argparse.Namespace) -> int:
    """Reaproveita verify_promotion_metadata.py (mesmo diretório) antes de promover."""
    script_path = Path(__file__).parent / "verify_promotion_metadata.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--promotion-recommendation-path",
            args.promotion_recommendation_path,
            "--run-metadata-path",
            args.run_metadata_path,
            "--approval-path",
            args.approval_path,
        ],
        check=False,
    )
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--registry-uri", default=None)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--challenger-version", required=True)
    parser.add_argument("--champion-alias", default="champion")
    parser.add_argument("--challenger-alias", default="challenger")
    parser.add_argument("--previous-champion-alias", default="previous_champion")
    parser.add_argument("--promotion-recommendation-path", required=True)
    parser.add_argument("--run-metadata-path", required=True)
    parser.add_argument("--approval-path", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--registry-mode", choices=["shared_uc", "separate_uc"], default="shared_uc")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Executa a promoção de fato. Sem esta flag, roda em modo dry-run (nenhuma alteração é feita).",
    )
    parser.add_argument("--output-path", default="artifacts/promotion/promotion_record.json")
    args = parser.parse_args(argv)

    verification_exit_code = run_verification(args)
    if verification_exit_code != 0:
        return fail(logger, "Verificação de metadados de promoção falhou -- promoção abortada")

    recommendation = read_json(args.promotion_recommendation_path)
    run_metadata = read_json(args.run_metadata_path)

    client = get_mlflow_client(args.tracking_uri, args.registry_uri)
    model_name = args.registered_model_name

    try:
        current_champion = client.get_model_version_by_alias(model_name, args.champion_alias)
    except Exception:  # noqa: BLE001
        current_champion = None

    plan = {
        "registered_model_name": model_name,
        "registry_mode": args.registry_mode,
        "challenger_version_to_promote": args.challenger_version,
        "current_champion_version": current_champion.version if current_champion else None,
        "will_set_previous_champion_to": current_champion.version if current_champion else None,
        "will_set_champion_to": args.challenger_version,
    }

    if not args.confirm:
        logger.info(f"[dry-run] Plano de promoção (nenhuma alteração aplicada): {plan}")
        write_json(args.output_path, {**plan, "dry_run": True})
        return succeed(logger, "Dry-run de promoção concluído -- use --confirm para aplicar", **plan)

    if current_champion is not None:
        client.set_registered_model_alias(model_name, args.previous_champion_alias, current_champion.version)

    client.set_registered_model_alias(model_name, args.champion_alias, args.challenger_version)

    promotion_record = {
        **plan,
        "dry_run": False,
        "approver": args.approver,
        "build_id": args.build_id,
        "git_commit": run_metadata.get("git_commit"),
        "mlflow_run_id": run_metadata.get("mlflow_run_id"),
        "promoted_at": datetime.now(UTC).isoformat(),
        "primary_metric": recommendation.get("primary_metric"),
        "challenger_metric": recommendation.get("challenger_metric"),
        "champion_metric_before": recommendation.get("champion_metric"),
    }
    write_json(args.output_path, promotion_record)

    return succeed(
        logger,
        "Promoção concluída: Challenger agora é Champion",
        registered_model_name=model_name,
        new_champion_version=args.challenger_version,
        previous_champion_version=promotion_record["current_champion_version"],
        approver=args.approver,
    )


if __name__ == "__main__":
    sys.exit(main())
