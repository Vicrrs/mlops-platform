#!/usr/bin/env python3
"""Verifica a consistência entre recomendação técnica, metadados de execução e aprovação manual.

Deve ser executado (e aprovado) antes de ``promote_model.py``. Confere que:
  - ``promotion_recommendation.json`` recomenda "promote";
  - o run_id/versão/commit da recomendação batem com ``run_metadata.json``;
  - existe um registro de aprovação manual (``approval.json``) com aprovador e
    referenciando exatamente a mesma versão/run.

Uso:
    python scripts/verify_promotion_metadata.py \
        --promotion-recommendation-path artifacts/model_comparison/promotion_recommendation.json \
        --run-metadata-path artifacts/training/run_metadata.json \
        --approval-path artifacts/approval.json
"""

from __future__ import annotations

import argparse
import sys

from _lib import fail, get_logger, read_json, succeed

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-recommendation-path", required=True)
    parser.add_argument("--run-metadata-path", required=True)
    parser.add_argument("--approval-path", required=True)
    args = parser.parse_args(argv)

    try:
        recommendation = read_json(args.promotion_recommendation_path)
        run_metadata = read_json(args.run_metadata_path)
        approval = read_json(args.approval_path)
    except FileNotFoundError as exc:
        return fail(logger, "Arquivo de metadados ausente", error=str(exc))

    problems = []

    if recommendation.get("recommendation") != "promote":
        problems.append(f"recomendação técnica é '{recommendation.get('recommendation')}', não 'promote'")
    if not recommendation.get("technical_approval"):
        problems.append("technical_approval é False na recomendação")

    if str(recommendation.get("challenger_version")) != str(run_metadata.get("model_version")):
        problems.append(
            f"challenger_version da recomendação ({recommendation.get('challenger_version')}) "
            f"difere de run_metadata.model_version ({run_metadata.get('model_version')})"
        )
    if recommendation.get("challenger_run_id") != run_metadata.get("mlflow_run_id"):
        problems.append(
            f"challenger_run_id da recomendação ({recommendation.get('challenger_run_id')}) "
            f"difere de run_metadata.mlflow_run_id ({run_metadata.get('mlflow_run_id')})"
        )
    if recommendation.get("git_commit") != run_metadata.get("git_commit"):
        problems.append(
            f"git_commit da recomendação ({recommendation.get('git_commit')}) "
            f"difere de run_metadata.git_commit ({run_metadata.get('git_commit')})"
        )

    approver = approval.get("approver")
    if not approver:
        problems.append("approval.json não contém 'approver'")
    if str(approval.get("model_version")) != str(run_metadata.get("model_version")):
        problems.append(
            f"approval.model_version ({approval.get('model_version')}) "
            f"difere de run_metadata.model_version ({run_metadata.get('model_version')})"
        )
    if approval.get("run_id") != run_metadata.get("mlflow_run_id"):
        problems.append(
            f"approval.run_id ({approval.get('run_id')}) difere de run_metadata.mlflow_run_id "
            f"({run_metadata.get('mlflow_run_id')})"
        )

    if problems:
        return fail(logger, "Metadados de promoção inconsistentes", problems=problems)

    return succeed(
        logger,
        "Metadados de promoção consistentes e aprovados",
        approver=approver,
        model_version=run_metadata.get("model_version"),
    )


if __name__ == "__main__":
    sys.exit(main())
