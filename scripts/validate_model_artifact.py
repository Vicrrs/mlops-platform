#!/usr/bin/env python3
"""Valida a integridade do artefato imutável antes do deploy em PRD.

Garante que a versão que chega em PRD é EXATAMENTE a mesma validada em HML:
recalcula o SHA-256 do artefato e compara com ``artifact_sha256.txt``, e
confere que commit/run_id/model_version não divergem do ``run_metadata.json``.

Uso:
    python scripts/validate_model_artifact.py \
        --run-metadata-path artifacts/training/run_metadata.json \
        --expected-commit $(BUILD_SOURCEVERSION) \
        --expected-run-id <run_id> \
        --expected-model-version <version> \
        --artifact-path artifacts/training/model.zip \
        --expected-sha256-path artifacts/training/artifact_sha256.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _lib import fail, get_logger, read_json, sha256_file, succeed

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metadata-path", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-model-version", required=True)
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="Arquivo cujo SHA-256 será recalculado e comparado a --expected-sha256-path.",
    )
    parser.add_argument("--expected-sha256-path", default=None)
    args = parser.parse_args(argv)

    try:
        run_metadata = read_json(args.run_metadata_path)
    except FileNotFoundError:
        return fail(logger, "run_metadata.json não encontrado", path=args.run_metadata_path)

    mismatches = []
    if run_metadata.get("git_commit") != args.expected_commit:
        mismatches.append(f"git_commit: {run_metadata.get('git_commit')} != {args.expected_commit}")
    if str(run_metadata.get("mlflow_run_id")) != str(args.expected_run_id):
        mismatches.append(f"mlflow_run_id: {run_metadata.get('mlflow_run_id')} != {args.expected_run_id}")
    if str(run_metadata.get("model_version")) != str(args.expected_model_version):
        mismatches.append(f"model_version: {run_metadata.get('model_version')} != {args.expected_model_version}")

    if args.artifact_path and args.expected_sha256_path:
        expected_sha256 = Path(args.expected_sha256_path).read_text(encoding="utf-8").strip()
        actual_sha256 = sha256_file(args.artifact_path)
        if actual_sha256 != expected_sha256:
            mismatches.append(f"artifact_sha256: {actual_sha256} != {expected_sha256}")

    if mismatches:
        return fail(logger, "Artefato imutável divergente do validado em HML", mismatches=mismatches)

    return succeed(logger, "Artefato íntegro e consistente com a execução validada em HML")


if __name__ == "__main__":
    sys.exit(main())
