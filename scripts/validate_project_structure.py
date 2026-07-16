#!/usr/bin/env python3
"""Valida que um repositório consumidor obedece à estrutura obrigatória de projeto ML.

Uso:
    python scripts/validate_project_structure.py --project-path /caminho/para/ml-fraude
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _lib import fail, get_logger, succeed, write_json

logger = get_logger(__name__)

REQUIRED_PATHS = [
    "azure-pipelines.yml",
    "databricks.yml",
    "pyproject.toml",
    "README.md",
    "conf/base.yml",
    "conf/dev.yml",
    "conf/hml.yml",
    "conf/prd.yml",
    "resources/data-quality.job.yml",
    "resources/eda.job.yml",
    "resources/training.job.yml",
    "resources/model-validation.job.yml",
    "resources/batch-inference.job.yml",
    "resources/monitoring.job.yml",
    "resources/smoke-test.job.yml",
    "src",
    "notebooks",
    "tests/unit",
    "tests/spark",
    "tests/integration",
    "tests/smoke",
]

REQUIRED_PACKAGE_MODULES = [
    "__init__.py",
    "config.py",
    "exceptions.py",
    "logging_config.py",
    "spark.py",
    "io/readers.py",
    "io/writers.py",
    "data/schemas.py",
    "data/validation.py",
    "data/quality.py",
    "data/eda.py",
    "features/transformations.py",
    "features/pipeline.py",
    "models/train.py",
    "models/evaluate.py",
    "models/registry.py",
    "models/champion_challenger.py",
    "models/inference.py",
    "monitoring/data_monitoring.py",
    "monitoring/prediction_monitoring.py",
    "monitoring/model_monitoring.py",
]


def find_package_dir(project_path: Path) -> Path | None:
    src_dir = project_path / "src"
    if not src_dir.is_dir():
        return None
    candidates = [p for p in src_dir.iterdir() if p.is_dir() and (p / "config.py").exists()]
    return candidates[0] if candidates else None


def validate_structure(project_path: Path) -> list[str]:
    missing = [str(rel) for rel in REQUIRED_PATHS if not (project_path / rel).exists()]

    package_dir = find_package_dir(project_path)
    if package_dir is None:
        missing.append("src/<package_name>/ (nenhum pacote com config.py encontrado)")
    else:
        for module in REQUIRED_PACKAGE_MODULES:
            if not (package_dir / module).exists():
                missing.append(f"src/{package_dir.name}/{module}")

    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--output-path", default=None, help="Caminho opcional para gravar o relatório JSON.")
    args = parser.parse_args(argv)

    project_path = Path(args.project_path)
    if not project_path.is_dir():
        return fail(logger, "Caminho de projeto não encontrado", project_path=str(project_path))

    missing = validate_structure(project_path)
    report = {
        "project_path": str(project_path),
        "passed": not missing,
        "missing_paths": missing,
    }
    if args.output_path:
        write_json(args.output_path, report)

    if missing:
        return fail(logger, "Estrutura de projeto inválida", missing_count=len(missing), missing=missing)

    return succeed(logger, "Estrutura de projeto válida", project_path=str(project_path))


if __name__ == "__main__":
    sys.exit(main())
