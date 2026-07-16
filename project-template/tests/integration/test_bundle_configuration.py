"""Valida a consistência entre o Databricks Asset Bundle e o pacote Python.

Garante que todo `entry_point` referenciado em resources/*.job.yml existe de
fato como console_script em pyproject.toml -- evita jobs quebrados por um
entry point renomeado/removido sem atualizar o bundle.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _declared_entry_points() -> set[str]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(pyproject["project"]["scripts"].keys())


def _entry_points_used_in_bundle() -> set[str]:
    used = set()
    for job_file in (PROJECT_ROOT / "resources").glob("*.job.yml"):
        content = yaml.safe_load(job_file.read_text(encoding="utf-8"))
        jobs = content.get("resources", {}).get("jobs", {})
        for job in jobs.values():
            for task in job.get("tasks", []):
                wheel_task = task.get("python_wheel_task")
                if wheel_task:
                    used.add(wheel_task["entry_point"])
    return used


def test_all_bundle_entry_points_exist_in_pyproject():
    declared = _declared_entry_points()
    used = _entry_points_used_in_bundle()
    assert used, "Nenhum entry point encontrado em resources/*.job.yml -- verifique os arquivos do bundle."
    missing = used - declared
    assert not missing, f"Entry points usados no bundle mas ausentes em pyproject.toml: {missing}"


def test_every_job_resource_file_has_exactly_one_job():
    for job_file in (PROJECT_ROOT / "resources").glob("*.job.yml"):
        content = yaml.safe_load(job_file.read_text(encoding="utf-8"))
        jobs = content["resources"]["jobs"]
        assert len(jobs) == 1, f"{job_file.name} deveria definir exatamente 1 job, encontrou {len(jobs)}"


def test_databricks_yml_declares_all_required_variables():
    databricks_yml = yaml.safe_load((PROJECT_ROOT / "databricks.yml").read_text(encoding="utf-8"))
    variables = databricks_yml["variables"]
    for required in ("environment", "catalog", "schema", "model_name", "git_commit", "build_id"):
        assert required in variables, f"Variável obrigatória '{required}' ausente em databricks.yml"


def test_databricks_yml_has_no_hardcoded_hosts():
    raw_text = (PROJECT_ROOT / "databricks.yml").read_text(encoding="utf-8")
    # Nenhuma URL real de workspace deve estar fixada -- apenas placeholders/variáveis.
    assert not re.search(r"https://[a-z0-9-]+\.azuredatabricks\.net", raw_text)
