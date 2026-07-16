from __future__ import annotations

from pathlib import Path

from validate_project_structure import REQUIRED_PACKAGE_MODULES, REQUIRED_PATHS, validate_structure


def _scaffold_valid_project(root: Path) -> None:
    for rel in REQUIRED_PATHS:
        path = root / rel
        if rel in ("src", "notebooks", "tests/unit", "tests/spark", "tests/integration", "tests/smoke"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder", encoding="utf-8")

    package_dir = root / "src" / "meu_pacote"
    for module in REQUIRED_PACKAGE_MODULES:
        module_path = package_dir / module
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# placeholder\n", encoding="utf-8")


def test_validate_structure_passes_on_complete_project(tmp_path):
    _scaffold_valid_project(tmp_path)
    missing = validate_structure(tmp_path)
    assert missing == []


def test_validate_structure_flags_missing_top_level_file(tmp_path):
    _scaffold_valid_project(tmp_path)
    (tmp_path / "databricks.yml").unlink()
    missing = validate_structure(tmp_path)
    assert "databricks.yml" in missing


def test_validate_structure_flags_missing_package_module(tmp_path):
    _scaffold_valid_project(tmp_path)
    (tmp_path / "src" / "meu_pacote" / "models" / "champion_challenger.py").unlink()
    missing = validate_structure(tmp_path)
    assert any("champion_challenger.py" in m for m in missing)


def test_validate_structure_flags_missing_package_entirely(tmp_path):
    tmp_path.joinpath("src").mkdir()
    missing = validate_structure(tmp_path)
    assert any("nenhum pacote" in m for m in missing)
