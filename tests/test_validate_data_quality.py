from __future__ import annotations

import json

from validate_data_quality import main as validate_data_quality_main


def _write_report(path, passed: bool, rules: list[dict]):
    path.write_text(json.dumps({"passed": passed, "row_count": 1000, "rules": rules}), encoding="utf-8")


def test_passes_when_report_is_clean(tmp_path):
    report_path = tmp_path / "data_quality_report.json"
    _write_report(report_path, True, [{"name": "required_columns", "status": "pass", "severity": "critical"}])
    assert validate_data_quality_main(["--report-path", str(report_path)]) == 0


def test_fails_on_blocking_severity_failure(tmp_path):
    report_path = tmp_path / "data_quality_report.json"
    _write_report(
        report_path,
        False,
        [
            {"name": "required_columns", "status": "pass", "severity": "critical"},
            {"name": "null_percentage[x]", "status": "fail", "severity": "error"},
        ],
    )
    assert validate_data_quality_main(["--report-path", str(report_path)]) == 1


def test_does_not_fail_on_non_blocking_warning(tmp_path):
    report_path = tmp_path / "data_quality_report.json"
    _write_report(
        report_path,
        True,
        [{"name": "cardinality[x]", "status": "fail", "severity": "warning"}],
    )
    assert validate_data_quality_main(["--report-path", str(report_path)]) == 0


def test_fails_when_report_file_missing(tmp_path):
    assert validate_data_quality_main(["--report-path", str(tmp_path / "does_not_exist.json")]) == 1
