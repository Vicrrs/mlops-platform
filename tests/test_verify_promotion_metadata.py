from __future__ import annotations

import json

from verify_promotion_metadata import main as verify_main


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _consistent_fixtures(tmp_path):
    recommendation = {
        "recommendation": "promote",
        "technical_approval": True,
        "challenger_version": "5",
        "challenger_run_id": "run-5",
        "git_commit": "sha123",
    }
    run_metadata = {"model_version": "5", "mlflow_run_id": "run-5", "git_commit": "sha123"}
    approval = {"approver": "jane@empresa.com", "model_version": "5", "run_id": "run-5"}

    rec_path = tmp_path / "promotion_recommendation.json"
    run_path = tmp_path / "run_metadata.json"
    approval_path = tmp_path / "approval.json"
    _write(rec_path, recommendation)
    _write(run_path, run_metadata)
    _write(approval_path, approval)
    return rec_path, run_path, approval_path


def test_passes_when_everything_matches(tmp_path):
    rec_path, run_path, approval_path = _consistent_fixtures(tmp_path)
    exit_code = verify_main(
        [
            "--promotion-recommendation-path", str(rec_path),
            "--run-metadata-path", str(run_path),
            "--approval-path", str(approval_path),
        ]
    )
    assert exit_code == 0


def test_fails_when_recommendation_is_reject(tmp_path):
    rec_path, run_path, approval_path = _consistent_fixtures(tmp_path)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["recommendation"] = "reject"
    rec["technical_approval"] = False
    _write(rec_path, rec)

    exit_code = verify_main(
        [
            "--promotion-recommendation-path", str(rec_path),
            "--run-metadata-path", str(run_path),
            "--approval-path", str(approval_path),
        ]
    )
    assert exit_code == 1


def test_fails_when_approval_references_different_version(tmp_path):
    rec_path, run_path, approval_path = _consistent_fixtures(tmp_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["model_version"] = "999"
    _write(approval_path, approval)

    exit_code = verify_main(
        [
            "--promotion-recommendation-path", str(rec_path),
            "--run-metadata-path", str(run_path),
            "--approval-path", str(approval_path),
        ]
    )
    assert exit_code == 1


def test_fails_when_approver_missing(tmp_path):
    rec_path, run_path, approval_path = _consistent_fixtures(tmp_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    del approval["approver"]
    _write(approval_path, approval)

    exit_code = verify_main(
        [
            "--promotion-recommendation-path", str(rec_path),
            "--run-metadata-path", str(run_path),
            "--approval-path", str(approval_path),
        ]
    )
    assert exit_code == 1


def test_fails_when_git_commit_diverges(tmp_path):
    rec_path, run_path, approval_path = _consistent_fixtures(tmp_path)
    run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
    run_metadata["git_commit"] = "different-sha"
    _write(run_path, run_metadata)

    exit_code = verify_main(
        [
            "--promotion-recommendation-path", str(rec_path),
            "--run-metadata-path", str(run_path),
            "--approval-path", str(approval_path),
        ]
    )
    assert exit_code == 1
