from __future__ import annotations

import json

from compare_champion_challenger import compute_recommendation
from compare_champion_challenger import main as compare_main

THRESHOLDS = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6, "roc_auc": 0.6}
CONTEXT = {
    "project_name": "ml-fraude",
    "registered_model_name": "prd_catalog.fraude.modelo_fraude",
    "champion_version": "4",
    "challenger_version": "5",
    "champion_run_id": "run-champ",
    "challenger_run_id": "run-chal",
    "git_commit": "abcdef1",
    "dataset_version": "v3",
}


def test_first_model_promoted_when_thresholds_met():
    challenger_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
    recommendation = compute_recommendation(None, challenger_metrics, THRESHOLDS, 0.01, "f1_score", CONTEXT)
    assert recommendation["is_first_model"] is True
    assert recommendation["technical_approval"] is True
    assert recommendation["recommendation"] == "promote"


def test_rejected_on_regression_beyond_limit():
    champion_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.90, "roc_auc": 0.9}
    challenger_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.80, "roc_auc": 0.9}
    recommendation = compute_recommendation(champion_metrics, challenger_metrics, THRESHOLDS, 0.01, "f1_score", CONTEXT)
    assert recommendation["regression_within_limit"] is False
    assert recommendation["recommendation"] == "reject"


def test_cli_writes_artifacts_and_fails_process_on_reject(tmp_path):
    challenger_metrics_path = tmp_path / "challenger_metrics.json"
    challenger_metrics_path.write_text(
        json.dumps({"accuracy": 0.1, "precision": 0.1, "recall": 0.1, "f1_score": 0.1, "roc_auc": 0.1}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = compare_main(
        [
            "--challenger-metrics-path", str(challenger_metrics_path),
            "--primary-metric", "f1_score",
            "--minimum-f1-score", "0.5",
            "--project-name", "ml-fraude",
            "--registered-model-name", "prd_catalog.fraude.modelo_fraude",
            "--challenger-version", "5",
            "--challenger-run-id", "run-chal",
            "--git-commit", "abcdef1",
            "--dataset-version", "v3",
            "--output-dir", str(output_dir),
        ]
    )
    assert exit_code == 1
    recommendation = json.loads((output_dir / "promotion_recommendation.json").read_text(encoding="utf-8"))
    assert recommendation["recommendation"] == "reject"
    assert (output_dir / "comparison.json").exists()


def test_cli_allow_reject_flag_returns_zero_but_still_records_rejection(tmp_path):
    challenger_metrics_path = tmp_path / "challenger_metrics.json"
    challenger_metrics_path.write_text(
        json.dumps({"accuracy": 0.1, "precision": 0.1, "recall": 0.1, "f1_score": 0.1, "roc_auc": 0.1}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    exit_code = compare_main(
        [
            "--challenger-metrics-path", str(challenger_metrics_path),
            "--minimum-f1-score", "0.5",
            "--project-name", "ml-fraude",
            "--registered-model-name", "prd_catalog.fraude.modelo_fraude",
            "--challenger-version", "5",
            "--challenger-run-id", "run-chal",
            "--git-commit", "abcdef1",
            "--dataset-version", "v3",
            "--output-dir", str(output_dir),
            "--allow-reject",
        ]
    )
    assert exit_code == 0
    recommendation = json.loads((output_dir / "promotion_recommendation.json").read_text(encoding="utf-8"))
    assert recommendation["recommendation"] == "reject"
