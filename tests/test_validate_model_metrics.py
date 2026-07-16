from __future__ import annotations

import json

from validate_model_metrics import main as validate_model_metrics_main


def _write_metrics(path, **metrics):
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_passes_when_all_thresholds_met(tmp_path):
    metrics_path = tmp_path / "model_metrics.json"
    _write_metrics(
        metrics_path,
        accuracy=0.9,
        precision=0.85,
        recall=0.85,
        f1_score=0.85,
        roc_auc=0.9,
        invalid_prediction_percentage=0.0,
    )
    exit_code = validate_model_metrics_main(
        [
            "--metrics-path", str(metrics_path),
            "--minimum-accuracy", "0.85", "--minimum-precision", "0.8",
            "--minimum-recall", "0.8", "--minimum-f1-score", "0.8", "--minimum-roc-auc", "0.85",
        ]
    )
    assert exit_code == 0


def test_fails_when_metric_below_minimum(tmp_path):
    metrics_path = tmp_path / "model_metrics.json"
    _write_metrics(metrics_path, accuracy=0.5, precision=0.5, recall=0.5, f1_score=0.5, roc_auc=0.5)
    exit_code = validate_model_metrics_main(
        ["--metrics-path", str(metrics_path), "--minimum-accuracy", "0.85"]
    )
    assert exit_code == 1


def test_fails_when_metric_missing_from_report(tmp_path):
    metrics_path = tmp_path / "model_metrics.json"
    _write_metrics(metrics_path, accuracy=0.9)  # precision/recall/f1/roc_auc ausentes
    exit_code = validate_model_metrics_main(
        ["--metrics-path", str(metrics_path), "--minimum-f1-score", "0.8"]
    )
    assert exit_code == 1


def test_fails_on_excessive_invalid_predictions(tmp_path):
    metrics_path = tmp_path / "model_metrics.json"
    _write_metrics(
        metrics_path, accuracy=0.9, precision=0.9, recall=0.9, f1_score=0.9, roc_auc=0.9,
        invalid_prediction_percentage=0.5,
    )
    exit_code = validate_model_metrics_main(
        ["--metrics-path", str(metrics_path), "--maximum-invalid-prediction-percentage", "0.01"]
    )
    assert exit_code == 1
