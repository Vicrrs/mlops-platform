from __future__ import annotations

import pytest

from churn_model.exceptions import ChallengerNotApprovedError
from churn_model.models.champion_challenger import (
    ComparisonContext,
    compute_promotion_recommendation,
    enforce_technical_approval,
    run_champion_challenger_comparison,
)
from churn_model.models.evaluate import evaluate_model
from churn_model.models.registry import configure_mlflow, log_training_run, register_model_version, set_alias
from churn_model.models.train import prepare_datasets, train_candidate


def _context(**overrides) -> ComparisonContext:
    base = dict(
        project_name="ml-churn-test",
        registered_model_name="test_catalog.churn.churn_model",
        primary_metric="f1_score",
        champion_version=None,
        challenger_version="1",
        champion_run_id=None,
        challenger_run_id="run-1",
        git_commit="local",
        dataset_version="v1",
    )
    base.update(overrides)
    return ComparisonContext(**base)


def test_first_model_is_approved_when_absolute_thresholds_met(test_config):
    challenger_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
    recommendation = compute_promotion_recommendation(
        None, challenger_metrics, test_config.model.thresholds, _context()
    )
    assert recommendation["is_first_model"] is True
    assert recommendation["technical_approval"] is True
    assert recommendation["recommendation"] == "promote"


def test_challenger_rejected_when_below_absolute_threshold(test_config):
    challenger_metrics = {"accuracy": 0.1, "precision": 0.1, "recall": 0.1, "f1_score": 0.1, "roc_auc": 0.1}
    recommendation = compute_promotion_recommendation(
        None, challenger_metrics, test_config.model.thresholds, _context()
    )
    assert recommendation["technical_approval"] is False
    assert recommendation["recommendation"] == "reject"


def test_challenger_rejected_on_metric_regression_beyond_limit(test_config):
    champion_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
    challenger_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.80, "roc_auc": 0.9}
    recommendation = compute_promotion_recommendation(
        champion_metrics, challenger_metrics, test_config.model.thresholds, _context(champion_version="1")
    )
    assert recommendation["regression_within_limit"] is False
    assert recommendation["technical_approval"] is False


def test_challenger_approved_within_allowed_regression(test_config):
    champion_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.90, "roc_auc": 0.9}
    challenger_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.895, "roc_auc": 0.9}
    recommendation = compute_promotion_recommendation(
        champion_metrics, challenger_metrics, test_config.model.thresholds, _context(champion_version="1")
    )
    assert recommendation["regression_within_limit"] is True
    assert recommendation["technical_approval"] is True


def test_enforce_technical_approval_raises_when_rejected():
    with pytest.raises(ChallengerNotApprovedError):
        enforce_technical_approval(
            {
                "technical_approval": False,
                "recommendation": "reject",
                "primary_metric": "f1_score",
                "challenger_metric": 0.1,
            }
        )


def test_run_champion_challenger_comparison_end_to_end(sample_customers_df, test_config, tmp_path):
    train_df, val_df, test_df = prepare_datasets(sample_customers_df, test_config)
    result = train_candidate(train_df, val_df, test_df, test_config)
    metrics = evaluate_model(result.pipeline_model, val_df, "churn", compute_size=False)

    client = configure_mlflow(test_config)
    run_id = log_training_run(
        client=client,
        config=test_config,
        pipeline_model=result.pipeline_model,
        train_sample=train_df,
        metrics=metrics,
        feature_config=result.feature_config,
        dataset_version="v1",
    )
    version = register_model_version(client, test_config.mlflow.registered_model_name, run_id)
    set_alias(client, test_config.mlflow.registered_model_name, "challenger", version)

    recommendation = run_champion_challenger_comparison(
        client=client,
        config=test_config,
        challenger_model=result.pipeline_model,
        challenger_run_id=run_id,
        challenger_version=version,
        test_df=test_df,
        dataset_version="v1",
        output_dir=tmp_path / "comparison",
    )

    assert recommendation["is_first_model"] is True
    assert recommendation["challenger_run_id"] == run_id
    for filename in (
        "champion_metrics.json",
        "challenger_metrics.json",
        "comparison.json",
        "comparison_report.md",
        "promotion_recommendation.json",
    ):
        assert (tmp_path / "comparison" / filename).exists()
