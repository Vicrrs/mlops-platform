"""Avaliação de modelo de classificação binária: métricas, latência e tamanho."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

from churn_model.exceptions import ModelEvaluationError
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_binary_classifier(
    predictions_df: DataFrame,
    label_column: str,
    prediction_column: str = "prediction",
    raw_prediction_column: str = "rawPrediction",
) -> dict:
    """Calcula accuracy, precision, recall, F1 e ROC AUC a partir de um DataFrame já pontuado."""
    try:
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol=label_column, rawPredictionCol=raw_prediction_column, metricName="areaUnderROC"
        )
        roc_auc = binary_evaluator.evaluate(predictions_df)

        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol=label_column, predictionCol=prediction_column
        )
        accuracy = multiclass_evaluator.evaluate(
            predictions_df, {multiclass_evaluator.metricName: "accuracy"}
        )
        precision = multiclass_evaluator.evaluate(
            predictions_df, {multiclass_evaluator.metricName: "weightedPrecision"}
        )
        recall = multiclass_evaluator.evaluate(
            predictions_df, {multiclass_evaluator.metricName: "weightedRecall"}
        )
        f1_score = multiclass_evaluator.evaluate(predictions_df, {multiclass_evaluator.metricName: "f1"})
    except Exception as exc:  # noqa: BLE001
        raise ModelEvaluationError(f"Falha ao calcular métricas de avaliação: {exc}") from exc

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1_score, 6),
        "roc_auc": round(roc_auc, 6),
        "error_rate": round(1 - accuracy, 6),
    }


def compute_invalid_prediction_percentage(
    predictions_df: DataFrame, prediction_column: str = "prediction"
) -> float:
    total = predictions_df.count()
    if total == 0:
        return 0.0
    invalid = predictions_df.where(
        predictions_df[prediction_column].isNull() | (~predictions_df[prediction_column].isin([0.0, 1.0]))
    ).count()
    return round(invalid / total, 6)


def measure_inference_latency_ms(
    pipeline_model: PipelineModel, sample_df: DataFrame, repeats: int = 3
) -> float:
    """Mede a latência média (ms) de ``transform`` + materialização sobre uma amostra pequena."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline_model.transform(sample_df).count()
        durations.append((time.perf_counter() - start) * 1000)
    return round(sum(durations) / len(durations), 3)


def compute_model_size_bytes(pipeline_model: PipelineModel) -> int:
    """Estima o tamanho do modelo persistindo-o em um diretório temporário."""
    tmp_dir = tempfile.mkdtemp(prefix="model_size_")
    try:
        model_path = str(Path(tmp_dir) / "model")
        pipeline_model.write().overwrite().save(model_path)
        total_size = sum(f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file())
        return total_size
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_model(
    pipeline_model: PipelineModel,
    df: DataFrame,
    label_column: str,
    compute_size: bool = True,
) -> dict:
    """Avaliação completa: métricas de classificação + latência + tamanho + previsões inválidas."""
    if df.rdd.isEmpty():
        raise ModelEvaluationError("Não é possível avaliar o modelo em um DataFrame vazio.")

    predictions_df = pipeline_model.transform(df)
    metrics = evaluate_binary_classifier(predictions_df, label_column)
    metrics["invalid_prediction_percentage"] = compute_invalid_prediction_percentage(predictions_df)
    metrics["latency_ms"] = measure_inference_latency_ms(pipeline_model, df.limit(min(50, df.count())))
    metrics["model_size_bytes"] = compute_model_size_bytes(pipeline_model) if compute_size else None

    logger.info("Avaliação de modelo concluída", extra={"extra_fields": {"metrics": metrics}})
    return metrics
