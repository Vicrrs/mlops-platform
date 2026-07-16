"""CLI: monitoramento de dados, previsões e performance (``resources/monitoring.job.yml``)."""

from __future__ import annotations

import sys

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError
from churn_model.logging_config import get_logger
from churn_model.models.inference import score
from churn_model.models.registry import configure_mlflow
from churn_model.monitoring.data_monitoring import monitor_incoming_data, write_data_monitoring_report
from churn_model.monitoring.model_monitoring import (
    monitor_model_performance,
    write_alerts,
    write_model_performance_report,
)
from churn_model.monitoring.prediction_monitoring import (
    monitor_predictions,
    write_prediction_monitoring_report,
)

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser(
        "Executa o monitoramento de dados, previsões e performance do modelo em produção."
    )
    parser.add_argument("--output-dir", default="artifacts/monitoring")
    parser.add_argument("--model-alias", default=None)
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Caminho Delta de um lote de referência (ex.: snapshot do treino) para calcular data drift. "
        "Quando omitido, o drift não é calculado (não há baseline confiável).",
    )
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-monitoring")
    try:
        current_df = load_raw_customers(spark, config)
        model_alias = args.model_alias or config.mlflow.registered_model_alias_champion

        client = configure_mlflow(config)
        version_info = client.get_model_version_by_alias(config.mlflow.registered_model_name, model_alias)
        model_version = version_info.version if version_info else "unknown"

        reference_df = None
        if args.reference_path:
            from churn_model.io.readers import read_delta

            reference_df = read_delta(spark, args.reference_path)

        data_payload, data_alerts = monitor_incoming_data(
            current_df, config, reference_df=reference_df, model_version=model_version
        )
        write_data_monitoring_report(data_payload, args.output_dir)

        all_alerts = list(data_alerts)

        if version_info is not None:
            predictions_df = score(config, current_df, model_alias=model_alias)
            pred_payload, pred_alerts = monitor_predictions(
                predictions_df, config, model_version=model_version
            )
            write_prediction_monitoring_report(pred_payload, args.output_dir)
            all_alerts.extend(pred_alerts)

            performance_payload, performance_alerts = monitor_model_performance(
                client, config, current_df, model_alias=model_alias
            )
            write_model_performance_report(performance_payload, args.output_dir)
            all_alerts.extend(performance_alerts)
        else:
            logger.warning(
                "Nenhum modelo encontrado para o alias informado; pulando monitoramento de previsões/performance.",
                extra={"extra_fields": {"model_alias": model_alias}},
            )

        write_alerts(all_alerts, args.output_dir)

        critical_alerts = [a for a in all_alerts if a.severity == "critical"]
        if critical_alerts:
            logger.error(
                "Alertas críticos de monitoramento detectados",
                extra={"extra_fields": {"count": len(critical_alerts)}},
            )
            return 1

        logger.info("Monitoramento concluído sem alertas críticos.")
        return 0
    except ChurnModelError as exc:
        logger.error("Falha no monitoramento", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
