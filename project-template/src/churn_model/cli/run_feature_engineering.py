"""CLI: calcula e publica a tabela de features (``resources/feature-engineering.job.yml``).

Separado do treino de propósito -- a feature store existe justamente para que
o cálculo de features seja compartilhável entre execuções (e, numa organização
real, entre modelos diferentes sobre o mesmo cliente), em vez de recalculado
a cada treino.
"""

from __future__ import annotations

import sys

from churn_model.cli import base_arg_parser, bootstrap, shutdown
from churn_model.data.loader import load_raw_customers
from churn_model.exceptions import ChurnModelError, ConfigurationError
from churn_model.features.feature_store import compute_customer_features, get_feature_store
from churn_model.logging_config import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Calcula as features derivadas de clientes e publica na feature store.")
    args = parser.parse_args(argv)

    config, spark = bootstrap(args.config, app_name="churn_model-feature-engineering")
    try:
        if not config.feature_store.enabled:
            raise ConfigurationError(
                "feature_store.enabled=false neste ambiente -- nada a publicar. "
                "Habilite em conf/<ambiente>.yml para usar a feature store."
            )

        raw_df = load_raw_customers(spark, config)
        features_df = compute_customer_features(raw_df)

        store = get_feature_store(config, spark)
        store.create_or_update_table(features_df, config.feature_store)

        logger.info(
            "Feature engineering concluída",
            extra={"extra_fields": {"table_name": config.feature_store.full_table_name}},
        )
        return 0
    except ChurnModelError as exc:
        logger.error("Falha na feature engineering", extra={"extra_fields": {"error": str(exc)}})
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
