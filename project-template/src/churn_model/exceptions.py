"""Exceções específicas do domínio do projeto.

Usar exceções tipadas em vez de Exception genérica permite que a pipeline
Azure DevOps e os jobs Databricks distingam falhas de negócio (dados,
qualidade, modelo) de falhas de infraestrutura.
"""

from __future__ import annotations


class ChurnModelError(Exception):
    """Erro base para o pacote churn_model."""


class ConfigurationError(ChurnModelError):
    """Configuração ausente, inválida ou inconsistente."""


class SparkSessionError(ChurnModelError):
    """Falha ao criar ou obter a SparkSession."""


class DataReadError(ChurnModelError):
    """Falha ao ler dados de uma fonte configurada."""


class DataWriteError(ChurnModelError):
    """Falha ao gravar dados em um destino configurado."""


class EmptyDataFrameError(ChurnModelError):
    """DataFrame vazio quando dados eram esperados."""


class SchemaValidationError(ChurnModelError):
    """Schema do DataFrame não corresponde ao schema esperado."""


class DataQualityError(ChurnModelError):
    """Uma ou mais regras de qualidade de dados com severidade bloqueante falharam."""

    def __init__(self, message: str, failed_rules: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed_rules = failed_rules or []


class FeatureEngineeringError(ChurnModelError):
    """Falha ao construir ou aplicar o pipeline de features."""


class ModelTrainingError(ChurnModelError):
    """Falha durante o treinamento do modelo."""


class ModelEvaluationError(ChurnModelError):
    """Falha durante a avaliação do modelo."""


class ModelRegistryError(ChurnModelError):
    """Falha ao interagir com o MLflow Model Registry."""


class ChampionNotFoundError(ModelRegistryError):
    """Nenhum modelo com alias champion foi encontrado."""


class ChallengerNotApprovedError(ChurnModelError):
    """O Challenger não atendeu aos critérios técnicos de aprovação."""


class PromotionError(ChurnModelError):
    """Falha ao promover um modelo para Champion."""


class ArtifactIntegrityError(ChurnModelError):
    """Falha de integridade de artefato imutável (hash, versão ou commit divergente)."""


class RollbackError(ChurnModelError):
    """Falha durante a execução de rollback."""


class InferenceError(ChurnModelError):
    """Falha durante a execução de inferência."""


class MonitoringError(ChurnModelError):
    """Falha durante a coleta ou avaliação de métricas de monitoramento."""
