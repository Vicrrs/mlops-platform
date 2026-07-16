"""Logging estruturado (JSON) para execução local, Databricks e Azure DevOps.

Todos os módulos do pacote devem usar ``get_logger(__name__)`` em vez de
``print``, para que os logs possam ser correlacionados por ``run_id``,
``project_name`` e ``environment`` em qualquer ambiente de execução.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False
_CONTEXT: dict[str, Any] = {}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_CONTEXT)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configura o handler raiz do processo uma única vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.handlers = [handler]
    _CONFIGURED = True


def bind_context(**kwargs: Any) -> None:
    """Adiciona campos fixos (project_name, environment, run_id...) a todo log subsequente."""
    _CONTEXT.update({k: v for k, v in kwargs.items() if v is not None})


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
