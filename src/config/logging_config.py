"""Centralized logging configuration for FenixAI Trading Bot.

This module provides a single source of truth for logging setup across
all components (API server, CLI, libraries).
"""

import logging
import logging.config
import os
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def _resolve_log_level(level):
    """Resolve log level from env var LOG_LEVEL or fallback to level arg."""
    env_val = os.getenv("LOG_LEVEL", "").upper()
    if env_val:
        resolved = LOG_LEVELS.get(env_val)
        if resolved is not None:
            return resolved
    return level


def configure_root_logger(level=logging.INFO, log_file=None):
    """Configure the root logger for the entire application.

    This MUST be called at the very start of the entry point module,
    before any other imports that might use logging.

    Args:
        level: Root logger level (can be overridden by LOG_LEVEL env var).
        log_file: Optional path to a log file.
    """
    level = _resolve_log_level(level)
    config = build_logging_config(
        level=level,
        log_file=log_file,
        include_uvicorn=False,
    )
    logging.config.dictConfig(config)


def build_logging_config(
    *,
    level=logging.INFO,
    log_file=None,
    include_uvicorn=True,
):
    """Build a dictConfig-compatible logging configuration.

    Args:
        level: Root logger level (can be overridden by LOG_LEVEL env var).
        log_file: Optional path to a log file.
        include_uvicorn: If True, include uvicorn loggers and access handler.

    Returns:
        A dict suitable for logging.config.dictConfig or uvicorn's log_config.
    """
    websocket_debug = os.getenv("LOG_LEVEL", "").upper() == "DEBUG"

    # Base formatter (for most logs)
    formatters = {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": LOG_DATE_FORMAT,
        },
    }

    # Handlers
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    }

    # Root handlers list (handlers that should receive application logs)
    root_handler_names = ["console"]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(log_file),
            "encoding": "utf-8",
        }
        root_handler_names.append("file")

    # Loggers dict
    loggers = {}

    if include_uvicorn:
        # Access handler for uvicorn.access with AccessFormatter
        handlers["access"] = {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        }
        formatters["access"] = {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s - %(levelprefix)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": LOG_DATE_FORMAT,
        }
        # Uvicorn internal loggers use console
        loggers.update(
            {
                "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
                "uvicorn.error": {"handlers": ["console"], "level": "INFO"},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            }
        )

    if websocket_debug:
        # Socket.IO and engineio should use the same handlers as root (console + file if any), NOT access.
        ws_handler_names = root_handler_names.copy()
        loggers.update(
            {
                "socketio": {"handlers": ws_handler_names, "level": "DEBUG", "propagate": False},
                "engineio": {"handlers": ws_handler_names, "level": "DEBUG", "propagate": False},
            }
        )

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {"level": level, "handlers": root_handler_names},
    }
    if loggers:
        config["loggers"] = loggers

    return config


def setup_logging(
    level=logging.INFO,
    log_file=None,
):
    """Configure logging for FenixAI using the unified configuration.

    This is suitable for non-uvicorn entry points (CLI, tests, etc.).

    Args:
        level: Root logging level (can be overridden by LOG_LEVEL env var).
        log_file: Optional log file path.
    """
    config = build_logging_config(
        level=level,
        log_file=log_file,
        include_uvicorn=False,
    )
    logging.config.dictConfig(config)


def get_uvicorn_log_config():
    """Get a uvicorn-compatible logging configuration.

    This config is intended to be assigned to uvicorn.config.LOGGING_CONFIG
    or passed to uvicorn.run(log_config=...). It does not include file logging
    by default; use build_logging_config directly if file output is needed.

    Returns:
        A dict suitable for uvicorn's logging configuration.
    """
    return build_logging_config(
        level=logging.INFO,
        log_file=None,
        include_uvicorn=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the Fenix naming convention."""
    return logging.getLogger(name)
