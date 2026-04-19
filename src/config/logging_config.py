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

# Define TRACE level (level 5, below DEBUG=10)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _resolve_log_level(level):
    """Resolve log level from env var LOG_LEVEL or fallback to level arg."""
    env_val = os.getenv("LOG_LEVEL", "").upper()
    if env_val:
        try:
            return getattr(logging, env_val, None) or level
        except (ValueError, TypeError):
            pass
    return level


def configure_root_logger(
    level=logging.INFO, log_file=None, websocket_debug=True, trace_libs=False
):
    """Configure the root logger for the entire application.

    This MUST be called at the very start of the entry point module,
    before any other imports that might use logging.

    Args:
        level: Root logger level.
        log_file: Optional path to a log file.
        websocket_debug: Enable DEBUG for Socket.IO/engineio.
        trace_libs: Enable TRACE for aiosqlite, httpcore, httpx, websockets.client.
    """
    level = _resolve_log_level(level)
    config = build_logging_config(
        level=level,
        log_file=log_file,
        websocket_debug=websocket_debug,
        include_uvicorn=False,
        trace_libs=trace_libs,
    )
    logging.config.dictConfig(config)


def build_logging_config(
    *,
    level=logging.INFO,
    log_file=None,
    websocket_debug=True,
    include_uvicorn=True,
    trace_libs=False,
):
    """Build a dictConfig-compatible logging configuration.

    Args:
        level: Root logger level.
        log_file: Optional path to a log file.
        websocket_debug: If True, add DEBUG handlers for Socket.IO and engineio.
        include_uvicorn: If True, include uvicorn loggers and access handler.
        trace_libs: If True, set TRACE level for aiosqlite, httpcore, httpx, websockets.client.

    Returns:
        A dict suitable for logging.config.dictConfig or uvicorn's log_config.
    """
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

    if trace_libs:
        # Third-party libs that need TRACE level
        lib_handler_names = root_handler_names.copy()
        loggers.update(
            {
                "aiosqlite": {
                    "handlers": lib_handler_names,
                    "level": TRACE_LEVEL,
                    "propagate": False,
                },
                "httpcore": {
                    "handlers": lib_handler_names,
                    "level": TRACE_LEVEL,
                    "propagate": False,
                },
                "httpx": {"handlers": lib_handler_names, "level": TRACE_LEVEL, "propagate": False},
                "websockets.client": {
                    "handlers": lib_handler_names,
                    "level": TRACE_LEVEL,
                    "propagate": False,
                },
            }
        )

        # Noisy libs to silence (set to WARNING)
        noisy_libs = [
            "matplotlib",
            "matplotlib.font_manager",
            "matplotlib.pyplot",
            "kaleido",
            "kaleido.kaleido",
            "passlib",
            "passlib.utils",
            "passlib.utils.compat",
            "passlib.registry",
            "PIL",
            "fontTools",
        ]
        for lib in noisy_libs:
            loggers[lib] = {"handlers": lib_handler_names, "level": "WARNING", "propagate": False}

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
    websocket_debug=True,
    trace_libs=False,
):
    """Configure logging for FenixAI using the unified configuration.

    This is suitable for non-uvicorn entry points (CLI, tests, etc.).

    Args:
        level: Root logging level.
        log_file: Optional log file path.
        websocket_debug: Enable DEBUG for Socket.IO/engineio.
        trace_libs: Enable TRACE for aiosqlite, httpcore, httpx, websockets.client.
    """
    config = build_logging_config(
        level=level,
        log_file=log_file,
        websocket_debug=websocket_debug,
        include_uvicorn=False,
        trace_libs=trace_libs,
    )
    logging.config.dictConfig(config)


def get_uvicorn_log_config(websocket_debug=True, trace_libs=False):
    """Get a uvicorn-compatible logging configuration.

    This config is intended to be assigned to uvicorn.config.LOGGING_CONFIG
    or passed to uvicorn.run(log_config=...). It does not include file logging
    by default; use build_logging_config directly if file output is needed.

    Args:
        websocket_debug: Enable DEBUG for Socket.IO/engineio.
        trace_libs: Enable TRACE for aiosqlite, httpcore, httpx, websockets.client.

    Returns:
        A dict suitable for uvicorn's logging configuration.
    """
    return build_logging_config(
        level=logging.INFO,
        log_file=None,
        websocket_debug=websocket_debug,
        include_uvicorn=True,
        trace_libs=trace_libs,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the Fenix naming convention."""
    return logging.getLogger(name)
