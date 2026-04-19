"""Logging configuration for Uvicorn when run directly via CLI.

This module provides a log_config dictionary compatible with uvicorn's
--log-config option. It ensures all logs use FenixAI's standardized format.
"""

import logging
import os
from src.config.logging_config import build_logging_config, LOG_LEVELS

# Resolve log level from environment
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_resolved_level = LOG_LEVELS.get(_log_level, logging.INFO)

# Build the logging configuration with uvicorn handlers
log_config = build_logging_config(
    level=_resolved_level,
    log_file=None,  # Console only; file logging handled separately if needed
    include_uvicorn=True,
)
