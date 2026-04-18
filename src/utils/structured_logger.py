# utils/structured_logger.py
import asyncio
import json
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(Enum):
    """Custom log levels"""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    PERFORMANCE = 70


class AlertSeverity(Enum):
    """Alert severity"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogContext:
    """Logging context"""

    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    trade_id: str | None = None
    symbol: str | None = None
    strategy: str | None = None
    component: str | None = None
    operation: str | None = None


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento"""

    name: str
    value: float
    unit: str
    timestamp: datetime
    context: dict[str, Any] | None = None


@dataclass
class SecurityEvent:
    """Evento de seguridad"""

    event_type: str
    severity: AlertSeverity
    description: str
    source_ip: str | None = None
    user_agent: str | None = None
    additional_data: dict[str, Any] | None = None


class StructuredLogger:
    """Logger estructurado con métricas y alertas"""

    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Configure base logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LogLevel.TRACE.value)

        # Thread-local context
        self._local = threading.local()

        # Callbacks for alerts
        self.alert_callbacks: list[Callable] = []

        # In-memory metrics
        self.metrics_buffer: list[PerformanceMetric] = []
        self.metrics_lock = threading.Lock()

        # Configure handlers
        self._setup_handlers()

        # Configure security filters
        self._setup_security_filters()

    def _setup_handlers(self):
        """Configurar handlers de logging"""
        # Handler para archivo JSON estructurado
        json_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_structured.jsonl", encoding="utf-8"
        )
        json_handler.setFormatter(StructuredFormatter())
        json_handler.setLevel(LogLevel.DEBUG.value)

        # Handler for human-readable text file
        text_handler = logging.FileHandler(self.log_dir / f"{self.name}.log", encoding="utf-8")
        text_handler.setFormatter(HumanReadableFormatter())
        text_handler.setLevel(LogLevel.INFO.value)

        # Handler para errores críticos
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log", encoding="utf-8"
        )
        error_handler.setFormatter(DetailedErrorFormatter())
        error_handler.setLevel(LogLevel.ERROR.value)

        # Handler para eventos de seguridad
        security_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_security.log", encoding="utf-8"
        )
        security_handler.setFormatter(SecurityFormatter())
        security_handler.setLevel(LogLevel.SECURITY.value)

        # Handler para métricas de rendimiento
        performance_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_performance.jsonl", encoding="utf-8"
        )
        performance_handler.setFormatter(PerformanceFormatter())
        performance_handler.setLevel(LogLevel.PERFORMANCE.value)

        # Handler para consola (solo en desarrollo)
        if os.getenv("ENVIRONMENT", "production") == "development":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredConsoleFormatter())
            console_handler.setLevel(LogLevel.INFO.value)
            self.logger.addHandler(console_handler)

        # Add handlers
        for handler in [
            json_handler,
            text_handler,
            error_handler,
            security_handler,
            performance_handler,
        ]:
            self.logger.addHandler(handler)

    def _setup_security_filters(self):
        """Configure security filters to avoid logging sensitive data."""
        sensitive_patterns = [
            r"api[_-]?key",
            r"secret",
            r"password",
            r"token",
            r"private[_-]?key",
            r"auth",
            r"credential",
        ]

        # Add filter to all handlers
        security_filter = SensitiveDataFilter(sensitive_patterns)
        for handler in self.logger.handlers:
            handler.addFilter(security_filter)

    def set_context(self, **kwargs):
        """Set context for current thread."""
        if not hasattr(self._local, "context"):
            self._local.context = LogContext()

        for key, value in kwargs.items():
            if hasattr(self._local.context, key):
                setattr(self._local.context, key, value)

    def get_context(self) -> LogContext:
        """Get context of current thread."""
        if not hasattr(self._local, "context"):
            self._local.context = LogContext()
        return self._local.context

    def clear_context(self):
        """Clear context of current thread."""
        if hasattr(self._local, "context"):
            self._local.context = LogContext()

    def _create_log_record(self, level: LogLevel, message: str, **kwargs) -> dict[str, Any]:
        """Create structured log record."""
        context = self.get_context()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.name,
            "logger": self.name,
            "message": message,
            "context": asdict(context),
            "thread_id": threading.get_ident(),
            "process_id": os.getpid(),
            **kwargs,
        }

        # Add stack info if error
        if level.value >= LogLevel.ERROR.value:
            record["stack_trace"] = "".join(traceback.format_stack())

        return record

    def trace(self, message: str, **kwargs):
        """Log level TRACE"""
        record = self._create_log_record(LogLevel.TRACE, message, **kwargs)
        self.logger.log(LogLevel.TRACE.value, json.dumps(record, ensure_ascii=False))

    def debug(self, message: str, **kwargs):
        """Log level DEBUG"""
        record = self._create_log_record(LogLevel.DEBUG, message, **kwargs)
        self.logger.log(LogLevel.DEBUG.value, json.dumps(record, ensure_ascii=False))

    def info(self, message: str, **kwargs):
        """Log level INFO"""
        record = self._create_log_record(LogLevel.INFO, message, **kwargs)
        self.logger.log(LogLevel.INFO.value, json.dumps(record, ensure_ascii=False))

    def warning(self, message: str, **kwargs):
        """Log level WARNING"""
        record = self._create_log_record(LogLevel.WARNING, message, **kwargs)
        self.logger.log(LogLevel.WARNING.value, json.dumps(record, ensure_ascii=False))

        # Send alert if needed
        self._send_alert(AlertSeverity.LOW, message, **kwargs)

    def error(self, message: str, exception: Exception | None = None, **kwargs):
        """Log level ERROR"""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                ),
            }

        record = self._create_log_record(LogLevel.ERROR, message, **kwargs)
        self.logger.log(LogLevel.ERROR.value, json.dumps(record, ensure_ascii=False))

        # Send alert
        self._send_alert(AlertSeverity.MEDIUM, message, **kwargs)

    def critical(self, message: str, exception: Exception | None = None, **kwargs):
        """Log level CRITICAL"""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                ),
            }

        record = self._create_log_record(LogLevel.CRITICAL, message, **kwargs)
        self.logger.log(LogLevel.CRITICAL.value, json.dumps(record, ensure_ascii=False))

        # Send critical alert
        self._send_alert(AlertSeverity.CRITICAL, message, **kwargs)

    def security(self, event: SecurityEvent):
        """Log security event"""
        # Convert event to dict with serializable enums
        event_dict = asdict(event)
        event_dict["severity"] = event.severity.value  # Convert enum to string

        record = self._create_log_record(
            LogLevel.SECURITY, f"Security event: {event.event_type}", security_event=event_dict
        )
        self.logger.log(LogLevel.SECURITY.value, json.dumps(record, ensure_ascii=False))

        # Send security alert
        self._send_alert(
            event.severity, f"Security: {event.description}", security_event=event_dict
        )

    def performance(self, metric: PerformanceMetric):
        """Log métrica de rendimiento"""
        with self.metrics_lock:
            self.metrics_buffer.append(metric)

        # Convertir el metric a dict con datetime serializable
        metric_dict = asdict(metric)
        metric_dict["timestamp"] = metric.timestamp.isoformat()  # Convertir datetime a string

        record = self._create_log_record(
            LogLevel.PERFORMANCE, f"Performance metric: {metric.name}", metric=metric_dict
        )
        self.logger.log(LogLevel.PERFORMANCE.value, json.dumps(record, ensure_ascii=False))

    def log_trade(self, trade_data: dict[str, Any]):
        """Log específico para trades"""
        self.set_context(
            trade_id=trade_data.get("trade_id"), symbol=trade_data.get("symbol"), operation="trade"
        )

        self.info("Trade executed", trade_data=trade_data)

    def log_api_call(
        self, endpoint: str, method: str, status_code: int, duration_ms: float, **kwargs
    ):
        """Log específico para llamadas API"""
        self.set_context(operation="api_call")

        level = LogLevel.INFO if status_code < 400 else LogLevel.ERROR

        api_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs,
        }

        if level == LogLevel.INFO:
            self.info(f"API call: {method} {endpoint}", api_data=api_data)
        else:
            self.error(f"API call failed: {method} {endpoint}", api_data=api_data)

        # Log métrica de rendimiento
        self.performance(
            PerformanceMetric(
                name=f"api_call_{endpoint.replace('/', '_')}",
                value=duration_ms,
                unit="ms",
                timestamp=datetime.now(timezone.utc),
                context=api_data,
            )
        )

    def register_alert_callback(
        self, callback: Callable[[AlertSeverity, str, dict[str, Any]], None]
    ):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)

    def _send_alert(self, severity: AlertSeverity, message: str, **kwargs):
        """Enviar alerta a callbacks registrados"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(severity, message, kwargs))
                else:
                    callback(severity, message, kwargs)
            except Exception as e:
                # Avoid infinite recursion
                print(f"Error in alert callback: {e}")

    def get_metrics(self, clear_buffer: bool = True) -> list[PerformanceMetric]:
        """Get metrics from buffer."""
        with self.metrics_lock:
            metrics = self.metrics_buffer.copy()
            if clear_buffer:
                self.metrics_buffer.clear()
        return metrics


# Custom formatters
class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logs"""

    def format(self, record):
        return record.getMessage()


class HumanReadableFormatter(logging.Formatter):
    """Formatter for human-readable logs"""

    def format(self, record):
        try:
            data = json.loads(record.getMessage())
            timestamp = data.get("timestamp", "")
            level = data.get("level", "")
            message = data.get("message", "")
            context = data.get("context", {})

            context_str = ""
            if any(context.values()):
                context_parts = [f"{k}={v}" for k, v in context.items() if v]
                context_str = f" [{', '.join(context_parts)}]"

            return f"{timestamp} [{level}]{context_str} {message}"
        except json.JSONDecodeError:
            return record.getMessage()


class DetailedErrorFormatter(logging.Formatter):
    """Detailed formatter for errors"""

    def format(self, record):
        try:
            data = json.loads(record.getMessage())
            if data.get("level") in ["ERROR", "CRITICAL"]:
                return json.dumps(data, indent=2, ensure_ascii=False)
            return record.getMessage()
        except json.JSONDecodeError:
            return record.getMessage()


class SecurityFormatter(logging.Formatter):
    """Formatter para eventos de seguridad"""

    def format(self, record):
        try:
            data = json.loads(record.getMessage())
            if "security_event" in data:
                return json.dumps(data, indent=2, ensure_ascii=False)
            return record.getMessage()
        except json.JSONDecodeError:
            return record.getMessage()


class PerformanceFormatter(logging.Formatter):
    """Formatter for performance metrics"""

    def format(self, record):
        return record.getMessage()


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter con colores para consola"""

    COLORS = {
        "TRACE": "\033[90m",  # Gris
        "DEBUG": "\033[36m",  # Cian
        "INFO": "\033[32m",  # Verde
        "WARNING": "\033[33m",  # Amarillo
        "ERROR": "\033[31m",  # Rojo
        "CRITICAL": "\033[35m",  # Magenta
        "SECURITY": "\033[41m",  # Fondo rojo
        "PERFORMANCE": "\033[34m",  # Azul
    }
    RESET = "\033[0m"

    def format(self, record):
        try:
            data = json.loads(record.getMessage())
            level = data.get("level", "")
            message = data.get("message", "")

            color = self.COLORS.get(level, "")
            return f"{color}[{level}]{self.RESET} {message}"
        except json.JSONDecodeError:
            return record.getMessage()


class SensitiveDataFilter(logging.Filter):
    """Filter to avoid logging sensitive data."""

    def __init__(self, sensitive_patterns: list[str]):
        super().__init__()
        import re

        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in sensitive_patterns]

    def filter(self, record):
        message = record.getMessage()

        # Check if message contains sensitive data
        for pattern in self.patterns:
            if pattern.search(message):
                # Replace sensitive data with [REDACTED]
                message = pattern.sub("[REDACTED]", message)
                record.msg = message

        return True


# Factory to create loggers
def get_logger(name: str, log_dir: str = "logs") -> StructuredLogger:
    """Factory to create structured loggers."""
    return StructuredLogger(name, log_dir)


# Logger global para el sistema
system_logger = get_logger("fenix_system")
