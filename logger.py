"""
Structured Logger

Provides consistent logging across all components.

Features:
- JSON formatted logs
- Contextual information
- Performance tracking
- Error categorization
- Log rotation
- Multi-handler support
"""

import logging
import logging.handlers
import json
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import sys
from pathlib import Path


class LogLevel(str, Enum):
    """Log levels with descriptions."""
    DEBUG = "DEBUG"        # Detailed diagnostic info
    INFO = "INFO"          # General information
    WARNING = "WARNING"    # Warning messages
    ERROR = "ERROR"        # Error messages
    CRITICAL = "CRITICAL"  # Critical failures


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing and analysis."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Add contextual information to logs."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize context filter."""
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        """Add context to record."""
        record.extra = self.context.copy()
        return True


def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    json_output: bool = True,
    log_file: Optional[Path] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name
        level: Log level
        json_output: Format logs as JSON
        log_file: Optional file path for logs
        context: Additional context for logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add context filter
    if context:
        logger.addFilter(ContextFilter(context))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (max 10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def configure_logging(
    log_dir: Path = None,
    log_level: LogLevel = LogLevel.INFO,
    json_output: bool = True
):
    """
    Configure logging for entire application.
    
    Creates log files for different components:
    - detection.log
    - orchestrator.log
    - storage.log
    - alert.log
    - audit.log
    """
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Component-specific loggers
    components = [
        'detection',
        'orchestrator',
        'storage',
        'alert',
        'audit',
        'performance'
    ]
    
    for component in components:
        logger = logging.getLogger(f'sentinelai.{component}')
        logger.setLevel(log_level)
        logger.handlers.clear()
        
        # File handler
        log_file = log_dir / f'{component}.log'
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,
            backupCount=5
        )
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return root_logger
