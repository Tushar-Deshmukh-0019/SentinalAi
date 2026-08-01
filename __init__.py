"""
Logging & Audit Module

Complete observability for surveillance system.

Components:
- Structured logging
- Audit trail tracking
- Error logging
- Performance logging
- Compliance reporting

Why This Matters:
=================

Defense systems MUST answer:
- What happened?
- When did it happen?
- Who was involved?
- Why did it happen?
- What was the response?

Logging enables complete accountability.
"""

from .logger import (
    get_logger,
    configure_logging,
    LogLevel
)

from .audit import (
    AuditLogger,
    AuditEvent,
    AuditLevel
)

from .formatter import (
    JSONFormatter,
    StructuredFormatter
)

__version__ = '0.1.0'

__all__ = [
    'get_logger',
    'configure_logging',
    'LogLevel',
    'AuditLogger',
    'AuditEvent',
    'AuditLevel',
    'JSONFormatter',
    'StructuredFormatter'
]
