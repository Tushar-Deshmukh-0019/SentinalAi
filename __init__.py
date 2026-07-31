"""
Storage Module

Database persistence for surveillance system.

Components:
- PostgreSQL for structured data
- TimescaleDB for time-series data
- SQLAlchemy ORM for database access
- Alembic for schema migrations

Why This Matters:
=================

Defense surveillance systems require:
- Complete audit trails (compliance)
- Historical analysis (pattern detection)
- Evidence preservation (investigations)
- Performance tracking (optimization)
- Disaster recovery (business continuity)

In-memory storage (Days 1-7) is NOT sufficient for production.
This module provides the persistent foundation.
"""

from .models import (
    Base,
    Detection,
    Person,
    Vehicle,
    Animal,
    Object,
    Alert,
    Camera,
    PerformanceMetric
)

from .database import (
    Database,
    get_database,
    init_database
)

__version__ = '0.1.0'

__all__ = [
    # Models
    'Base',
    'Detection',
    'Person',
    'Vehicle',
    'Animal',
    'Object',
    'Alert',
    'Camera',
    'PerformanceMetric',
    
    # Database
    'Database',
    'get_database',
    'init_database'
]
