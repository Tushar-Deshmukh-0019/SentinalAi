"""
Configuration Module for SentinelAI

Provides centralized, environment-aware configuration management with:
- YAML/JSON configuration loading
- Environment variable overrides
- Validation and defaults
- Runtime reconfiguration support
- Multi-environment support (development, staging, production)

Usage:
    from ai.config import ConfigManager
    
    # Load configuration
    config = ConfigManager.load_config()
    
    # Access settings
    db_host = config.get('database.host')
    
    # Override at runtime
    config.set('database.host', 'new-host')
    
    # Get complete config dict
    all_settings = config.to_dict()
"""

from .manager import ConfigManager
from .validator import ConfigValidator

__all__ = ['ConfigManager', 'ConfigValidator']
