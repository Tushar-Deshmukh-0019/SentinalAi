"""
Configuration Manager

Centralized configuration management with:
- Multi-format support (YAML, JSON)
- Environment-specific overrides
- Runtime configuration changes
- Validation and defaults
- Change notification system
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Callable, List
from pathlib import Path
import yaml

from .validator import ConfigValidator, ValidationError

logger = logging.getLogger('config.manager')


class ConfigManager:
    """
    Centralized configuration manager for SentinelAI.
    
    Supports:
    - Multiple configuration files (app.yaml, app.{environment}.yaml)
    - Environment variable overrides (SENTINELAI_*)
    - Runtime configuration updates
    - Validation against schema
    - Change notification callbacks
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    _validator: ConfigValidator = ConfigValidator()
    _callbacks: Dict[str, List[Callable]] = {}
    _environment: str = 'development'
    
    def __new__(cls):
        """Singleton pattern for ConfigManager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load_config(
        cls,
        config_file: Optional[str] = None,
        environment: Optional[str] = None
    ) -> 'ConfigManager':
        """
        Load configuration from file(s) and environment variables.
        
        Configuration loading priority (highest to lowest):
        1. Environment variables (SENTINELAI_*)
        2. Environment-specific file (app.{environment}.yaml)
        3. Base configuration file (app.yaml)
        4. Validator defaults
        
        Args:
            config_file: Path to base configuration file (default: config/app.yaml)
            environment: Environment name (dev/staging/prod, default: from env var)
            
        Returns:
            ConfigManager instance with loaded configuration
        """
        instance = cls()
        
        # Determine environment
        if environment:
            instance._environment = environment
        else:
            instance._environment = os.getenv('SENTINELAI_ENV', 'development')
        
        logger.info(f"Loading configuration for environment: {instance._environment}")
        
        # Determine config file path
        if not config_file:
            config_file = os.getenv(
                'SENTINELAI_CONFIG',
                'config/app.yaml'
            )
        
        config_path = Path(config_file)
        instance._config = {}
        
        # Load base configuration
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            instance._config = instance._load_file(config_path)
        else:
            logger.warning(f"Configuration file not found: {config_path}")
        
        # Load environment-specific configuration
        env_config_path = config_path.parent / f"app.{instance._environment}.yaml"
        if env_config_path.exists():
            logger.info(f"Loading environment configuration from {env_config_path}")
            env_config = instance._load_file(env_config_path)
            instance._merge_configs(instance._config, env_config)
        
        # Apply environment variable overrides
        instance._apply_env_overrides()
        
        # Apply defaults for missing values
        instance._apply_defaults()
        
        # Validate configuration
        errors = instance._validator.validate(instance._flatten_config(instance._config))
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        logger.info("Configuration loaded and validated successfully")
        return instance
    
    @staticmethod
    def _load_file(file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            return {}
    
    @staticmethod
    def _merge_configs(base: Dict, override: Dict):
        """Recursively merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides (SENTINELAI_* prefix)"""
        for key, value in os.environ.items():
            if not key.startswith('SENTINELAI_'):
                continue
            
            # Convert SENTINELAI_DATABASE_HOST to database.host
            config_key = key[11:].lower().replace('_', '.')
            
            # Parse value type
            parsed_value = self._parse_env_value(value)
            
            logger.debug(f"Environment override: {config_key} = {parsed_value}")
            self.set(config_key, parsed_value)
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _apply_defaults(self):
        """Apply default values from validator"""
        all_rules = self._validator.get_all_rules()
        flat_config = self._flatten_config(self._config)
        
        for key, rule in all_rules.items():
            if key not in flat_config and 'default' in rule:
                self.set(key, rule['default'])
    
    @staticmethod
    def _flatten_config(config: Dict[str, Any], parent: str = '') -> Dict[str, Any]:
        """Flatten nested configuration dictionary to dot-notation"""
        flat = {}
        
        for key, value in config.items():
            full_key = f"{parent}.{key}" if parent else key
            
            if isinstance(value, dict):
                flat.update(ConfigManager._flatten_config(value, full_key))
            else:
                flat[full_key] = value
        
        return flat
    
    @staticmethod
    def _unflatten_config(flat: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dot-notation configuration back to nested dictionary"""
        nested = {}
        
        for key, value in flat.items():
            parts = key.split('.')
            current = nested
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return nested
    
    def get(
        self,
        key: str,
        default: Any = None,
        type_hint: Optional[type] = None
    ) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
            type_hint: Expected type for validation
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config = ConfigManager.load_config()
            >>> host = config.get('database.host', 'localhost')
            >>> timeout = config.get('processing.timeout', 30, int)
        """
        flat_config = self._flatten_config(self._config)
        value = flat_config.get(key, default)
        
        if value is None and 'default' not in self._validator.get_rule(key) or {}:
            return default
        
        return value
    
    def set(
        self,
        key: str,
        value: Any,
        notify: bool = True
    ) -> None:
        """
        Set configuration value by dot-notation key.
        
        Validates value against schema and triggers change callbacks.
        
        Args:
            key: Configuration key (e.g., 'database.host')
            value: New value
            notify: Whether to trigger change callbacks
            
        Raises:
            ValidationError: If value fails validation
            
        Example:
            >>> config = ConfigManager.load_config()
            >>> config.set('database.host', 'new-host')
            >>> config.subscribe('database.host', callback)
        """
        # Validate single value
        test_config = self._flatten_config(self._config)
        test_config[key] = value
        
        rule = self._validator.get_rule(key)
        if rule:
            # Type check
            if 'type' in rule and not isinstance(value, rule['type']):
                raise ValidationError(
                    f"Invalid type for {key}: expected {rule['type'].__name__}"
                )
            
            # Enum check
            if 'enum' in rule and value not in rule['enum']:
                raise ValidationError(
                    f"Invalid value for {key}: {value} not in {rule['enum']}"
                )
            
            # Range check
            if isinstance(value, (int, float)):
                if 'min' in rule and value < rule['min']:
                    raise ValidationError(
                        f"Value for {key} below minimum: {value} < {rule['min']}"
                    )
                if 'max' in rule and value > rule['max']:
                    raise ValidationError(
                        f"Value for {key} exceeds maximum: {value} > {rule['max']}"
                    )
        
        # Get old value for comparison
        old_value = self.get(key)
        
        # Update nested structure
        parts = key.split('.')
        current = self._config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
        
        logger.debug(f"Configuration updated: {key} = {value}")
        
        # Notify subscribers
        if notify and old_value != value:
            self._notify_subscribers(key, old_value, value)
    
    def subscribe(self, key: str, callback: Callable) -> None:
        """
        Subscribe to configuration changes.
        
        Args:
            key: Configuration key to watch
            callback: Function to call on change
                     (signature: callback(key, old_value, new_value))
                     
        Example:
            >>> def on_log_level_change(key, old, new):
            ...     logger.setLevel(new)
            >>> config.subscribe('logging.level', on_log_level_change)
        """
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
        logger.debug(f"Subscribed to {key}")
    
    def unsubscribe(self, key: str, callback: Callable) -> None:
        """Unsubscribe from configuration changes"""
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)
            logger.debug(f"Unsubscribed from {key}")
    
    def _notify_subscribers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify subscribers of configuration change"""
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in config change callback for {key}: {e}")
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive values (default: False)
            
        Returns:
            Configuration dictionary
        """
        if include_sensitive:
            return self._config.copy()
        
        # Remove sensitive values
        config_copy = json.loads(json.dumps(self._config))
        flat = self._flatten_config(config_copy)
        
        for key in list(flat.keys()):
            rule = self._validator.get_rule(key)
            if rule and rule.get('sensitive'):
                flat[key] = '***REDACTED***'
        
        return self._unflatten_config(flat)
    
    def get_environment(self) -> str:
        """Get current environment (development/staging/production)"""
        return self._environment
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self._environment == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self._environment == 'development'
    
    def reload(self, config_file: Optional[str] = None) -> None:
        """Reload configuration from file"""
        logger.info("Reloading configuration")
        self.__class__.load_config(config_file, self._environment)
    
    def export_schema(self, output_file: str) -> None:
        """Export configuration schema to file for reference"""
        schema = {}
        
        for key, rule in self._validator.get_all_rules().items():
            schema[key] = {
                'type': rule.get('type').__name__ if isinstance(rule.get('type'), type) else str(rule.get('type')),
                'default': rule.get('default'),
                'description': rule.get('description', ''),
            }
            
            if 'enum' in rule:
                schema[key]['enum'] = rule['enum']
            if 'min' in rule:
                schema[key]['min'] = rule['min']
            if 'max' in rule:
                schema[key]['max'] = rule['max']
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=True)
        
        logger.info(f"Configuration schema exported to {output_path}")
