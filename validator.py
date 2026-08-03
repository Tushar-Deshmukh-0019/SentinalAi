"""
Configuration Validator

Validates configuration values against defined schemas with:
- Type checking
- Range validation  
- Enum constraints
- Required field verification
- Custom validation rules
- Detailed error reporting
"""

from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import logging

logger = logging.getLogger('config.validator')


class ValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigValidator:
    """Validates configuration against defined schema"""
    
    def __init__(self):
        """Initialize validator with default rules"""
        self.rules: Dict[str, Dict[str, Any]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Define validation rules for all configuration parameters"""
        self.rules = {
            # Detection Configuration
            'detection.confidence_thresholds.person': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.45,
                'description': 'Minimum confidence for person detection'
            },
            'detection.confidence_thresholds.vehicle': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.50,
                'description': 'Minimum confidence for vehicle detection'
            },
            'detection.confidence_thresholds.animal': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.40,
                'description': 'Minimum confidence for animal detection'
            },
            'detection.confidence_thresholds.object': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.50,
                'description': 'Minimum confidence for object detection'
            },
            'detection.nms_threshold': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.45,
                'description': 'Non-Maximum Suppression threshold'
            },
            'detection.model_size': {
                'type': str,
                'enum': ['nano', 'small', 'medium', 'large', 'xlarge'],
                'default': 'medium',
                'description': 'YOLOv8 model size for detection'
            },
            
            # Camera Configuration
            'camera.max_cameras': {
                'type': int,
                'min': 1,
                'max': 100,
                'default': 10,
                'description': 'Maximum number of simultaneous camera streams'
            },
            'camera.buffer_size': {
                'type': int,
                'min': 1,
                'max': 100,
                'default': 10,
                'description': 'Frame buffer size per camera'
            },
            'camera.reconnection_attempts': {
                'type': int,
                'min': 1,
                'max': 10,
                'default': 5,
                'description': 'Number of reconnection attempts on failure'
            },
            'camera.reconnection_delay': {
                'type': int,
                'min': 1,
                'max': 60,
                'default': 5,
                'description': 'Delay in seconds between reconnection attempts'
            },
            
            # Frame Buffer Configuration
            'buffer.max_priority_levels': {
                'type': int,
                'min': 2,
                'max': 10,
                'default': 5,
                'description': 'Number of priority levels in frame buffer'
            },
            'buffer.critical_preserve_rate': {
                'type': float,
                'min': 0.5,
                'max': 1.0,
                'default': 0.99,
                'description': 'Target preserve rate for CRITICAL priority frames'
            },
            'buffer.minimal_preserve_rate': {
                'type': float,
                'min': 0.0,
                'max': 0.5,
                'default': 0.44,
                'description': 'Target preserve rate for MINIMAL priority frames'
            },
            
            # Processing Configuration
            'processing.mode': {
                'type': str,
                'enum': ['sequential', 'parallel'],
                'default': 'sequential',
                'description': 'Detection processing mode'
            },
            'processing.max_workers': {
                'type': int,
                'min': 1,
                'max': 32,
                'default': 4,
                'description': 'Maximum worker threads for parallel processing'
            },
            'processing.timeout': {
                'type': int,
                'min': 1,
                'max': 60,
                'default': 30,
                'description': 'Processing timeout in seconds'
            },
            
            # Database Configuration
            'database.host': {
                'type': str,
                'default': 'localhost',
                'description': 'Database host address'
            },
            'database.port': {
                'type': int,
                'min': 1,
                'max': 65535,
                'default': 5432,
                'description': 'Database port'
            },
            'database.name': {
                'type': str,
                'default': 'sentinelai',
                'description': 'Database name'
            },
            'database.user': {
                'type': str,
                'default': 'postgres',
                'description': 'Database user'
            },
            'database.password': {
                'type': str,
                'description': 'Database password (sensitive)',
                'sensitive': True
            },
            'database.pool_size': {
                'type': int,
                'min': 1,
                'max': 50,
                'default': 10,
                'description': 'Database connection pool size'
            },
            'database.max_overflow': {
                'type': int,
                'min': 0,
                'max': 100,
                'default': 20,
                'description': 'Maximum overflow connections'
            },
            'database.echo': {
                'type': bool,
                'default': False,
                'description': 'Enable SQL logging'
            },
            
            # Logging Configuration
            'logging.level': {
                'type': str,
                'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                'default': 'INFO',
                'description': 'Logging level'
            },
            'logging.format': {
                'type': str,
                'default': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'description': 'Log format string'
            },
            'logging.file': {
                'type': str,
                'default': 'logs/sentinelai.log',
                'description': 'Log file path'
            },
            'logging.max_size': {
                'type': int,
                'min': 1024,
                'default': 10485760,  # 10MB
                'description': 'Maximum log file size in bytes'
            },
            'logging.backup_count': {
                'type': int,
                'min': 1,
                'max': 20,
                'default': 5,
                'description': 'Number of backup log files to keep'
            },
            'logging.enable_audit': {
                'type': bool,
                'default': True,
                'description': 'Enable audit logging'
            },
            
            # Threat Assessment Configuration
            'threat.min_score': {
                'type': int,
                'min': 0,
                'max': 100,
                'default': 0,
                'description': 'Minimum threat score for alerts'
            },
            'threat.alert_threshold': {
                'type': int,
                'min': 0,
                'max': 100,
                'default': 60,
                'description': 'Threat score threshold for operator alert'
            },
            'threat.critical_threshold': {
                'type': int,
                'min': 0,
                'max': 100,
                'default': 85,
                'description': 'Threat score threshold for critical alert'
            },
            
            # Feature Flags
            'features.abandoned_object_detection': {
                'type': bool,
                'default': True,
                'description': 'Enable abandoned object detection'
            },
            'features.animal_filtering': {
                'type': bool,
                'default': True,
                'description': 'Enable animal false-positive filtering'
            },
            'features.conflict_resolution': {
                'type': bool,
                'default': True,
                'description': 'Enable person-animal conflict resolution'
            },
            'features.performance_monitoring': {
                'type': bool,
                'default': True,
                'description': 'Enable performance metrics collection'
            },
        }
    
    def validate(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against rules.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for key, rule in self.rules.items():
            if key not in config_dict:
                # Check if required (no default)
                if 'default' not in rule:
                    errors.append(f"Missing required configuration: {key}")
                continue
            
            value = config_dict[key]
            
            # Type validation
            if 'type' in rule:
                expected_type = rule['type']
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for {key}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                    continue
            
            # Enum validation
            if 'enum' in rule:
                if value not in rule['enum']:
                    errors.append(
                        f"Invalid value for {key}: {value} not in {rule['enum']}"
                    )
                    continue
            
            # Numeric range validation
            if isinstance(value, (int, float)):
                if 'min' in rule and value < rule['min']:
                    errors.append(
                        f"Value for {key} below minimum: {value} < {rule['min']}"
                    )
                if 'max' in rule and value > rule['max']:
                    errors.append(
                        f"Value for {key} exceeds maximum: {value} > {rule['max']}"
                    )
        
        return errors
    
    def get_rule(self, key: str) -> Optional[Dict[str, Any]]:
        """Get validation rule for configuration key"""
        return self.rules.get(key)
    
    def get_default(self, key: str) -> Any:
        """Get default value for configuration key"""
        rule = self.get_rule(key)
        return rule.get('default') if rule else None
    
    def get_all_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get all validation rules"""
        return self.rules.copy()
