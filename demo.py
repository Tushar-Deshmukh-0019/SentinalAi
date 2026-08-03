"""
Configuration Management System - Demo

Demonstrates:
1. Loading configuration from different environments
2. Accessing configuration values
3. Environment variable overrides
4. Runtime configuration changes
5. Configuration validation
6. Change notifications
7. Exporting schema
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.config import ConfigManager, ConfigValidator
from ai.logging import setup_logger

# Initialize logging
logger = setup_logger('config.demo')


def demo_basic_loading():
    """Demo 1: Basic configuration loading"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Configuration Loading")
    print("="*70)
    
    # Load development configuration (default)
    config = ConfigManager.load_config()
    
    logger.info(f"Environment: {config.get_environment()}")
    logger.info(f"Is Development: {config.is_development()}")
    logger.info(f"Is Production: {config.is_production()}")
    
    # Access configuration values
    person_threshold = config.get('detection.confidence_thresholds.person')
    db_host = config.get('database.host')
    log_level = config.get('logging.level')
    
    print(f"\nLoaded Configuration:")
    print(f"  Person Detection Threshold: {person_threshold}")
    print(f"  Database Host: {db_host}")
    print(f"  Logging Level: {log_level}")


def demo_environment_specific():
    """Demo 2: Environment-specific configuration"""
    print("\n" + "="*70)
    print("DEMO 2: Environment-Specific Configuration")
    print("="*70)
    
    environments = ['development', 'staging', 'production']
    settings_to_check = [
        'detection.model_size',
        'camera.max_cameras',
        'processing.mode',
        'database.host',
        'logging.level',
        'threat.critical_threshold'
    ]
    
    for env in environments:
        print(f"\n{env.upper()} Configuration:")
        config = ConfigManager.load_config(environment=env)
        
        for setting in settings_to_check:
            value = config.get(setting)
            print(f"  {setting}: {value}")


def demo_environment_variables():
    """Demo 3: Environment variable overrides"""
    print("\n" + "="*70)
    print("DEMO 3: Environment Variable Overrides")
    print("="*70)
    
    print("\nSetting environment variables...")
    os.environ['SENTINELAI_DETECTION_CONFIDENCE_THRESHOLDS_PERSON'] = '0.75'
    os.environ['SENTINELAI_DATABASE_HOST'] = 'prod-db.example.com'
    os.environ['SENTINELAI_LOGGING_LEVEL'] = 'DEBUG'
    os.environ['SENTINELAI_PROCESSING_MAX_WORKERS'] = '8'
    
    # Reload configuration with environment variables
    ConfigManager._instance = None  # Reset singleton
    config = ConfigManager.load_config()
    
    print("\nConfiguration after environment variable overrides:")
    print(f"  Person Threshold: {config.get('detection.confidence_thresholds.person')}")
    print(f"  Database Host: {config.get('database.host')}")
    print(f"  Logging Level: {config.get('logging.level')}")
    print(f"  Max Workers: {config.get('processing.max_workers')}")
    
    # Clean up
    del os.environ['SENTINELAI_DETECTION_CONFIDENCE_THRESHOLDS_PERSON']
    del os.environ['SENTINELAI_DATABASE_HOST']
    del os.environ['SENTINELAI_LOGGING_LEVEL']
    del os.environ['SENTINELAI_PROCESSING_MAX_WORKERS']


def demo_runtime_changes():
    """Demo 4: Runtime configuration changes"""
    print("\n" + "="*70)
    print("DEMO 4: Runtime Configuration Changes")
    print("="*70)
    
    ConfigManager._instance = None  # Reset singleton
    config = ConfigManager.load_config()
    
    print(f"\nOriginal threat threshold: {config.get('threat.critical_threshold')}")
    
    # Change configuration at runtime
    print("Changing threat threshold to 75...")
    config.set('threat.critical_threshold', 75)
    
    print(f"New threat threshold: {config.get('threat.critical_threshold')}")
    
    # Try to set invalid value (will raise error)
    print("\nTrying to set invalid threat threshold (150)...")
    try:
        config.set('threat.critical_threshold', 150)
        print("  ERROR: Should have raised ValidationError!")
    except Exception as e:
        print(f"  Correctly rejected: {e}")


def demo_change_notifications():
    """Demo 5: Configuration change notifications"""
    print("\n" + "="*70)
    print("DEMO 5: Configuration Change Notifications")
    print("="*70)
    
    ConfigManager._instance = None  # Reset singleton
    config = ConfigManager.load_config()
    
    # Define callback
    def on_log_level_change(key, old_value, new_value):
        print(f"\n  [CALLBACK] {key} changed: {old_value} → {new_value}")
        print(f"  [ACTION] Logger level would be updated to: {new_value}")
    
    # Subscribe to changes
    config.subscribe('logging.level', on_log_level_change)
    
    print("Subscribed to logging.level changes")
    print("Changing logging level to DEBUG...")
    config.set('logging.level', 'DEBUG')
    
    print("Changing logging level back to INFO...")
    config.set('logging.level', 'INFO')


def demo_validation():
    """Demo 6: Configuration validation"""
    print("\n" + "="*70)
    print("DEMO 6: Configuration Validation")
    print("="*70)
    
    validator = ConfigValidator()
    
    # Show validation rules for detection parameters
    print("\nValidation Rules for Detection Configuration:")
    
    rules_to_show = [
        'detection.confidence_thresholds.person',
        'detection.model_size',
        'detection.nms_threshold',
        'camera.max_cameras',
        'logging.level'
    ]
    
    for key in rules_to_show:
        rule = validator.get_rule(key)
        if rule:
            print(f"\n  {key}:")
            print(f"    Type: {rule.get('type', 'unknown')}")
            print(f"    Default: {rule.get('default')}")
            print(f"    Description: {rule.get('description', 'N/A')}")
            
            if 'enum' in rule:
                print(f"    Valid Values: {rule['enum']}")
            if 'min' in rule:
                print(f"    Range: [{rule.get('min')}, {rule.get('max')}]")


def demo_configuration_dict():
    """Demo 7: Get complete configuration as dictionary"""
    print("\n" + "="*70)
    print("DEMO 7: Configuration as Dictionary")
    print("="*70)
    
    ConfigManager._instance = None  # Reset singleton
    config = ConfigManager.load_config()
    
    # Get without sensitive values
    config_dict = config.to_dict(include_sensitive=False)
    
    print("\nTop-level configuration sections:")
    for key in sorted(config_dict.keys()):
        if isinstance(config_dict[key], dict):
            num_keys = len(config_dict[key])
            print(f"  {key}: {num_keys} subsections")


def demo_schema_export():
    """Demo 8: Export configuration schema"""
    print("\n" + "="*70)
    print("DEMO 8: Export Configuration Schema")
    print("="*70)
    
    config = ConfigManager.load_config()
    schema_file = 'config/schema.yaml'
    
    print(f"\nExporting configuration schema to {schema_file}...")
    config.export_schema(schema_file)
    
    # Show first few lines
    schema_path = Path(schema_file)
    if schema_path.exists():
        print(f"Schema file created successfully ({schema_path.stat().st_size} bytes)")
        print("\nFirst 20 lines of schema:")
        with open(schema_path) as f:
            for i, line in enumerate(f):
                if i < 20:
                    print(f"  {line.rstrip()}")
                else:
                    break


def demo_threat_scoring():
    """Demo 9: Threat scoring configuration"""
    print("\n" + "="*70)
    print("DEMO 9: Threat Scoring Configuration")
    print("="*70)
    
    ConfigManager._instance = None  # Reset singleton
    config = ConfigManager.load_config(environment='production')
    
    print("\nProduction Threat Configuration:")
    print(f"  Min Score: {config.get('threat.min_score')}")
    print(f"  Alert Threshold: {config.get('threat.alert_threshold')}")
    print(f"  Critical Threshold: {config.get('threat.critical_threshold')}")
    
    print("\nThreat Scoring Weights:")
    weights = config.get('threat.weights', {})
    for layer, weight in sorted(weights.items()):
        print(f"  {layer}: {weight}")


def demo_multi_environment_comparison():
    """Demo 10: Compare configurations across environments"""
    print("\n" + "="*70)
    print("DEMO 10: Multi-Environment Comparison")
    print("="*70)
    
    environments = ['development', 'staging', 'production']
    comparison_keys = [
        'detection.model_size',
        'camera.max_cameras',
        'buffer.critical_preserve_rate',
        'processing.mode',
        'logging.level',
        'threat.alert_threshold',
        'monitoring.max_latency_ms'
    ]
    
    print(f"\n{'Setting':<40} {'Dev':<15} {'Staging':<15} {'Prod':<15}")
    print("-" * 85)
    
    for key in comparison_keys:
        values = []
        for env in environments:
            ConfigManager._instance = None
            config = ConfigManager.load_config(environment=env)
            value = config.get(key)
            values.append(str(value)[:12])
        
        print(f"{key:<40} {values[0]:<15} {values[1]:<15} {values[2]:<15}")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🛡️  SENTINELAI - CONFIGURATION MANAGEMENT SYSTEM")
    print("Day 10: Comprehensive Configuration Management")
    print("="*70)
    
    try:
        demo_basic_loading()
        demo_environment_specific()
        demo_environment_variables()
        demo_runtime_changes()
        demo_change_notifications()
        demo_validation()
        demo_configuration_dict()
        demo_schema_export()
        demo_threat_scoring()
        demo_multi_environment_comparison()
        
        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
