# 🔧 Configuration Management System

**Day 10: Comprehensive Configuration Management**

## Overview

The Configuration Management System provides centralized, environment-aware configuration for SentinelAI with:
- Multi-format support (YAML, JSON)
- Environment-specific overrides (development, staging, production)
- Environment variable overrides
- Runtime reconfiguration with validation
- Change notification callbacks
- Comprehensive validation against schema
- Singleton pattern for application-wide access

## Why This Matters

> **The Tactical Problem**: A surveillance system running on-site and in production cannot afford to:
> - Have developers hardcode settings
> - Require code changes for different deployments
> - Accidentally deploy development settings to production
> - Restart the entire system to apply configuration changes

**SentinelAI's Answer**: Single source of truth with environment-aware layering.

### Real-World Scenario

```
Day 1: Development
  - Person threshold: 0.45 (catch more potential threats)
  - 10 cameras max
  - Debug logging enabled

Day 50: Staging/Testing
  - Person threshold: 0.50 (balanced testing)
  - 20 cameras for load testing
  - Debug logging for components
  - Experimental features enabled

Day 100: Production Deployment
  - Person threshold: 0.55 (minimize false alarms for operators)
  - 50+ cameras supported
  - Production logging only
  - No experimental features
  - Robust database pooling

Result: Same code, different behavior per environment. No restarts needed.
```

## Configuration Architecture

### Loading Priority (Highest to Lowest)

1. **Environment Variables** (`SENTINELAI_*` prefix)
   ```bash
   export SENTINELAI_DATABASE_HOST=prod-db.example.com
   export SENTINELAI_DETECTION_CONFIDENCE_THRESHOLDS_PERSON=0.55
   ```

2. **Environment-Specific File** (`config/app.{environment}.yaml`)
   - `config/app.production.yaml`
   - `config/app.staging.yaml`
   - `config/app.development.yaml`

3. **Base Configuration** (`config/app.yaml`)
   - Default settings for all environments

4. **Validator Defaults**
   - Schema-defined defaults for missing keys

### Configuration Structure

```
config/
├── app.yaml                    # Base configuration (inherited by all)
├── app.development.yaml        # Development overrides
├── app.staging.yaml           # Staging overrides
├── app.production.yaml        # Production overrides
├── database.yaml              # Database-specific config (legacy)
└── schema.yaml               # Generated schema reference
```

## Usage

### 1. Basic Configuration Loading

```python
from ai.config import ConfigManager

# Load configuration (auto-detects environment)
config = ConfigManager.load_config()

# Load specific environment
config = ConfigManager.load_config(environment='production')

# Load with specific config file
config = ConfigManager.load_config(
    config_file='config/app.yaml',
    environment='staging'
)
```

### 2. Accessing Configuration Values

```python
# Get single value with default
person_threshold = config.get('detection.confidence_thresholds.person')

# Get with fallback default
db_host = config.get('database.host', 'localhost')

# Get entire configuration as dictionary
all_settings = config.to_dict()

# Get without sensitive values
safe_config = config.to_dict(include_sensitive=False)
```

### 3. Environment Variable Overrides

Environment variables override all file-based configuration:

```bash
# Set database host (takes precedence over app.yaml)
export SENTINELAI_DATABASE_HOST=prod-db.example.com

# Set numeric values (auto-parsed to int/float)
export SENTINELAI_CAMERA_MAX_CAMERAS=50
export SENTINELAI_DETECTION_CONFIDENCE_THRESHOLDS_PERSON=0.60

# Set boolean values
export SENTINELAI_DATABASE_ECHO=true

# Naming convention: SENTINELAI_{key_in_dot_notation_uppercase}
# detection.confidence_thresholds.person → SENTINELAI_DETECTION_CONFIDENCE_THRESHOLDS_PERSON
```

### 4. Runtime Configuration Changes

```python
# Change configuration at runtime
config.set('threat.critical_threshold', 75)

# Invalid values are rejected with validation
try:
    config.set('threat.critical_threshold', 150)  # Max is 100
except ValidationError as e:
    print(f"Invalid configuration: {e}")

# Changes can notify subscribers
config.set('detection.model_size', 'large')  # Subscribers notified
```

### 5. Change Notifications

Subscribe to configuration changes:

```python
def on_log_level_change(key, old_value, new_value):
    logger.info(f"Log level changed: {old_value} → {new_value}")
    # Update logger level dynamically
    logging.getLogger().setLevel(new_value)

config.subscribe('logging.level', on_log_level_change)

# When this happens:
config.set('logging.level', 'DEBUG')
# Callback is triggered with (key, old_value, new_value)
```

### 6. Validation and Schema

```python
from ai.config import ConfigValidator

# Get validator
validator = ConfigValidator()

# Get rule for a key
rule = validator.get_rule('detection.confidence_thresholds.person')
# Returns: {
#     'type': float,
#     'min': 0.0,
#     'max': 1.0,
#     'default': 0.45,
#     'description': '...'
# }

# Validate configuration
errors = validator.validate(config_dict)
if errors:
    for error in errors:
        print(f"Error: {error}")

# Export schema as reference
config.export_schema('config/schema.yaml')
```

## Configuration Sections

### Detection Configuration

```yaml
detection:
  model_size: medium  # nano, small, medium, large, xlarge
  confidence_thresholds:
    person: 0.45    # 0.0-1.0 (higher = fewer detections)
    vehicle: 0.50
    animal: 0.40    # Aggressive wildlife filtering
    object: 0.50
  nms_threshold: 0.45  # Non-Maximum Suppression
```

**Impact**: Controls detection sensitivity
- **Development**: Lower thresholds (catch more)
- **Production**: Higher thresholds (reduce false positives)

### Camera Configuration

```yaml
camera:
  max_cameras: 10
  buffer_size: 10
  reconnection_attempts: 5
  reconnection_delay: 5
```

**Impact**: Multi-camera handling
- **Dev**: 10 cameras for testing
- **Production**: 50+ cameras with larger buffers

### Frame Buffer & Priority

```yaml
buffer:
  critical_preserve_rate: 0.99   # 99% CRITICAL frames preserved
  minimal_preserve_rate: 0.44    # 44% MINIMAL frames preserved
```

**Impact**: Frame dropping under load
- Main gate = CRITICAL (99% preserved)
- Parking lot = MINIMAL (44% preserved)

### Processing Configuration

```yaml
processing:
  mode: sequential  # sequential or parallel
  max_workers: 4
  timeout: 30
```

**Impact**: Performance vs. Memory
- **Development**: Sequential (predictable, low memory)
- **Production**: Sequential (stable) or Parallel (fast)

### Database Configuration

```yaml
database:
  host: localhost
  port: 5432
  name: sentinelai
  user: postgres
  password: postgres  # USE ENV VAR IN PRODUCTION!
  pool_size: 10
  max_overflow: 20
```

**Impact**: Data storage and retrieval
- **Dev**: Local PostgreSQL
- **Production**: Remote managed database with robust pooling

### Logging Configuration

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: logs/sentinelai.log
  max_size: 10485760  # 10MB
  backup_count: 5
  enable_audit: true
```

**Impact**: Observability and debugging
- **Dev**: DEBUG level, all details logged
- **Production**: INFO level, only important events

### Threat Assessment

```yaml
threat:
  min_score: 0
  alert_threshold: 60     # MODERATE - notify operator
  critical_threshold: 85  # HIGH - immediate action
  weights:
    layer_0_detection: 20
    layer_3_behavior: 15
    layer_8_weapon: 15
```

**Impact**: Threat scoring and alerting
- Thresholds affect operator alert frequency
- Weights tune threat calculation

### Feature Flags

```yaml
features:
  abandoned_object_detection: true
  animal_filtering: true
  conflict_resolution: true
  performance_monitoring: true
```

**Impact**: Enable/disable features without code changes

## Environment Comparison

### Development

```yaml
# Maximize detection sensitivity
detection:
  model_size: medium
  confidence_thresholds:
    person: 0.45

# Limited cameras for local testing
camera:
  max_cameras: 10

# Detailed logging
logging:
  level: DEBUG

# Sensitive threat detection
threat:
  alert_threshold: 50
```

### Staging

```yaml
# Balanced settings for testing
detection:
  model_size: large
  confidence_thresholds:
    person: 0.50

# Load test capacity
camera:
  max_cameras: 20

# Debug logging by component
logging:
  level: DEBUG
  components:
    detection: DEBUG
```

### Production

```yaml
# High accuracy, minimize false alarms
detection:
  model_size: xlarge
  confidence_thresholds:
    person: 0.55

# Full capacity
camera:
  max_cameras: 50

# Production logging only
logging:
  level: INFO

# Conservative threat detection
threat:
  alert_threshold: 65
```

## Real-World Integration

### Integrating Configuration in Components

```python
# In detection orchestrator
from ai.config import ConfigManager

class DetectionOrchestrator:
    def __init__(self):
        self.config = ConfigManager.load_config()
        self.person_threshold = self.config.get(
            'detection.confidence_thresholds.person'
        )
        
        # Subscribe to threshold changes
        self.config.subscribe(
            'detection.confidence_thresholds.person',
            self._on_threshold_change
        )
    
    def _on_threshold_change(self, key, old, new):
        self.person_threshold = new
        logger.info(f"Threshold updated: {old} → {new}")
    
    def process_frame(self, frame):
        # Use current threshold
        detections = self.person_detector.detect(
            frame,
            confidence_threshold=self.person_threshold
        )
        return detections
```

### Multi-Environment Deployment

```bash
# Development (local)
python main.py
# Loads: config/app.yaml + environment defaults

# Staging (test server)
SENTINELAI_ENV=staging python main.py
# Loads: config/app.yaml → app.staging.yaml

# Production (deployment)
export SENTINELAI_ENV=production
export SENTINELAI_DATABASE_HOST=prod-db.internal
export SENTINELAI_DATABASE_PASSWORD=$DB_PASSWORD
python main.py
# Loads: app.yaml → app.production.yaml → env var overrides
```

## Validation Rules

The configuration system validates all values against defined rules:

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| `detection.confidence_thresholds.person` | float | 0.0-1.0 | 0.45 | Person detection threshold |
| `detection.model_size` | str | enum | medium | YOLOv8 model size |
| `camera.max_cameras` | int | 1-100 | 10 | Max simultaneous cameras |
| `processing.mode` | str | enum | sequential | Detection mode |
| `logging.level` | str | enum | INFO | Log level |
| `threat.alert_threshold` | int | 0-100 | 60 | Alert threshold |

View all rules: `config.export_schema('schema.yaml')`

## Best Practices

### 1. Environment Variables for Secrets

```bash
# ✅ GOOD: Use environment variables for sensitive data
export SENTINELAI_DATABASE_PASSWORD=$DB_PASSWORD

# ❌ BAD: Don't hardcode passwords
database:
  password: my_secret_password
```

### 2. Environment-Specific Files

```bash
# ✅ GOOD: Use app.{env}.yaml for environment-specific settings
config/
  ├── app.yaml              # Shared base
  ├── app.staging.yaml      # Staging overrides
  └── app.production.yaml   # Production overrides

# ❌ BAD: Copying entire config files
app.yaml
app_staging.yaml
app_production.yaml
```

### 3. Runtime Changes with Validation

```python
# ✅ GOOD: Changes are validated
config.set('threat.alert_threshold', 65)  # Validated

# ❌ BAD: Unchecked changes
self.alert_threshold = 200  # No validation!
```

### 4. Subscribe to Changes

```python
# ✅ GOOD: React to configuration changes
config.subscribe('database.host', reconnect_database)

# ❌ BAD: Require application restart for changes
# Hardcoded database connection at startup
```

## Troubleshooting

### Configuration Not Loading

```python
# Check what environment is being used
config = ConfigManager.load_config()
print(config.get_environment())  # Should be 'production' in prod

# Check if environment file exists
import os
env = os.getenv('SENTINELAI_ENV', 'development')
config_file = f'config/app.{env}.yaml'
print(os.path.exists(config_file))
```

### Validation Errors

```python
# Get detailed validation errors
try:
    config = ConfigManager.load_config()
except ValidationError as e:
    print(e)  # Shows which settings failed validation
```

### Environment Variable Overrides Not Working

```python
# Check variable naming (uppercase, dots to underscores)
# Wrong: SENTINELAI_detection_confidence_thresholds_person
# Right: SENTINELAI_DETECTION_CONFIDENCE_THRESHOLDS_PERSON

# Verify variable is set
import os
print(os.environ.get('SENTINELAI_DATABASE_HOST'))
```

## API Reference

### ConfigManager

```python
ConfigManager.load_config(
    config_file: Optional[str] = None,
    environment: Optional[str] = None
) -> ConfigManager
```

Load configuration with optional file and environment override.

```python
config.get(key: str, default: Any = None) -> Any
```

Get configuration value by dot-notation key.

```python
config.set(key: str, value: Any, notify: bool = True) -> None
```

Set configuration value with validation and change notification.

```python
config.subscribe(key: str, callback: Callable) -> None
```

Subscribe to configuration changes.

```python
config.to_dict(include_sensitive: bool = False) -> Dict[str, Any]
```

Export configuration as dictionary (optionally excluding sensitive values).

### ConfigValidator

```python
validator = ConfigValidator()
errors = validator.validate(config_dict)
```

Validate configuration dictionary against schema.

```python
rule = validator.get_rule(key)
default = validator.get_default(key)
```

Inspect validation rules and defaults.

## Performance Impact

Configuration loading:
- **Startup**: ~50ms (YAML parsing)
- **Runtime get**: <1ms (dictionary lookup)
- **Runtime set**: <1ms (validation + notification)

Negligible impact on real-time processing.

## Security Considerations

1. **Sensitive Data**: Store passwords in environment variables, not files
2. **File Permissions**: Restrict read access to `config/` directory
3. **Audit Trail**: All changes logged to audit log
4. **Production**: Use secrets management (HashiCorp Vault, AWS Secrets Manager, etc.)

## Summary

Day 10 completes **Phase 1** with a production-grade configuration system:

✅ **What We Built**:
- Centralized configuration management
- Multi-environment support (dev/staging/prod)
- Environment variable overrides
- Runtime reconfiguration with validation
- Change notification system
- Comprehensive validation schema
- Zero-downtime configuration changes

✅ **Real-World Ready**:
- Same code runs in all environments
- Sensitive data protected
- Operators can reconfigure without restart
- Complete audit trail of changes
- Integrates with all system components

✅ **Phase 1 Complete**: 
- Days 1-4: Detection Core (person, vehicle, animal, object)
- Days 5-7: Infrastructure (camera manager, priority queue, orchestrator)
- Days 8-9: Storage & Logging (database, audit trail)
- Day 10: Configuration Management ✓

**Progress**: 10.8% (9/85 modules) → **12%** (10/85 modules after Day 10)
