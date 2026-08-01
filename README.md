# Logging & Audit System - Day 9

## Overview

Complete observability and auditability for SentinelAI.

## Quick Start

```python
# Configure logging
from ai.logging import configure_logging, get_logger

configure_logging(log_dir="logs")

# Get logger for your component
logger = get_logger("my_component")
logger.info("Component started")

# Audit logging
from ai.logging import get_audit_logger

audit = get_audit_logger()
audit.log_detection(
    camera_id="main_gate",
    detection_id=12345,
    person_count=1,
    threat_score=75.0
)
```

## Features

✅ Structured JSON logging  
✅ Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)  
✅ Audit trail for compliance  
✅ Database integration  
✅ Log rotation  
✅ Performance tracking  
✅ Error categorization  

See docs/DAY_9_SUMMARY.md for complete documentation.
