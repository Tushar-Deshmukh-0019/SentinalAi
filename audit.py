"""
Audit Logger

Tracks all security-relevant events for compliance.

Features:
- Immutable audit trail
- Database integration (Day 8)
- Compliance reporting
- Investigation support
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import json

from ..storage import get_database


class AuditLevel(str, Enum):
    """Audit event levels."""
    INFO = "INFO"              # General audit event
    SECURITY = "SECURITY"      # Security-relevant
    ALERT = "ALERT"            # Alert generated
    RESPONSE = "RESPONSE"      # Operator response
    ERROR = "ERROR"            # Error in system
    COMPLIANCE = "COMPLIANCE"  # Compliance-related


class AuditEvent:
    """Audit event record."""
    
    def __init__(
        self,
        level: AuditLevel,
        event_type: str,
        description: str,
        user: Optional[str] = None,
        camera_id: Optional[str] = None,
        detection_id: Optional[int] = None,
        alert_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize audit event."""
        self.level = level
        self.event_type = event_type
        self.description = description
        self.user = user
        self.camera_id = camera_id
        self.detection_id = detection_id
        self.alert_id = alert_id
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'event_type': self.event_type,
            'description': self.description,
            'user': self.user,
            'camera_id': self.camera_id,
            'detection_id': self.detection_id,
            'alert_id': self.alert_id,
            'details': self.details
        }
    
    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Centralized audit logging.
    
    All security-relevant events are logged:
    - Detection events
    - Alert generation
    - Operator responses
    - System errors
    - Configuration changes
    """
    
    def __init__(self, store_to_db: bool = True):
        """Initialize audit logger."""
        self.logger = logging.getLogger('sentinelai.audit')
        self.store_to_db = store_to_db
        self.db = None
        
        if store_to_db:
            try:
                self.db = get_database()
            except RuntimeError:
                self.logger.warning("Database not initialized, logging to file only")
    
    def log_event(self, event: AuditEvent):
        """
        Log audit event.
        
        Args:
            event: Audit event to log
        """
        # Log to file
        self.logger.info(f"{event.level.value}: {event.description}", extra={'extra': event.to_dict()})
        
        # Log to database if available
        if self.store_to_db and self.db:
            try:
                self.db.store_metric(
                    component='audit',
                    metric_type=event.event_type,
                    value=1,  # Count
                    additional_data=event.to_dict()
                )
            except Exception as e:
                self.logger.error(f"Failed to store audit event in database: {e}")
    
    # Detection Events
    
    def log_detection(
        self,
        camera_id: str,
        detection_id: int,
        person_count: int = 0,
        vehicle_count: int = 0,
        threat_score: float = 0.0,
        threat_level: str = "NONE"
    ):
        """Log detection event."""
        event = AuditEvent(
            level=AuditLevel.INFO,
            event_type='detection',
            description=f"Detection on {camera_id}: {person_count} persons, {vehicle_count} vehicles",
            camera_id=camera_id,
            detection_id=detection_id,
            details={
                'person_count': person_count,
                'vehicle_count': vehicle_count,
                'threat_score': threat_score,
                'threat_level': threat_level
            }
        )
        self.log_event(event)
    
    # Alert Events
    
    def log_alert_generated(
        self,
        alert_id: int,
        camera_id: str,
        detection_id: int,
        threat_level: str,
        priority: int
    ):
        """Log alert generation."""
        event = AuditEvent(
            level=AuditLevel.ALERT,
            event_type='alert_generated',
            description=f"Alert {threat_level} priority {priority} on {camera_id}",
            camera_id=camera_id,
            detection_id=detection_id,
            alert_id=alert_id,
            details={
                'threat_level': threat_level,
                'priority': priority
            }
        )
        self.log_event(event)
    
    def log_alert_acknowledged(
        self,
        alert_id: int,
        user: str,
        response_time_seconds: float
    ):
        """Log operator acknowledgment."""
        event = AuditEvent(
            level=AuditLevel.RESPONSE,
            event_type='alert_acknowledged',
            description=f"Alert acknowledged by {user} ({response_time_seconds:.1f}s response time)",
            user=user,
            alert_id=alert_id,
            details={
                'response_time_seconds': response_time_seconds
            }
        )
        self.log_event(event)
    
    def log_alert_resolved(
        self,
        alert_id: int,
        user: str,
        resolution: str,
        resolution_notes: str
    ):
        """Log alert resolution."""
        event = AuditEvent(
            level=AuditLevel.RESPONSE,
            event_type='alert_resolved',
            description=f"Alert resolved as {resolution} by {user}",
            user=user,
            alert_id=alert_id,
            details={
                'resolution': resolution,
                'notes': resolution_notes
            }
        )
        self.log_event(event)
    
    # Security Events
    
    def log_conflict_resolution(
        self,
        camera_id: str,
        detection_id: int,
        conflict_type: str,
        resolution: str
    ):
        """Log conflict resolution (e.g., person vs. animal)."""
        event = AuditEvent(
            level=AuditLevel.SECURITY,
            event_type='conflict_resolved',
            description=f"Conflict {conflict_type} resolved: {resolution}",
            camera_id=camera_id,
            detection_id=detection_id,
            details={
                'conflict_type': conflict_type,
                'resolution': resolution
            }
        )
        self.log_event(event)
    
    def log_false_positive(
        self,
        alert_id: int,
        original_threat_level: str,
        root_cause: str
    ):
        """Log false positive detection."""
        event = AuditEvent(
            level=AuditLevel.SECURITY,
            event_type='false_positive',
            description=f"False positive detected: {original_threat_level} threat, root cause: {root_cause}",
            alert_id=alert_id,
            details={
                'original_threat_level': original_threat_level,
                'root_cause': root_cause
            }
        )
        self.log_event(event)
    
    # Error Events
    
    def log_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        camera_id: Optional[str] = None,
        recovery_action: Optional[str] = None
    ):
        """Log system error."""
        event = AuditEvent(
            level=AuditLevel.ERROR,
            event_type=f'{component}_error',
            description=f"{error_type}: {error_message}",
            camera_id=camera_id,
            details={
                'error_type': error_type,
                'recovery_action': recovery_action
            }
        )
        self.log_event(event)
    
    # Performance Events
    
    def log_performance_degradation(
        self,
        component: str,
        metric: str,
        threshold: float,
        current_value: float
    ):
        """Log performance degradation."""
        event = AuditEvent(
            level=AuditLevel.COMPLIANCE,
            event_type='performance_degradation',
            description=f"{component} {metric} degraded: {current_value} (threshold: {threshold})",
            details={
                'component': component,
                'metric': metric,
                'threshold': threshold,
                'current_value': current_value
            }
        )
        self.log_event(event)
    
    # Configuration Events
    
    def log_config_change(
        self,
        component: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user: Optional[str] = None
    ):
        """Log configuration change."""
        event = AuditEvent(
            level=AuditLevel.COMPLIANCE,
            event_type='config_changed',
            description=f"{component}.{config_key} changed",
            user=user,
            details={
                'component': component,
                'config_key': config_key,
                'old_value': str(old_value),
                'new_value': str(new_value)
            }
        )
        self.log_event(event)
    
    # Compliance Events
    
    def log_data_access(
        self,
        user: str,
        resource_type: str,
        resource_id: Any,
        access_type: str = "READ"
    ):
        """Log data access for compliance."""
        event = AuditEvent(
            level=AuditLevel.COMPLIANCE,
            event_type='data_access',
            description=f"{user} {access_type} {resource_type} {resource_id}",
            user=user,
            details={
                'resource_type': resource_type,
                'resource_id': str(resource_id),
                'access_type': access_type
            }
        )
        self.log_event(event)


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def log_detection(
    camera_id: str,
    detection_id: int,
    **kwargs
):
    """Convenience function to log detection."""
    get_audit_logger().log_detection(camera_id, detection_id, **kwargs)


def log_alert_generated(alert_id: int, **kwargs):
    """Convenience function to log alert generation."""
    get_audit_logger().log_alert_generated(alert_id, **kwargs)


def log_alert_acknowledged(alert_id: int, **kwargs):
    """Convenience function to log acknowledgment."""
    get_audit_logger().log_alert_acknowledged(alert_id, **kwargs)


def log_alert_resolved(alert_id: int, **kwargs):
    """Convenience function to log resolution."""
    get_audit_logger().log_alert_resolved(alert_id, **kwargs)
