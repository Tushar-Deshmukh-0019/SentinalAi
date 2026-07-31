"""
Database Models

SQLAlchemy models for surveillance system data.

Schema Design Philosophy:
==========================

1. Time-Series Optimized
   - All tables have timestamps
   - Indexed for time-based queries
   - Compatible with TimescaleDB hypertables

2. Denormalized Where Appropriate
   - Store complete detection results
   - Fast queries over storage efficiency
   - Surveillance data is append-only

3. Audit Trail Complete
   - Every detection recorded
   - Every alert logged
   - Every action tracked

4. Compliance Ready
   - Data retention policies
   - Privacy controls
   - Access logging

Real-World Requirements:
========================

Indian Army / Defense Use Case:
- Store ALL detections (evidence)
- Maintain audit trail (regulations)
- Support investigations (queries)
- Enable pattern analysis (intelligence)
- Preserve for years (compliance)

This is production-grade schema design.
"""

from sqlalchemy import (
    Column, Integer, BigInteger, Float, String, Text,
    Boolean, DateTime, JSON, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import json

Base = declarative_base()


# Enums

class ThreatLevel(str, Enum):
    """Threat level classification."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert handling status."""
    PENDING = "pending"           # Alert generated, not yet viewed
    ACKNOWLEDGED = "acknowledged" # Operator saw alert
    INVESTIGATING = "investigating" # Under review
    RESOLVED = "resolved"         # Threat handled
    FALSE_ALARM = "false_alarm"   # No threat after review


# Core Models

class Camera(Base):
    """
    Camera configuration and metadata.
    
    Stores camera information for cross-referencing detections.
    """
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String(50), unique=True, nullable=False, index=True)
    """Unique camera identifier (e.g., 'main_gate', 'perimeter_01')."""
    
    name = Column(String(100), nullable=False)
    """Human-readable camera name."""
    
    location = Column(String(200))
    """Physical location description."""
    
    latitude = Column(Float)
    longitude = Column(Float)
    """GPS coordinates (if available)."""
    
    priority = Column(Integer, default=5)
    """Processing priority (1-10)."""
    
    source = Column(String(500))
    """Camera source (RTSP URL, device index, etc.)."""
    
    is_active = Column(Boolean, default=True)
    """Whether camera is currently active."""
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    detections = relationship("Detection", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="camera", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Camera(camera_id='{self.camera_id}', name='{self.name}')>"


class Detection(Base):
    """
    Complete detection result from orchestrator (Day 7).
    
    Stores ALL information produced by detection pipeline:
    - All entity detections (persons, vehicles, animals, objects)
    - Correlations and relationships
    - Threat assessment
    - Performance metrics
    
    This is the PRIMARY intelligence record.
    """
    __tablename__ = 'detections'
    
    id = Column(BigInteger, primary_key=True)
    """Unique detection ID."""
    
    # Source information
    camera_id = Column(String(50), ForeignKey('cameras.camera_id'), nullable=False, index=True)
    frame_number = Column(BigInteger, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    """Frame capture timestamp - CRITICAL for time-series queries."""
    
    # Detection counts (for quick filtering)
    person_count = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    animal_count = Column(Integer, default=0)
    object_count = Column(Integer, default=0)
    
    # Threat assessment
    threat_score = Column(Float, default=0.0, index=True)
    """Threat score (0-100)."""
    
    threat_level = Column(SQLEnum(ThreatLevel), default=ThreatLevel.NONE, index=True)
    """Classified threat level."""
    
    explanation = Column(Text)
    """Human-readable explanation of threat assessment."""
    
    # Complete detection data (JSON for flexibility)
    persons_data = Column(JSON)
    """Array of person detections with all attributes."""
    
    vehicles_data = Column(JSON)
    """Array of vehicle detections with all attributes."""
    
    animals_data = Column(JSON)
    """Array of animal detections with all attributes."""
    
    objects_data = Column(JSON)
    """Array of object detections with all attributes."""
    
    correlations = Column(JSON)
    """Person-vehicle and person-object associations."""
    
    # Performance metrics
    processing_time_ms = Column(Float)
    """Total processing time in milliseconds."""
    
    detector_times = Column(JSON)
    """Individual detector timings."""
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    camera = relationship("Camera", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_detection_time_camera', 'timestamp', 'camera_id'),
        Index('idx_detection_threat', 'threat_level', 'timestamp'),
        Index('idx_detection_time_desc', 'timestamp desc'),
    )
    
    def __repr__(self):
        return f"<Detection(id={self.id}, camera='{self.camera_id}', threat={self.threat_level.value})>"


class Person(Base):
    """
    Individual person detection record.
    
    While Detection stores complete results, this table stores
    individual person detections for granular queries.
    
    Use cases:
    - Track specific individuals across frames
    - Analyze person movement patterns
    - Generate person-centric reports
    """
    __tablename__ = 'persons'
    
    id = Column(BigInteger, primary_key=True)
    detection_id = Column(BigInteger, ForeignKey('detections.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    
    # Detection details
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    """Bounding box coordinates."""
    
    confidence = Column(Float, nullable=False)
    """Detection confidence (0-1)."""
    
    # Attributes (from Day 1)
    pose = Column(String(50))
    """Body pose (standing, sitting, lying, etc.)."""
    
    # Tracking ID (will be populated by tracking system - Days 13-22)
    track_id = Column(String(100), index=True)
    """Persistent tracking ID across frames."""
    
    # Associations
    vehicle_id = Column(BigInteger, ForeignKey('vehicles.id'))
    """Associated vehicle (if person in/near vehicle)."""
    
    object_ids = Column(JSON)
    """Associated objects (if person carrying objects)."""
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_person_time_camera', 'timestamp', 'camera_id'),
        Index('idx_person_track', 'track_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Person(id={self.id}, confidence={self.confidence:.2f})>"


class Vehicle(Base):
    """
    Individual vehicle detection record.
    
    Stores vehicle-specific attributes and associations.
    """
    __tablename__ = 'vehicles'
    
    id = Column(BigInteger, primary_key=True)
    detection_id = Column(BigInteger, ForeignKey('detections.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    
    # Detection details
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    confidence = Column(Float, nullable=False)
    
    # Vehicle attributes (from Day 2)
    vehicle_type = Column(String(50))
    """Type: car, truck, motorcycle, bus, etc."""
    
    size_category = Column(String(20))
    """Size: small, medium, large."""
    
    color = Column(String(50))
    """Vehicle color (if detected)."""
    
    # License plate (region detection from Day 2)
    has_plate = Column(Boolean, default=False)
    plate_bbox = Column(JSON)
    """License plate bounding box (if detected)."""
    
    # Tracking
    track_id = Column(String(100), index=True)
    
    # Authorization (Day 58)
    is_authorized = Column(Boolean)
    """Whether vehicle is authorized (will be populated by Day 58)."""
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_vehicle_time_camera', 'timestamp', 'camera_id'),
        Index('idx_vehicle_type', 'vehicle_type', 'timestamp'),
        Index('idx_vehicle_track', 'track_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Vehicle(id={self.id}, type='{self.vehicle_type}')>"


class Animal(Base):
    """
    Individual animal detection record.
    
    Important for:
    - Wildlife activity logging
    - False positive analysis
    - Conflict resolution auditing
    """
    __tablename__ = 'animals'
    
    id = Column(BigInteger, primary_key=True)
    detection_id = Column(BigInteger, ForeignKey('detections.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    
    # Detection details
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    confidence = Column(Float, nullable=False)
    
    # Animal attributes (from Day 3)
    species = Column(String(50), index=True)
    """Animal species: deer, dog, bear, etc."""
    
    category = Column(String(20))
    """Category: wildlife, domestic, etc."""
    
    threat_level = Column(String(20))
    """Animal-specific threat: none, low, moderate, high."""
    
    # Conflict resolution
    resolved_person_conflict = Column(Boolean, default=False)
    """Whether this animal resolved a person detection conflict."""
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_animal_time_camera', 'timestamp', 'camera_id'),
        Index('idx_animal_species', 'species', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Animal(id={self.id}, species='{self.species}')>"


class Object(Base):
    """
    Individual object detection record.
    
    Tracks security-critical objects (backpacks, bags, etc.).
    """
    __tablename__ = 'objects'
    
    id = Column(BigInteger, primary_key=True)
    detection_id = Column(BigInteger, ForeignKey('detections.id'), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    
    # Detection details
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    confidence = Column(Float, nullable=False)
    
    # Object attributes (from Day 4)
    object_type = Column(String(50), index=True)
    """Type: backpack, handbag, suitcase, etc."""
    
    size_category = Column(String(20))
    """Size: small, medium, large."""
    
    # Ownership tracking
    owner_person_id = Column(BigInteger, ForeignKey('persons.id'))
    """Associated person (owner)."""
    
    is_abandoned = Column(Boolean, default=False, index=True)
    """Whether object is abandoned (Day 4 feature)."""
    
    abandoned_duration_seconds = Column(Float)
    """How long object has been abandoned."""
    
    # Track across frames
    track_id = Column(String(100), index=True)
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_object_time_camera', 'timestamp', 'camera_id'),
        Index('idx_object_type', 'object_type', 'timestamp'),
        Index('idx_object_abandoned', 'is_abandoned', 'timestamp'),
        Index('idx_object_track', 'track_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Object(id={self.id}, type='{self.object_type}')>"


class Alert(Base):
    """
    Alert records generated by orchestrator (Day 7).
    
    Critical for:
    - Operator notification
    - Response tracking
    - Audit trail
    - Performance analysis
    
    Every alert must be tracked from generation to resolution.
    """
    __tablename__ = 'alerts'
    
    id = Column(BigInteger, primary_key=True)
    detection_id = Column(BigInteger, ForeignKey('detections.id'), nullable=False, index=True)
    camera_id = Column(String(50), ForeignKey('cameras.camera_id'), nullable=False, index=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False, index=True)
    """Type: critical_threat, high_threat, moderate_activity, etc."""
    
    priority = Column(Integer, nullable=False, index=True)
    """Priority (1-10, higher = more urgent)."""
    
    message = Column(Text, nullable=False)
    """Alert message for operator."""
    
    details = Column(Text)
    """Detailed explanation."""
    
    recommended_action = Column(Text)
    """Recommended response action."""
    
    # Threat context
    threat_score = Column(Float)
    threat_level = Column(SQLEnum(ThreatLevel))
    
    # Alert lifecycle
    status = Column(SQLEnum(AlertStatus), default=AlertStatus.PENDING, index=True)
    """Current alert status."""
    
    created_at = Column(DateTime, server_default=func.now(), index=True)
    """When alert was generated."""
    
    acknowledged_at = Column(DateTime)
    """When operator saw alert."""
    
    resolved_at = Column(DateTime)
    """When alert was resolved."""
    
    # Operator actions
    acknowledged_by = Column(String(100))
    """Username of operator who acknowledged."""
    
    resolved_by = Column(String(100))
    """Username of operator who resolved."""
    
    resolution_notes = Column(Text)
    """Notes about how alert was resolved."""
    
    # Response metrics
    response_time_seconds = Column(Float)
    """Time from alert to acknowledgment."""
    
    resolution_time_seconds = Column(Float)
    """Time from alert to resolution."""
    
    # Relationships
    detection = relationship("Detection", back_populates="alerts")
    camera = relationship("Camera", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alert_time_priority', 'created_at', 'priority'),
        Index('idx_alert_status_time', 'status', 'created_at'),
        Index('idx_alert_camera_time', 'camera_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Alert(id={self.id}, type='{self.alert_type}', status='{self.status.value}')>"


class PerformanceMetric(Base):
    """
    System performance metrics.
    
    Tracks:
    - Processing times
    - Throughput
    - Resource usage
    - System health
    
    Used for:
    - Performance optimization
    - Capacity planning
    - Anomaly detection
    - SLA monitoring
    """
    __tablename__ = 'performance_metrics'
    
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Component identification
    component = Column(String(50), nullable=False, index=True)
    """Component: camera_manager, frame_buffer, orchestrator, etc."""
    
    metric_type = Column(String(50), nullable=False, index=True)
    """Metric type: processing_time, throughput, memory_usage, etc."""
    
    # Metric values
    value = Column(Float, nullable=False)
    """Numeric metric value."""
    
    unit = Column(String(20))
    """Unit: ms, fps, mb, percent, etc."""
    
    # Context
    camera_id = Column(String(50), index=True)
    """Related camera (if applicable)."""
    
    additional_data = Column(JSON)
    """Additional context data."""
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_metric_time_component', 'timestamp', 'component'),
        Index('idx_metric_type_time', 'metric_type', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetric(component='{self.component}', type='{self.metric_type}', value={self.value})>"


# Utility Models

class SystemLog(Base):
    """
    System event logs (Day 9).
    
    Comprehensive logging for debugging and audit.
    """
    __tablename__ = 'system_logs'
    
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    level = Column(String(20), nullable=False, index=True)
    """Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    
    component = Column(String(50), nullable=False, index=True)
    """Source component."""
    
    message = Column(Text, nullable=False)
    """Log message."""
    
    details = Column(JSON)
    """Additional details."""
    
    exception = Column(Text)
    """Exception traceback (if error)."""
    
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_log_time_level', 'timestamp', 'level'),
        Index('idx_log_component_time', 'component', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemLog(level='{self.level}', component='{self.component}')>"


# Schema metadata
def get_schema_version():
    """Get current schema version."""
    return "1.0.0"


def get_schema_description():
    """Get schema description."""
    return """
    SentinelAI Database Schema v1.0.0
    
    Complete persistence layer for surveillance intelligence system.
    
    Core Tables:
    - cameras: Camera configuration
    - detections: Complete detection results
    - persons: Individual person detections
    - vehicles: Individual vehicle detections
    - animals: Individual animal detections
    - objects: Individual object detections
    - alerts: Generated alerts and responses
    - performance_metrics: System performance data
    - system_logs: Event logs (Day 9)
    
    Optimizations:
    - TimescaleDB hypertables for time-series
    - Indexes for common query patterns
    - JSON columns for flexibility
    - Denormalized for query speed
    
    Designed for:
    - Real-time surveillance
    - Historical analysis
    - Compliance requirements
    - Performance monitoring
    """
