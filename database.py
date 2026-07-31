"""
Database Interface

Provides high-level database operations for surveillance system.

Key Features:
=============

1. Connection Management
   - Connection pooling
   - Automatic reconnection
   - Transaction management

2. CRUD Operations
   - Store detections
   - Query results
   - Update alerts
   - Performance metrics

3. Time-Series Optimization
   - TimescaleDB integration
   - Efficient time-range queries
   - Automatic data retention

4. Error Handling
   - Graceful failures
   - Retry logic
   - Logging

Real-World Requirements:
========================

Defense systems need:
- High availability (no single point of failure)
- Data integrity (ACID compliance)
- Query performance (fast investigations)
- Scalability (millions of records)

This is production-grade database interface.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, and_, or_, desc, func
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from .models import (
    Base, Detection, Person, Vehicle, Animal, Object,
    Alert, Camera, PerformanceMetric, SystemLog,
    ThreatLevel, AlertStatus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """
    Main database interface for surveillance system.
    
    Handles all database operations with:
    - Connection pooling
    - Error handling
    - Transaction management
    - Query optimization
    """
    
    def __init__(
        self,
        database_url: str = "postgresql://sentinel:sentinel@localhost/sentinelai",
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False
    ):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection string
            pool_size: Number of connections in pool
            max_overflow: Maximum overflow connections
            echo: Echo SQL statements (for debugging)
        """
        self.database_url = database_url
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            echo=echo
        )
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        logger.info(f"Database initialized: {database_url}")
    
    def create_tables(self):
        """Create all tables in database."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            
            # Initialize TimescaleDB hypertables (if TimescaleDB extension exists)
            self._init_timescaledb()
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def _init_timescaledb(self):
        """
        Initialize TimescaleDB hypertables for time-series optimization.
        
        TimescaleDB is PostgreSQL extension for time-series data.
        Provides:
        - Automatic partitioning
        - Better query performance
        - Data retention policies
        - Compression
        """
        try:
            with self.engine.connect() as conn:
                # Check if TimescaleDB extension exists
                result = conn.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                    )
                """)
                
                if not result.scalar():
                    logger.info("TimescaleDB not installed, skipping hypertable creation")
                    return
                
                # Create hypertables for time-series tables
                hypertables = [
                    ('detections', 'timestamp'),
                    ('persons', 'timestamp'),
                    ('vehicles', 'timestamp'),
                    ('animals', 'timestamp'),
                    ('objects', 'timestamp'),
                    ('alerts', 'created_at'),
                    ('performance_metrics', 'timestamp'),
                    ('system_logs', 'timestamp')
                ]
                
                for table_name, time_column in hypertables:
                    try:
                        conn.execute(f"""
                            SELECT create_hypertable(
                                '{table_name}',
                                '{time_column}',
                                if_not_exists => TRUE
                            )
                        """)
                        logger.info(f"Created hypertable: {table_name}")
                    except Exception as e:
                        logger.warning(f"Could not create hypertable {table_name}: {e}")
                
                # Set up data retention policies (30 days for logs, 90 days for detections)
                retention_policies = [
                    ('system_logs', '30 days'),
                    ('performance_metrics', '30 days'),
                    ('detections', '90 days'),  # Adjust based on requirements
                ]
                
                for table_name, retention in retention_policies:
                    try:
                        conn.execute(f"""
                            SELECT add_retention_policy(
                                '{table_name}',
                                INTERVAL '{retention}',
                                if_not_exists => TRUE
                            )
                        """)
                        logger.info(f"Set retention policy for {table_name}: {retention}")
                    except Exception as e:
                        logger.warning(f"Could not set retention for {table_name}: {e}")
                
                logger.info("TimescaleDB hypertables initialized")
                
        except Exception as e:
            logger.warning(f"TimescaleDB initialization failed: {e}")
            # Continue without TimescaleDB (will work as regular PostgreSQL)
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around operations.
        
        Usage:
            with db.session_scope() as session:
                session.add(detection)
                session.commit()
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    # Camera Operations
    
    def add_camera(self, camera_id: str, name: str, **kwargs) -> Camera:
        """Add or update camera configuration."""
        with self.session_scope() as session:
            camera = session.query(Camera).filter_by(camera_id=camera_id).first()
            
            if camera:
                # Update existing
                camera.name = name
                for key, value in kwargs.items():
                    setattr(camera, key, value)
                camera.updated_at = datetime.now()
            else:
                # Create new
                camera = Camera(camera_id=camera_id, name=name, **kwargs)
                session.add(camera)
            
            session.commit()
            logger.info(f"Camera saved: {camera_id}")
            return camera
    
    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Get camera by ID."""
        with self.session_scope() as session:
            return session.query(Camera).filter_by(camera_id=camera_id).first()
    
    # Detection Operations
    
    def store_detection(self, detection_result) -> Detection:
        """
        Store complete detection result from orchestrator.
        
        Args:
            detection_result: DetectionResult from Day 7 orchestrator
            
        Returns:
            Stored Detection record
        """
        with self.session_scope() as session:
            # Create main detection record
            detection = Detection(
                camera_id=detection_result.camera_id,
                frame_number=detection_result.frame_number,
                timestamp=datetime.fromtimestamp(detection_result.timestamp),
                person_count=len(detection_result.persons),
                vehicle_count=len(detection_result.vehicles),
                animal_count=len(detection_result.animals),
                object_count=len(detection_result.objects),
                threat_score=detection_result.threat_score,
                threat_level=ThreatLevel(detection_result.threat_level.value),
                explanation=detection_result.explanation,
                persons_data=[self._serialize_person(p) for p in detection_result.persons],
                vehicles_data=[self._serialize_vehicle(v) for v in detection_result.vehicles],
                animals_data=[self._serialize_animal(a) for a in detection_result.animals],
                objects_data=[self._serialize_object(o) for o in detection_result.objects],
                correlations={
                    'person_vehicle': detection_result.person_vehicle_associations,
                    'person_object': detection_result.person_object_associations
                },
                processing_time_ms=detection_result.processing_time_ms,
                detector_times=detection_result.detector_times
            )
            
            session.add(detection)
            session.flush()  # Get detection.id
            
            # Store individual detections (for granular queries)
            timestamp = datetime.fromtimestamp(detection_result.timestamp)
            
            # Persons
            for person in detection_result.persons:
                person_record = Person(
                    detection_id=detection.id,
                    timestamp=timestamp,
                    camera_id=detection_result.camera_id,
                    bbox_x1=person.bbox[0],
                    bbox_y1=person.bbox[1],
                    bbox_x2=person.bbox[2],
                    bbox_y2=person.bbox[3],
                    confidence=person.confidence,
                    pose=getattr(person, 'pose', None)
                )
                session.add(person_record)
            
            # Vehicles
            for vehicle in detection_result.vehicles:
                vehicle_record = Vehicle(
                    detection_id=detection.id,
                    timestamp=timestamp,
                    camera_id=detection_result.camera_id,
                    bbox_x1=vehicle.bbox[0],
                    bbox_y1=vehicle.bbox[1],
                    bbox_x2=vehicle.bbox[2],
                    bbox_y2=vehicle.bbox[3],
                    confidence=vehicle.confidence,
                    vehicle_type=getattr(vehicle, 'vehicle_type', None),
                    size_category=getattr(vehicle, 'size', None)
                )
                session.add(vehicle_record)
            
            # Animals
            for animal in detection_result.animals:
                animal_record = Animal(
                    detection_id=detection.id,
                    timestamp=timestamp,
                    camera_id=detection_result.camera_id,
                    bbox_x1=animal.bbox[0],
                    bbox_y1=animal.bbox[1],
                    bbox_x2=animal.bbox[2],
                    bbox_y2=animal.bbox[3],
                    confidence=animal.confidence,
                    species=getattr(animal, 'species', None),
                    category=getattr(animal, 'category', None)
                )
                session.add(animal_record)
            
            # Objects
            for obj in detection_result.objects:
                object_record = Object(
                    detection_id=detection.id,
                    timestamp=timestamp,
                    camera_id=detection_result.camera_id,
                    bbox_x1=obj.bbox[0],
                    bbox_y1=obj.bbox[1],
                    bbox_x2=obj.bbox[2],
                    bbox_y2=obj.bbox[3],
                    confidence=obj.confidence,
                    object_type=getattr(obj, 'object_type', None),
                    size_category=getattr(obj, 'size', None)
                )
                session.add(object_record)
            
            # Store alerts
            for alert_data in detection_result.alerts:
                alert = Alert(
                    detection_id=detection.id,
                    camera_id=detection_result.camera_id,
                    alert_type=alert_data['type'],
                    priority=alert_data['priority'],
                    message=alert_data['message'],
                    details=alert_data.get('details', ''),
                    recommended_action=alert_data.get('action', ''),
                    threat_score=detection_result.threat_score,
                    threat_level=ThreatLevel(detection_result.threat_level.value),
                    status=AlertStatus.PENDING
                )
                session.add(alert)
            
            session.commit()
            logger.debug(f"Stored detection: {detection.id}")
            return detection
    
    def query_detections(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        threat_level: Optional[ThreatLevel] = None,
        min_threat_score: Optional[float] = None,
        has_persons: Optional[bool] = None,
        has_vehicles: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Detection]:
        """
        Query detections with filters.
        
        Common use cases:
        - Find all HIGH threats in last 24 hours
        - Get all detections from specific camera
        - Find detections with persons
        - Investigation queries
        """
        with self.session_scope() as session:
            query = session.query(Detection)
            
            # Apply filters
            if camera_id:
                query = query.filter(Detection.camera_id == camera_id)
            
            if start_time:
                query = query.filter(Detection.timestamp >= start_time)
            
            if end_time:
                query = query.filter(Detection.timestamp <= end_time)
            
            if threat_level:
                query = query.filter(Detection.threat_level == threat_level)
            
            if min_threat_score is not None:
                query = query.filter(Detection.threat_score >= min_threat_score)
            
            if has_persons is not None:
                if has_persons:
                    query = query.filter(Detection.person_count > 0)
                else:
                    query = query.filter(Detection.person_count == 0)
            
            if has_vehicles is not None:
                if has_vehicles:
                    query = query.filter(Detection.vehicle_count > 0)
                else:
                    query = query.filter(Detection.vehicle_count == 0)
            
            # Order by time (most recent first)
            query = query.order_by(desc(Detection.timestamp))
            
            # Pagination
            query = query.limit(limit).offset(offset)
            
            return query.all()
    
    # Alert Operations
    
    def get_pending_alerts(self, limit: int = 50) -> List[Alert]:
        """Get pending alerts (not yet acknowledged)."""
        with self.session_scope() as session:
            return session.query(Alert)\
                .filter(Alert.status == AlertStatus.PENDING)\
                .order_by(desc(Alert.priority), desc(Alert.created_at))\
                .limit(limit)\
                .all()
    
    def acknowledge_alert(self, alert_id: int, operator: str) -> Alert:
        """Mark alert as acknowledged by operator."""
        with self.session_scope() as session:
            alert = session.query(Alert).get(alert_id)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = operator
                alert.response_time_seconds = (
                    (alert.acknowledged_at - alert.created_at).total_seconds()
                )
                session.commit()
                logger.info(f"Alert {alert_id} acknowledged by {operator}")
            return alert
    
    def resolve_alert(
        self,
        alert_id: int,
        operator: str,
        resolution_notes: str,
        is_false_alarm: bool = False
    ) -> Alert:
        """Mark alert as resolved."""
        with self.session_scope() as session:
            alert = session.query(Alert).get(alert_id)
            if alert:
                alert.status = AlertStatus.FALSE_ALARM if is_false_alarm else AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.resolved_by = operator
                alert.resolution_notes = resolution_notes
                alert.resolution_time_seconds = (
                    (alert.resolved_at - alert.created_at).total_seconds()
                )
                session.commit()
                logger.info(f"Alert {alert_id} resolved by {operator}")
            return alert
    
    # Performance Metrics
    
    def store_metric(
        self,
        component: str,
        metric_type: str,
        value: float,
        unit: str = None,
        camera_id: str = None,
        additional_data: Dict = None
    ):
        """Store performance metric."""
        with self.session_scope() as session:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                component=component,
                metric_type=metric_type,
                value=value,
                unit=unit,
                camera_id=camera_id,
                additional_data=additional_data
            )
            session.add(metric)
            session.commit()
    
    def get_metrics(
        self,
        component: str = None,
        metric_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[PerformanceMetric]:
        """Query performance metrics."""
        with self.session_scope() as session:
            query = session.query(PerformanceMetric)
            
            if component:
                query = query.filter(PerformanceMetric.component == component)
            
            if metric_type:
                query = query.filter(PerformanceMetric.metric_type == metric_type)
            
            if start_time:
                query = query.filter(PerformanceMetric.timestamp >= start_time)
            
            if end_time:
                query = query.filter(PerformanceMetric.timestamp <= end_time)
            
            query = query.order_by(desc(PerformanceMetric.timestamp)).limit(limit)
            
            return query.all()
    
    # Statistics & Analytics
    
    def get_statistics(
        self,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict[str, Any]:
        """Get system statistics."""
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()
        
        with self.session_scope() as session:
            # Detection counts
            detection_count = session.query(func.count(Detection.id))\
                .filter(Detection.timestamp.between(start_time, end_time))\
                .scalar()
            
            # Threat distribution
            threat_dist = {}
            for level in ThreatLevel:
                count = session.query(func.count(Detection.id))\
                    .filter(
                        Detection.timestamp.between(start_time, end_time),
                        Detection.threat_level == level
                    ).scalar()
                threat_dist[level.value] = count
            
            # Alert statistics
            alert_count = session.query(func.count(Alert.id))\
                .filter(Alert.created_at.between(start_time, end_time))\
                .scalar()
            
            pending_alerts = session.query(func.count(Alert.id))\
                .filter(
                    Alert.created_at.between(start_time, end_time),
                    Alert.status == AlertStatus.PENDING
                ).scalar()
            
            # Average response time
            avg_response = session.query(func.avg(Alert.response_time_seconds))\
                .filter(
                    Alert.created_at.between(start_time, end_time),
                    Alert.response_time_seconds.isnot(None)
                ).scalar()
            
            # Camera activity
            camera_activity = session.query(
                Detection.camera_id,
                func.count(Detection.id).label('detection_count')
            ).filter(
                Detection.timestamp.between(start_time, end_time)
            ).group_by(Detection.camera_id).all()
            
            return {
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'detections': {
                    'total': detection_count,
                    'threat_distribution': threat_dist
                },
                'alerts': {
                    'total': alert_count,
                    'pending': pending_alerts,
                    'avg_response_time_seconds': float(avg_response) if avg_response else None
                },
                'cameras': {
                    cam_id: count for cam_id, count in camera_activity
                }
            }
    
    # Helper methods
    
    def _serialize_person(self, person) -> Dict:
        """Serialize person detection for JSON storage."""
        return {
            'bbox': person.bbox,
            'confidence': person.confidence,
            'pose': getattr(person, 'pose', None)
        }
    
    def _serialize_vehicle(self, vehicle) -> Dict:
        """Serialize vehicle detection for JSON storage."""
        return {
            'bbox': vehicle.bbox,
            'confidence': vehicle.confidence,
            'type': getattr(vehicle, 'vehicle_type', None),
            'size': getattr(vehicle, 'size', None)
        }
    
    def _serialize_animal(self, animal) -> Dict:
        """Serialize animal detection for JSON storage."""
        return {
            'bbox': animal.bbox,
            'confidence': animal.confidence,
            'species': getattr(animal, 'species', None),
            'category': getattr(animal, 'category', None)
        }
    
    def _serialize_object(self, obj) -> Dict:
        """Serialize object detection for JSON storage."""
        return {
            'bbox': obj.bbox,
            'confidence': obj.confidence,
            'type': getattr(obj, 'object_type', None),
            'size': getattr(obj, 'size', None)
        }
    
    def close(self):
        """Close database connection."""
        self.Session.remove()
        self.engine.dispose()
        logger.info("Database connection closed")


# Global database instance
_database = None


def init_database(database_url: str = None, **kwargs) -> Database:
    """Initialize global database instance."""
    global _database
    if database_url is None:
        database_url = "postgresql://sentinel:sentinel@localhost/sentinelai"
    _database = Database(database_url, **kwargs)
    _database.create_tables()
    return _database


def get_database() -> Database:
    """Get global database instance."""
    global _database
    if _database is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _database


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = init_database()
    
    print("Database initialized successfully!")
    print("Tables created:")
    print("- cameras")
    print("- detections")
    print("- persons")
    print("- vehicles")
    print("- animals")
    print("- objects")
    print("- alerts")
    print("- performance_metrics")
    print("- system_logs")
