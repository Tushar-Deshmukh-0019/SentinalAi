# Storage Module - Day 8

## Overview

Complete database persistence layer for SentinelAI surveillance system.

**What This Solves:**

Days 1-7 produced real-time intelligence but stored NOTHING permanently.
- System restart = ALL data lost ❌
- No history = No investigations ❌
- No audit trail = No compliance ❌

Day 8 adds permanent storage with PostgreSQL + TimescaleDB.
- All detections stored permanently ✓
- Complete audit trail ✓
- Historical queries enabled ✓
- Compliance ready ✓

## Database Schema

### Core Tables

```
cameras
├─ Camera configuration
├─ Location metadata
└─ Priority settings

detections (PRIMARY intelligence table)
├─ Complete detection results
├─ Threat assessments
├─ All entity counts
└─ Performance metrics

persons
├─ Individual person detections
├─ Bounding boxes
├─ Tracking IDs (future)
└─ Associations

vehicles
├─ Individual vehicle detections
├─ Type and size classification
├─ License plate detection
└─ Authorization status (future)

animals
├─ Wildlife detections
├─ Species classification
└─ Conflict resolution data

objects
├─ Security-critical objects
├─ Ownership tracking
└─ Abandoned detection

alerts
├─ Generated alerts
├─ Operator responses
├─ Resolution tracking
└─ Performance metrics

performance_metrics
├─ System performance data
├─ Component timings
└─ Resource usage

system_logs (Day 9)
├─ Event logs
├─ Error tracking
└─ Audit trail
```

### Entity Relationships

```
Camera
  ├──< Detection (many)
  │      ├──< Person (many)
  │      ├──< Vehicle (many)
  │      ├──< Animal (many)
  │      ├──< Object (many)
  │      └──< Alert (many)
  └──< Alert (many)
```

## Installation

### 1. Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Windows:**
Download from https://www.postgresql.org/download/windows/

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

### 2. Install TimescaleDB (Optional but Recommended)

TimescaleDB provides time-series optimizations.

**Ubuntu/Debian:**
```bash
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt update
sudo apt install timescaledb-postgresql-14
sudo timescaledb-tune
```

**See**: https://docs.timescale.com/install/latest/

### 3. Create Database

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE sentinelai;
CREATE USER sentinel WITH PASSWORD 'sentinel';
GRANT ALL PRIVILEGES ON DATABASE sentinelai TO sentinel;

# Enable TimescaleDB extension (if installed)
\c sentinelai
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Exit
\q
```

### 4. Install Python Dependencies

```bash
pip install psycopg2-binary sqlalchemy alembic
```

### 5. Initialize Schema

```python
from ai.storage import init_database

# Initialize database (creates all tables)
db = init_database("postgresql://sentinel:sentinel@localhost/sentinelai")
```

## Usage

### Basic Operations

```python
from ai.storage import Database, init_database
from datetime import datetime, timedelta

# Initialize
db = init_database()

# Add camera
db.add_camera(
    camera_id="main_gate",
    name="Main Gate Camera",
    location="North Entrance",
    priority=10,
    latitude=28.6139,
    longitude=77.2090
)

# Store detection result (from Day 7 orchestrator)
result = orchestrator.process_frame(frame)
detection = db.store_detection(result)

print(f"Stored detection {detection.id}")
print(f"Threat: {detection.threat_level.value}")
print(f"Score: {detection.threat_score}/100")
```

### Querying Detections

```python
# Get all HIGH threats from last 24 hours
from ai.storage.models import ThreatLevel

high_threats = db.query_detections(
    start_time=datetime.now() - timedelta(hours=24),
    threat_level=ThreatLevel.HIGH
)

for detection in high_threats:
    print(f"[{detection.timestamp}] {detection.camera_id}")
    print(f"  Threat: {detection.threat_score}/100")
    print(f"  Persons: {detection.person_count}")
    print(f"  Explanation: {detection.explanation}")

# Get detections from specific camera
main_gate_detections = db.query_detections(
    camera_id="main_gate",
    start_time=datetime.now() - timedelta(hours=1),
    limit=50
)

# Find detections with persons and high threat
suspicious = db.query_detections(
    has_persons=True,
    min_threat_score=70.0,
    start_time=datetime.now() - timedelta(days=7)
)
```

### Alert Management

```python
# Get pending alerts
pending = db.get_pending_alerts(limit=10)

for alert in pending:
    print(f"Alert {alert.id}: {alert.message}")
    print(f"  Priority: {alert.priority}")
    print(f"  Camera: {alert.camera_id}")
    print(f"  Created: {alert.created_at}")

# Acknowledge alert
db.acknowledge_alert(
    alert_id=alert.id,
    operator="operator_smith"
)

# Resolve alert
db.resolve_alert(
    alert_id=alert.id,
    operator="operator_smith",
    resolution_notes="Security team dispatched, area secured",
    is_false_alarm=False
)
```

### Performance Metrics

```python
# Store metric
db.store_metric(
    component="orchestrator",
    metric_type="processing_time",
    value=52.3,
    unit="ms",
    camera_id="main_gate"
)

# Query metrics
metrics = db.get_metrics(
    component="orchestrator",
    metric_type="processing_time",
    start_time=datetime.now() - timedelta(hours=1)
)

avg_time = sum(m.value for m in metrics) / len(metrics)
print(f"Average processing time: {avg_time:.1f}ms")
```

### Statistics

```python
# Get system statistics
stats = db.get_statistics(
    start_time=datetime.now() - timedelta(hours=24)
)

print(f"Detections: {stats['detections']['total']}")
print(f"Threat Distribution:")
for level, count in stats['detections']['threat_distribution'].items():
    print(f"  {level}: {count}")

print(f"\nAlerts:")
print(f"  Total: {stats['alerts']['total']}")
print(f"  Pending: {stats['alerts']['pending']}")
print(f"  Avg Response: {stats['alerts']['avg_response_time_seconds']:.1f}s")

print(f"\nCamera Activity:")
for camera_id, count in stats['cameras'].items():
    print(f"  {camera_id}: {count} detections")
```

## Integration with Day 7

```python
from ai.pipelines import (
    CameraFeedManager,
    FrameBuffer,
    DetectionOrchestrator
)
from ai.storage import init_database

# Initialize components
manager = CameraFeedManager()
buffer = FrameBuffer(max_size=100)
orchestrator = DetectionOrchestrator()
db = init_database()

# Setup cameras
manager.add_camera(...)
manager.start_all()

# Processing loop with storage
while True:
    # Get frame
    frames = manager.get_all_frames()
    for camera_id, frame in frames.items():
        buffer.put(frame)
    
    # Process
    frame = buffer.get(timeout=0.1)
    if frame:
        # Detect
        result = orchestrator.process_frame(frame)
        
        # Store permanently ✓
        detection = db.store_detection(result)
        
        # Handle alerts
        if result.alerts:
            print(f"Alert generated! Detection ID: {detection.id}")
            # Alerts automatically stored by store_detection()
```

## TimescaleDB Hypertables

If TimescaleDB is installed, tables are automatically converted to hypertables:

**Benefits:**
- Automatic time-based partitioning
- 10-100x faster time-range queries
- Automatic data compression
- Built-in data retention policies

**Hypertables Created:**
- `detections` partitioned by `timestamp`
- `persons` partitioned by `timestamp`
- `vehicles` partitioned by `timestamp`
- `animals` partitioned by `timestamp`
- `objects` partitioned by `timestamp`
- `alerts` partitioned by `created_at`
- `performance_metrics` partitioned by `timestamp`
- `system_logs` partitioned by `timestamp`

**Retention Policies:**
- System logs: 30 days
- Performance metrics: 30 days
- Detections: 90 days (configurable)
- Alerts: 365 days (configurable)

## Query Performance

**Indexed Queries (Fast ⚡):**

```python
# Time-range queries (TimescaleDB optimized)
db.query_detections(
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now()
)

# Camera + time queries
db.query_detections(
    camera_id="main_gate",
    start_time=datetime.now() - timedelta(hours=1)
)

# Threat level queries
db.query_detections(
    threat_level=ThreatLevel.HIGH,
    start_time=datetime.now() - timedelta(days=7)
)

# Alert status queries
db.get_pending_alerts()
```

**Complex Queries (Use SQL):**

```python
from sqlalchemy import and_, or_, func
from ai.storage.models import Detection, Person

with db.session_scope() as session:
    # Find patterns across cameras
    results = session.query(
        Detection.camera_id,
        func.count(Detection.id).label('count')
    ).filter(
        Detection.timestamp > datetime.now() - timedelta(hours=1),
        Detection.threat_level == ThreatLevel.HIGH
    ).group_by(Detection.camera_id).all()
    
    for camera_id, count in results:
        print(f"{camera_id}: {count} high threats")
```

## Backup & Recovery

### Manual Backup

```bash
# Backup entire database
pg_dump sentinelai > backup_$(date +%Y%m%d).sql

# Backup specific tables
pg_dump sentinelai -t detections -t alerts > critical_data.sql

# Compress backup
pg_dump sentinelai | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore

```bash
# Restore database
psql sentinelai < backup_20240101.sql

# Restore from compressed
gunzip -c backup_20240101.sql.gz | psql sentinelai
```

### Automated Backups (Production)

```bash
# Add to crontab
# Daily backup at 3 AM
0 3 * * * pg_dump sentinelai | gzip > /backups/sentinelai_$(date +\%Y\%m\%d).sql.gz

# Keep only last 30 days
0 4 * * * find /backups -name "sentinelai_*.sql.gz" -mtime +30 -delete
```

## Monitoring

### Database Size

```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('sentinelai'));

-- Check table sizes
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::regclass))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(tablename::regclass) DESC;
```

### Query Performance

```python
# Enable query timing
db = Database(database_url, echo=True)

# Check slow queries
with db.session_scope() as session:
    result = session.execute("""
        SELECT query, mean_exec_time, calls
        FROM pg_stat_statements
        WHERE mean_exec_time > 1000
        ORDER BY mean_exec_time DESC
        LIMIT 10
    """)
    
    for row in result:
        print(f"Query: {row[0][:100]}")
        print(f"Avg time: {row[1]:.1f}ms, Calls: {row[2]}")
```

## Troubleshooting

### Connection Issues

```python
# Test connection
from sqlalchemy import create_engine

engine = create_engine("postgresql://sentinel:sentinel@localhost/sentinelai")
try:
    connection = engine.connect()
    print("Connection successful!")
    connection.close()
except Exception as e:
    print(f"Connection failed: {e}")
```

### Slow Queries

```sql
-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Disk Space

```bash
# Check disk usage
df -h /var/lib/postgresql

# Check database size
psql -c "SELECT pg_size_pretty(pg_database_size('sentinelai'));"

# Clean old data (if retention policies not working)
psql sentinelai -c "DELETE FROM detections WHERE timestamp < NOW() - INTERVAL '90 days';"
psql sentinelai -c "VACUUM FULL detections;"
```

## Security Best Practices

1. **Use Strong Passwords**
   ```bash
   # Change default password
   psql -c "ALTER USER sentinel WITH PASSWORD 'very_strong_password_here';"
   ```

2. **Restrict Network Access**
   ```bash
   # Edit pg_hba.conf
   # Only allow local connections
   local   sentinelai   sentinel   md5
   ```

3. **Enable SSL/TLS (Production)**
   ```python
   db = Database(
       "postgresql://sentinel:password@localhost/sentinelai?sslmode=require"
   )
   ```

4. **Regular Backups**
   - Automated daily backups
   - Off-site backup storage
   - Regular restore testing

5. **Access Control**
   ```sql
   -- Create read-only user for reporting
   CREATE USER sentinel_readonly WITH PASSWORD 'password';
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO sentinel_readonly;
   ```

## Next Steps

**Day 9: Logging & Audit System**
- Structured logging
- Audit trails
- Error tracking
- Log aggregation

**Day 10: Configuration Management**
- Centralized configuration
- Environment-specific settings
- Runtime reconfiguration

**After Day 10:**
Complete core infrastructure ready for tracking & behavior analysis (Phase 2)

---

**Day 8 Complete!** 
Database persistence enables investigations, compliance, and historical analysis. ✅
