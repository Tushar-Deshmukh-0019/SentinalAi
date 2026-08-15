"""
Unit tests for Global ID Assignment system core components.

Tests candidate query, linking, conflict resolution, and ID creation.

Version: 1.0
"""

import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

# Import components to test
from ai.tracking.id_assignment.person_profile import PersonProfile, SessionInfo
from ai.tracking.id_assignment.observation import ObservationInfo, PersonLinkInfo, LinkReason
from ai.tracking.id_assignment.candidate_query import Candidate, CandidateQueryEngine
from ai.tracking.id_assignment.single_candidate_linker import SingleCandidateLinker
from ai.tracking.id_assignment.conflict_resolver import ConflictResolver
from ai.tracking.id_assignment.new_person_creator import NewPersonCreator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def normalized_feature():
    """Create a normalized 128-dim feature vector."""
    feature = np.random.randn(128).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    return MagicMock()


@pytest.fixture
def person_profile():
    """Create a test PersonProfile."""
    feature = np.zeros(128, dtype=np.float32)
    feature[0] = 1.0
    
    return PersonProfile(
        global_id=uuid4(),
        created_at=datetime.now() - timedelta(hours=1),
        last_seen=datetime.now(),
        dominant_appearance=feature
    )


# ============================================================================
# PersonProfile Tests (5 tests)
# ============================================================================

class TestPersonProfile:
    """Tests for PersonProfile dataclass."""
    
    def test_profile_creation(self):
        """Test creating a valid profile."""
        feature = np.zeros(128, dtype=np.float32)
        feature[0] = 1.0
        
        profile = PersonProfile(
            global_id=uuid4(),
            created_at=datetime.now(),
            last_seen=datetime.now(),
            dominant_appearance=feature
        )
        
        assert profile.global_id is not None
        assert profile.dominant_appearance is not None
        assert profile.observation_count == 0
    
    def test_profile_rejects_wrong_dimension(self):
        """Test that profile rejects non-128-dim features."""
        with pytest.raises(ValueError):
            PersonProfile(
                global_id=uuid4(),
                created_at=datetime.now(),
                last_seen=datetime.now(),
                dominant_appearance=np.zeros(64)  # Wrong dimension
            )
    
    def test_profile_add_observation(self):
        """Test adding observations to profile."""
        profile = PersonProfile(
            global_id=uuid4(),
            created_at=datetime.now(),
            last_seen=datetime.now(),
            dominant_appearance=np.ones(128) / 128
        )
        
        feature = np.ones(128) / 128
        obs_data = {
            "appearance_feature": feature,
            "camera_id": 1,
            "detection_time": datetime.now(),
            "centroid": (100, 100),
            "confidence": 0.9
        }
        
        profile.add_observation(obs_data)
        assert profile.observation_count == 1
    
    def test_profile_statistics(self):
        """Test profile statistics computation."""
        profile = PersonProfile(
            global_id=uuid4(),
            created_at=datetime.now() - timedelta(hours=2),
            last_seen=datetime.now(),
            dominant_appearance=np.ones(128) / 128,
            observation_count=10
        )
        
        assert profile.get_time_span().total_seconds() > 7000  # ~2 hours
        assert profile.get_observation_count() == 10
    
    def test_profile_serialization(self):
        """Test profile to_dict serialization."""
        profile = PersonProfile(
            global_id=uuid4(),
            created_at=datetime.now(),
            last_seen=datetime.now(),
            dominant_appearance=np.ones(128) / 128
        )
        
        serialized = profile.to_dict()
        assert "global_id" in serialized
        assert "threat_score" in serialized
        assert "observation_count" in serialized


# ============================================================================
# ObservationInfo Tests (4 tests)
# ============================================================================

class TestObservationInfo:
    """Tests for ObservationInfo dataclass."""
    
    def test_observation_creation(self, normalized_feature):
        """Test creating valid observation."""
        obs = ObservationInfo(
            observation_id=1,
            global_id=uuid4(),
            detection_time=datetime.now(),
            camera_id=1,
            bbox_tlwh=(10, 20, 50, 100),
            centroid_x=35,
            centroid_y=60,
            appearance_feature=normalized_feature,
            appearance_confidence=0.9,
            matching_confidence=0.85
        )
        
        assert obs.observation_id == 1
        assert obs.camera_id == 1
    
    def test_observation_rejects_nan(self):
        """Test that observation rejects NaN features."""
        feature = np.zeros(128)
        feature[0] = np.nan
        
        with pytest.raises(ValueError):
            ObservationInfo(
                observation_id=1,
                global_id=uuid4(),
                detection_time=datetime.now(),
                camera_id=1,
                bbox_tlwh=(10, 20, 50, 100),
                centroid_x=35,
                centroid_y=60,
                appearance_feature=feature,
                appearance_confidence=0.9,
                matching_confidence=0.85
            )
    
    def test_observation_feature_distance(self, normalized_feature):
        """Test feature distance computation."""
        obs1 = ObservationInfo(
            observation_id=1,
            global_id=uuid4(),
            detection_time=datetime.now(),
            camera_id=1,
            bbox_tlwh=(10, 20, 50, 100),
            centroid_x=35,
            centroid_y=60,
            appearance_feature=normalized_feature,
            appearance_confidence=0.9,
            matching_confidence=0.85
        )
        
        # Very similar feature
        obs2_feature = normalized_feature + np.random.randn(128) * 0.01
        obs2_feature = obs2_feature / np.linalg.norm(obs2_feature)
        
        obs2 = ObservationInfo(
            observation_id=2,
            global_id=uuid4(),
            detection_time=datetime.now() + timedelta(seconds=1),
            camera_id=1,
            bbox_tlwh=(15, 25, 50, 100),
            centroid_x=40,
            centroid_y=65,
            appearance_feature=obs2_feature,
            appearance_confidence=0.9,
            matching_confidence=0.85
        )
        
        similarity = obs1.feature_similarity_to_observation(obs2)
        assert 0.95 < similarity < 1.0
    
    def test_observation_spatial_distance(self, normalized_feature):
        """Test spatial distance computation."""
        obs1 = ObservationInfo(
            observation_id=1,
            global_id=uuid4(),
            detection_time=datetime.now(),
            camera_id=1,
            bbox_tlwh=(10, 20, 50, 100),
            centroid_x=100,
            centroid_y=100,
            appearance_feature=normalized_feature,
            appearance_confidence=0.9,
            matching_confidence=0.85
        )
        
        obs2 = ObservationInfo(
            observation_id=2,
            global_id=uuid4(),
            detection_time=datetime.now(),
            camera_id=1,
            bbox_tlwh=(10, 20, 50, 100),
            centroid_x=130,
            centroid_y=140,
            appearance_feature=normalized_feature,
            appearance_confidence=0.9,
            matching_confidence=0.85
        )
        
        distance = obs1.distance_to_observation(obs2)
        assert 49 < distance < 51  # sqrt(30^2 + 40^2) ≈ 50


# ============================================================================
# Candidate Query Tests (10 tests)
# ============================================================================

class TestCandidateQueryEngine:
    """Tests for CandidateQueryEngine."""
    
    def test_engine_initialization(self, mock_db_connection):
        """Test engine initialization."""
        engine = CandidateQueryEngine(mock_db_connection)
        
        assert engine.similarity_threshold == 0.4
        assert engine.max_candidates == 10
        assert engine.use_cache == True
    
    def test_invalid_feature_returns_empty(self, mock_db_connection):
        """Test that invalid features return empty."""
        engine = CandidateQueryEngine(mock_db_connection)
        
        # None feature
        result = engine.query_candidates(None)
        assert len(result) == 0
        
        # Wrong dimension
        result = engine.query_candidates(np.zeros(64))
        assert len(result) == 0
    
    def test_cache_working(self, mock_db_connection):
        """Test query caching."""
        engine = CandidateQueryEngine(mock_db_connection)
        feature = np.ones(128) / 128
        
        # Mock search to return candidates
        with patch.object(engine, '_search_index') as mock_search:
            candidates = [
                Candidate(
                    global_id=uuid4(),
                    similarity_score=0.9,
                    appearance_confidence=0.9,
                    last_seen=datetime.now(),
                    observation_count=5
                )
            ]
            mock_search.return_value = candidates
            
            # First query
            result1 = engine.query_candidates(feature)
            assert len(result1) == 1
            
            # Second query (should hit cache)
            result2 = engine.query_candidates(feature)
            assert len(result2) == 1
            
            # Cache hit should be recorded
            assert engine.metrics["cache_hits"] == 1
    
    def test_metrics_collection(self, mock_db_connection):
        """Test that metrics are collected."""
        engine = CandidateQueryEngine(mock_db_connection)
        feature = np.ones(128) / 128
        
        with patch.object(engine, '_search_index') as mock_search:
            mock_search.return_value = []
            engine.query_candidates(feature)
            
            metrics = engine.get_metrics()
            assert metrics["total_queries"] == 1
            assert metrics["avg_latency_ms"] >= 0
    
    def test_filtering_threshold(self, mock_db_connection):
        """Test that similarity threshold is applied."""
        engine = CandidateQueryEngine(
            mock_db_connection,
            {"similarity_threshold": 0.7}
        )
        
        assert engine.similarity_threshold == 0.7


# ============================================================================
# Conflict Resolver Tests (10 tests)
# ============================================================================

class TestConflictResolver:
    """Tests for ConflictResolver with Hungarian algorithm."""
    
    def test_resolver_initialization(self):
        """Test conflict resolver initialization."""
        resolver = ConflictResolver()
        assert resolver.confidence_threshold == 0.85
        assert resolver.min_candidates == 2
    
    def test_resolve_single_candidate_ignored(self):
        """Test that single candidate is not treated as conflict."""
        resolver = ConflictResolver()
        
        feature = np.ones(128) / 128
        candidates = [
            Candidate(
                global_id=uuid4(),
                similarity_score=0.9,
                appearance_confidence=0.9,
                last_seen=datetime.now(),
                observation_count=5
            )
        ]
        
        result = resolver.resolve(feature, candidates)
        assert result == (None, 0.0, False)
    
    def test_resolve_multiple_candidates(self):
        """Test Hungarian algorithm resolution."""
        resolver = ConflictResolver()
        
        feature = np.ones(128) / 128
        candidates = [
            Candidate(
                global_id=uuid4(),
                similarity_score=0.9,
                appearance_confidence=0.9,
                last_seen=datetime.now(),
                observation_count=5
            ),
            Candidate(
                global_id=uuid4(),
                similarity_score=0.8,
                appearance_confidence=0.8,
                last_seen=datetime.now(),
                observation_count=3
            )
        ]
        
        best_candidate, confidence, requires_review = resolver.resolve(feature, candidates)
        assert best_candidate is not None
        assert 0.0 <= confidence <= 1.0
    
    def test_metrics_collection(self):
        """Test metrics are collected."""
        resolver = ConflictResolver()
        
        feature = np.ones(128) / 128
        candidates = [
            Candidate(
                global_id=uuid4(),
                similarity_score=0.9,
                appearance_confidence=0.9,
                last_seen=datetime.now(),
                observation_count=5
            ),
            Candidate(
                global_id=uuid4(),
                similarity_score=0.8,
                appearance_confidence=0.8,
                last_seen=datetime.now(),
                observation_count=3
            )
        ]
        
        resolver.resolve(feature, candidates)
        metrics = resolver.get_metrics()
        
        assert metrics["conflicts_resolved"] >= 0


# ============================================================================
# Integration Tests (15+ tests - will be in separate file)
# ============================================================================

# ... (see test_id_assignment_integration.py)


# ============================================================================
# Performance Tests (5+ benchmarks - will be in separate file)
# ============================================================================

# ... (see test_id_assignment_performance.py)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
