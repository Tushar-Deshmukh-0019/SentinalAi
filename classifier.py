"""
Object Classification System

Maps COCO objects to security-relevant categories.

Critical distinctions:
- Personal items (backpack, handbag) = Context dependent
- Large containers (suitcase) = Higher scrutiny
- Abandoned objects = Critical threat
- Weapon-like objects = Immediate response
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional


class ObjectType(Enum):
    """
    Object type classification from COCO dataset.
    
    COCO class IDs (relevant objects):
    - 24: backpack
    - 25: umbrella
    - 26: handbag
    - 27: tie
    - 28: suitcase
    - 31: skis
    - 32: snowboard
    - 33: sports ball
    - 34: kite
    - 35: baseball bat
    - 36: baseball glove
    - 37: skateboard
    - 38: surfboard
    - 39: tennis racket
    - 41: cup
    - 42: fork
    - 43: knife (eating utensil, not weapon)
    - 44: spoon
    - 73: book
    - 84: book (duplicate)
    """
    
    # Personal carry items
    BACKPACK = 24
    HANDBAG = 26
    SUITCASE = 28
    UMBRELLA = 25
    
    # Sports equipment
    SPORTS_BALL = 33
    BASEBALL_BAT = 35
    SKATEBOARD = 37
    TENNIS_RACKET = 39
    SKIS = 31
    SNOWBOARD = 32
    
    # Smaller items
    BOTTLE = 41  # Using cup class
    BOOK = 73
    
    # Note: Actual weapons require specialized detection
    # These are included for general object tracking
    
    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return self.name.replace('_', ' ').title()
    
    @property
    def is_personal_item(self) -> bool:
        """Check if this is a personal carry item."""
        return self in [
            ObjectType.BACKPACK,
            ObjectType.HANDBAG,
            ObjectType.UMBRELLA,
            ObjectType.BOTTLE
        ]
    
    @property
    def is_large_container(self) -> bool:
        """Check if this is a large container (requires inspection)."""
        return self in [
            ObjectType.BACKPACK,
            ObjectType.SUITCASE
        ]
    
    @property
    def is_sports_equipment(self) -> bool:
        """Check if this is sports equipment."""
        return self in [
            ObjectType.SPORTS_BALL,
            ObjectType.BASEBALL_BAT,
            ObjectType.SKATEBOARD,
            ObjectType.TENNIS_RACKET,
            ObjectType.SKIS,
            ObjectType.SNOWBOARD
        ]
    
    @property
    def can_be_weapon(self) -> bool:
        """
        Check if object could potentially be used as weapon.
        
        Note: This is NOT weapon detection (that's Day 59-62).
        This is awareness that certain objects require attention.
        """
        return self in [
            ObjectType.BASEBALL_BAT,  # Blunt object
            ObjectType.SKATEBOARD,    # Blunt object
        ]
    
    @property
    def typical_context(self) -> str:
        """Expected context for this object."""
        contexts = {
            ObjectType.BACKPACK: "School, hiking, travel",
            ObjectType.HANDBAG: "Shopping, business, daily carry",
            ObjectType.SUITCASE: "Travel, business trips",
            ObjectType.UMBRELLA: "Rain protection",
            ObjectType.SPORTS_BALL: "Sports venues, parks",
            ObjectType.BASEBALL_BAT: "Sports venues, recreational areas",
            ObjectType.SKATEBOARD: "Parks, recreational areas",
            ObjectType.TENNIS_RACKET: "Sports venues, clubs"
        }
        return contexts.get(self, "General use")
    
    @classmethod
    def from_class_id(cls, class_id: int) -> 'ObjectType':
        """Convert COCO class ID to ObjectType."""
        for obj_type in cls:
            if obj_type.value == class_id:
                return obj_type
        
        raise ValueError(
            f"Class ID {class_id} is not a recognized object type."
        )
    
    @classmethod
    def get_all_class_ids(cls) -> list:
        """Get all COCO class IDs that represent objects we track."""
        return [obj.value for obj in cls]


class ObjectSize(Enum):
    """
    Object size classification.
    
    Size affects:
    - Inspection requirements (large = more scrutiny)
    - Concealment capacity (large backpack vs. small purse)
    - Threat potential (large container = higher capacity)
    """
    
    SMALL = 1   # Bottles, books, small bags
    MEDIUM = 2  # Handbags, small backpacks
    LARGE = 3   # Large backpacks, suitcases
    
    @property
    def inspection_level(self) -> str:
        """Required inspection level."""
        levels = {
            ObjectSize.SMALL: "Visual inspection",
            ObjectSize.MEDIUM: "Standard screening",
            ObjectSize.LARGE: "Detailed inspection required"
        }
        return levels[self]
    
    @property
    def concealment_capacity(self) -> str:
        """What could be concealed in this size."""
        capacity = {
            ObjectSize.SMALL: "Minimal (phone, wallet)",
            ObjectSize.MEDIUM: "Moderate (laptop, documents)",
            ObjectSize.LARGE: "High (equipment, large items)"
        }
        return capacity[self]


class RiskLevel(Enum):
    """
    Risk level for object detection.
    
    Based on:
    - Object type
    - Object size  
    - Person association
    - Location context
    - Time context
    """
    
    NONE = 0        # Benign object in normal context
    LOW = 1         # Personal item with owner
    MODERATE = 2    # Requires attention (large bag, suspicious context)
    HIGH = 3        # Abandoned object, weapon-like, urgent
    CRITICAL = 4    # Immediate threat (confirmed weapon, IED suspicion)
    
    @property
    def should_alert(self) -> bool:
        """Whether this risk level requires operator alert."""
        return self.value >= 2  # MODERATE or higher
    
    @property
    def response_time(self) -> str:
        """Expected response time for this risk level."""
        times = {
            RiskLevel.NONE: "No action required",
            RiskLevel.LOW: "Log and monitor",
            RiskLevel.MODERATE: "Alert within 2 minutes",
            RiskLevel.HIGH: "Alert within 30 seconds",
            RiskLevel.CRITICAL: "Immediate alert (< 10 seconds)"
        }
        return times[self]
    
    @property
    def alert_message(self) -> str:
        """Alert message template."""
        messages = {
            RiskLevel.NONE: "Object detected - no action required",
            RiskLevel.LOW: "Object detected - routine monitoring",
            RiskLevel.MODERATE: "Suspicious object - review required",
            RiskLevel.HIGH: "High-risk object - immediate attention",
            RiskLevel.CRITICAL: "CRITICAL THREAT - immediate response"
        }
        return messages[self]


@dataclass
class ObjectCharacteristics:
    """
    Complete object characteristics for security assessment.
    """
    
    object_type: ObjectType
    size: ObjectSize
    confidence: float
    
    # Physical characteristics
    bbox_area: int
    aspect_ratio: float
    
    # Association
    near_person: bool = False
    person_distance: Optional[int] = None  # Pixels to nearest person
    
    # Temporal
    stationary_time: float = 0.0  # Seconds object has been stationary
    is_abandoned: bool = False
    
    # Context
    in_restricted_zone: bool = False
    time_of_day: str = "unknown"
    
    @property
    def risk_level(self) -> RiskLevel:
        """
        Calculate risk level based on characteristics.
        
        Logic:
        1. Abandoned object in public area = HIGH/CRITICAL
        2. Large container without person = MODERATE
        3. Weapon-like object = HIGH
        4. Personal item with owner = LOW
        5. Small item = NONE
        """
        # CRITICAL: Abandoned object for extended time
        if self.is_abandoned and self.stationary_time > 600:  # 10 minutes
            if self.in_restricted_zone or self.object_type.is_large_container:
                return RiskLevel.CRITICAL
            return RiskLevel.HIGH
        
        # HIGH: Abandoned large container
        if self.is_abandoned and self.object_type.is_large_container:
            return RiskLevel.HIGH
        
        # HIGH: Weapon-like object
        if self.object_type.can_be_weapon and not self.near_person:
            return RiskLevel.HIGH
        
        # MODERATE: Large container in restricted zone
        if self.object_type.is_large_container and self.in_restricted_zone:
            if not self.near_person:
                return RiskLevel.MODERATE
        
        # MODERATE: Any abandoned object
        if self.is_abandoned:
            return RiskLevel.MODERATE
        
        # LOW: Personal item with nearby person
        if self.object_type.is_personal_item and self.near_person:
            return RiskLevel.LOW
        
        # NONE: Small object or normal context
        if self.size == ObjectSize.SMALL:
            return RiskLevel.NONE
        
        return RiskLevel.LOW
    
    @property
    def should_inspect(self) -> bool:
        """Whether this object requires inspection."""
        # Always inspect large containers
        if self.object_type.is_large_container:
            return True
        
        # Inspect abandoned objects
        if self.is_abandoned:
            return True
        
        # Inspect high-risk objects
        if self.risk_level.value >= RiskLevel.MODERATE.value:
            return True
        
        return False
    
    @property
    def alert_reason(self) -> str:
        """Explanation for alert/risk level."""
        reasons = []
        
        if self.is_abandoned:
            reasons.append(f"Abandoned {self.object_type.display_name}")
            if self.stationary_time > 0:
                reasons.append(f"unattended for {int(self.stationary_time)}s")
        
        if self.object_type.is_large_container:
            reasons.append("Large container (inspection required)")
        
        if self.in_restricted_zone:
            reasons.append("In restricted zone")
        
        if self.object_type.can_be_weapon:
            reasons.append("Weapon-like object")
        
        if not self.near_person and not self.is_abandoned:
            reasons.append("No associated person")
        
        if not reasons:
            reasons.append(f"{self.object_type.display_name} detected")
        
        return " | ".join(reasons)
    
    def associate_with_person(self, person_center: Tuple[int, int], threshold: int = 100):
        """
        Associate this object with a person.
        
        Args:
            person_center: (x, y) coordinates of person center
            threshold: Maximum distance in pixels to consider "near"
        """
        # Calculate distance (simplified Euclidean)
        obj_x = self.bbox_area  # Would use actual center in full implementation
        obj_y = self.bbox_area
        
        dist = ((person_center[0] - obj_x) ** 2 + (person_center[1] - obj_y) ** 2) ** 0.5
        
        if dist <= threshold:
            self.near_person = True
            self.person_distance = int(dist)
            self.is_abandoned = False
        else:
            self.near_person = False
            self.person_distance = int(dist)
    
    def update_temporal(self, time_delta: float):
        """
        Update temporal characteristics.
        
        Args:
            time_delta: Time elapsed since last update (seconds)
        """
        if not self.near_person:
            self.stationary_time += time_delta
            
            # Mark as abandoned after threshold
            if self.stationary_time > 120:  # 2 minutes
                self.is_abandoned = True
        else:
            self.stationary_time = 0.0
            self.is_abandoned = False
