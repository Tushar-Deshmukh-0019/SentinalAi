"""
Animal Classification System

Maps COCO dataset animals to surveillance categories.

Critical distinction:
- Wildlife (deer, bear) = Filter out, no alert
- Companion animals (dog, cat) = Context dependent
  - Dog alone = Filter out
  - Dog with person = Normal
  - Dog without person in restricted zone = Unusual
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class AnimalType(Enum):
    """
    Animal type classification from COCO dataset.
    
    COCO class IDs:
    - 16: bird
    - 17: cat
    - 18: dog
    - 19: horse
    - 20: sheep
    - 21: cow
    - 22: elephant
    - 23: bear
    - 24: zebra
    - 25: giraffe
    """
    
    # Common wildlife (filter out)
    BIRD = 16
    BEAR = 23
    DEER = 19  # Using horse class for deer (similar shape)
    
    # Domestic animals (context dependent)
    DOG = 18
    CAT = 17
    
    # Livestock (usually expected in certain zones)
    COW = 21
    SHEEP = 20
    HORSE = 19
    
    # Exotic (rare but handled)
    ELEPHANT = 22
    ZEBRA = 24
    GIRAFFE = 25
    
    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return self.name.replace('_', ' ').title()
    
    @property
    def is_wildlife(self) -> bool:
        """
        Check if this is wild animal (should trigger no alert).
        
        Wildlife = animals that naturally occur in environment
        Non-wildlife = domestic animals that may be authorized
        """
        return self in [
            AnimalType.BIRD,
            AnimalType.BEAR,
            AnimalType.DEER
        ]
    
    @property
    def is_domestic(self) -> bool:
        """
        Check if domestic animal (context matters).
        
        Domestic animals:
        - May be authorized (patrol dogs, livestock)
        - May accompany persons
        - Presence alone is not always filtered
        """
        return self in [
            AnimalType.DOG,
            AnimalType.CAT,
            AnimalType.COW,
            AnimalType.SHEEP,
            AnimalType.HORSE
        ]
    
    @property
    def typical_behavior(self) -> str:
        """Expected behavior patterns for this animal."""
        behaviors = {
            AnimalType.BIRD: "Flying, perching, rapid movement",
            AnimalType.BEAR: "Quadrupedal, lumbering, large size",
            AnimalType.DEER: "Quadrupedal, graceful, alert posture",
            AnimalType.DOG: "Quadrupedal, variable size, may be with person",
            AnimalType.CAT: "Quadrupedal, small, stealthy",
            AnimalType.COW: "Quadrupedal, large, slow, grazing",
            AnimalType.SHEEP: "Quadrupedal, medium, herding behavior",
            AnimalType.HORSE: "Quadrupedal, large, powerful"
        }
        return behaviors.get(self, "Unknown behavior pattern")
    
    @classmethod
    def from_class_id(cls, class_id: int) -> 'AnimalType':
        """
        Convert COCO class ID to AnimalType.
        
        Args:
            class_id: COCO dataset class ID
            
        Returns:
            AnimalType enum
            
        Raises:
            ValueError: If class_id doesn't map to an animal
        """
        # Direct mapping
        for animal_type in cls:
            if animal_type.value == class_id:
                return animal_type
        
        raise ValueError(
            f"Class ID {class_id} is not a recognized animal type."
        )
    
    @classmethod
    def get_all_class_ids(cls) -> list:
        """Get all COCO class IDs that represent animals."""
        return [animal.value for animal in cls]


class AnimalSize(Enum):
    """
    Animal size classification.
    
    Used for:
    - Validation (deer should be LARGE, not SMALL)
    - Person/animal discrimination (human-sized vs. animal-sized)
    - False positive filtering
    """
    
    SMALL = 1   # Birds, cats, small dogs
    MEDIUM = 2  # Dogs, sheep, coyotes
    LARGE = 3   # Deer, bears, cattle, horses
    
    @property
    def typical_height_range(self) -> Tuple[int, int]:
        """
        Typical height range in pixels at 640x640 resolution.
        
        Used to validate size classification.
        Example: If "deer" detected but height < 100px, might be misclassification.
        """
        ranges = {
            AnimalSize.SMALL: (20, 80),
            AnimalSize.MEDIUM: (60, 150),
            AnimalSize.LARGE: (120, 300)
        }
        return ranges[self]
    
    @property
    def confusion_with_person(self) -> str:
        """
        Likelihood of confusion with person detection.
        
        LARGE animals are most likely to trigger false person detections
        because they're similar in size to humans.
        """
        confusion = {
            AnimalSize.SMALL: "Low (too small)",
            AnimalSize.MEDIUM: "Moderate (similar size to crouching person)",
            AnimalSize.LARGE: "High (similar size to standing person)"
        }
        return confusion[self]


class ThreatLevel(Enum):
    """
    Threat level for animal detection.
    
    Different from vehicle/person threat - this is about:
    1. Should we alert operators?
    2. Is this animal dangerous?
    3. Does presence indicate something unusual?
    """
    
    NONE = 0        # Common wildlife, filter out completely
    LOW = 1         # Domestic animal, log but no alert
    MODERATE = 2    # Unusual animal, notify operator
    HIGH = 3        # Dangerous animal or suspicious circumstance
    
    @property
    def should_alert(self) -> bool:
        """Whether this threat level should trigger operator alert."""
        return self.value >= 2  # MODERATE or HIGH
    
    @property
    def alert_message(self) -> str:
        """Alert message template for this threat level."""
        messages = {
            ThreatLevel.NONE: "Wildlife detected - no action required",
            ThreatLevel.LOW: "Domestic animal detected - logged",
            ThreatLevel.MODERATE: "Unusual animal activity - review recommended",
            ThreatLevel.HIGH: "Suspicious animal presence - immediate attention"
        }
        return messages[self]


@dataclass
class AnimalCharacteristics:
    """
    Complete animal characteristics for analysis.
    
    This information determines:
    - Whether to filter out detection
    - Whether to alert operator
    - How to log the event
    """
    
    animal_type: AnimalType
    size: AnimalSize
    confidence: float
    
    # Physical characteristics
    bbox_area: int
    aspect_ratio: float
    
    # Behavioral flags
    is_moving: bool = True
    is_alone: bool = True
    near_person: bool = False  # Within 50 pixels of person detection
    
    # Context
    time_of_day: str = "unknown"  # "day", "night", "twilight"
    in_expected_zone: bool = True  # e.g., cattle in pasture zone
    
    @property
    def threat_level(self) -> ThreatLevel:
        """
        Calculate threat level based on characteristics.
        
        Logic:
        1. Wildlife alone = NONE (filter out)
        2. Domestic animal in expected zone = LOW
        3. Domestic animal in unexpected zone = MODERATE
        4. Animal with suspicious circumstances = HIGH
        """
        # Wildlife is generally not a threat
        if self.animal_type.is_wildlife:
            # Exception: Bear is always at least LOW threat
            if self.animal_type == AnimalType.BEAR:
                return ThreatLevel.LOW
            return ThreatLevel.NONE
        
        # Domestic animals - context matters
        if self.animal_type.is_domestic:
            # Dog/cat with person = normal, LOW
            if self.near_person:
                return ThreatLevel.LOW
            
            # Animal in expected zone (e.g., livestock area) = LOW
            if self.in_expected_zone:
                return ThreatLevel.LOW
            
            # Dog alone in restricted zone at night = MODERATE
            if self.animal_type == AnimalType.DOG:
                if self.time_of_day == "night" and not self.in_expected_zone:
                    return ThreatLevel.MODERATE
            
            # Default for domestic = LOW
            return ThreatLevel.LOW
        
        # Unknown/exotic animals = MODERATE (unusual)
        return ThreatLevel.MODERATE
    
    @property
    def should_filter(self) -> bool:
        """
        Determine if this detection should be filtered out (no alert).
        
        Returns True if:
        - Threat level is NONE
        - High confidence wildlife detection
        - Common expected animal
        """
        if self.threat_level == ThreatLevel.NONE:
            return True
        
        # High confidence wildlife with low threat
        if self.animal_type.is_wildlife and self.confidence > 0.8:
            if self.threat_level == ThreatLevel.LOW:
                return True
        
        return False
    
    @property
    def filter_reason(self) -> str:
        """Explanation for why this was filtered."""
        if not self.should_filter:
            return "Not filtered - alert required"
        
        reasons = []
        
        if self.animal_type.is_wildlife:
            reasons.append(f"{self.animal_type.display_name} is wildlife")
        
        if self.threat_level == ThreatLevel.NONE:
            reasons.append("No threat to security")
        
        if self.confidence > 0.8:
            reasons.append(f"High confidence ({self.confidence:.0%})")
        
        return " | ".join(reasons)
    
    def get_conflict_resolution(
        self, 
        person_confidence: float
    ) -> Tuple[str, float]:
        """
        Resolve conflict when both person and animal detected.
        
        Args:
            person_confidence: Confidence from person detector
            
        Returns:
            (decision, confidence) where decision is "person" or "animal"
            
        Logic:
        - If animal confidence >> person confidence: Animal
        - If person confidence >> animal confidence: Person
        - If similar: Need additional analysis
        
        Threshold: 0.2 difference
        """
        diff = self.confidence - person_confidence
        
        if diff > 0.2:  # Animal confidence much higher
            return ("animal", self.confidence)
        elif diff < -0.2:  # Person confidence much higher
            return ("person", person_confidence)
        else:
            # Similar confidence - use size/type heuristics
            if self.size == AnimalSize.LARGE:
                # Large animals more likely to be confused
                if self.animal_type in [AnimalType.DEER, AnimalType.BEAR]:
                    return ("animal", self.confidence)
            
            # Default to person if ambiguous (better safe than sorry)
            return ("person", person_confidence)
