"""
Vehicle Classification

Maps COCO/YOLOv8 vehicle classes to tactical categories.

Real defense scenario:
- Civilian car at border = investigate
- Military truck at border = check authorization
- Motorcycle alone = high mobility, single occupant
- Bus = mass transport, unusual at border

Each vehicle type has different threat implications.
"""

from enum import Enum
from dataclasses import dataclass


class VehicleType(Enum):
    """
    Vehicle type classification.
    
    Maps COCO dataset classes to tactical categories.
    """
    
    # COCO class_id mapping:
    CAR = 2          # Cars, sedans, SUVs
    MOTORCYCLE = 3   # Motorcycles, scooters
    BUS = 5          # Buses, large passenger vehicles
    TRUCK = 7        # Trucks, pickups, cargo vehicles
    
    # Future expansion (requires custom training):
    # MILITARY_VEHICLE = 100
    # ARMORED_VEHICLE = 101
    # EMERGENCY = 102
    
    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return self.name.replace('_', ' ').title()
    
    @property
    def tactical_priority(self) -> int:
        """
        Priority level for tactical assessment (1-5).
        
        Higher = More significant in threat assessment.
        
        Why these priorities?
        - Trucks: Large capacity, can carry equipment/personnel
        - Cars: Standard, most common, medium concern
        - Motorcycles: High mobility, evasion capability
        - Buses: Unusual at borders, mass transport
        """
        priority_map = {
            VehicleType.TRUCK: 5,      # Highest concern
            VehicleType.BUS: 4,        # Unusual, investigate
            VehicleType.CAR: 3,        # Standard concern
            VehicleType.MOTORCYCLE: 3  # High mobility
        }
        return priority_map.get(self, 3)
    
    @property
    def typical_occupants(self) -> tuple:
        """
        Typical occupant count (min, max).
        
        Used for vehicle-person correlation analysis.
        If detected persons don't match expected range, raises suspicion.
        
        Example:
        - Car + 7 people = unusual (expect 1-5)
        - Truck + 1 person = normal
        - Motorcycle + 3 people = highly unusual
        """
        occupant_map = {
            VehicleType.CAR: (1, 5),
            VehicleType.MOTORCYCLE: (1, 2),
            VehicleType.TRUCK: (1, 3),
            VehicleType.BUS: (5, 50)
        }
        return occupant_map.get(self, (1, 10))
    
    @classmethod
    def from_class_id(cls, class_id: int) -> 'VehicleType':
        """
        Convert COCO class ID to VehicleType.
        
        Args:
            class_id: COCO dataset class ID
            
        Returns:
            VehicleType enum
            
        Raises:
            ValueError: If class_id doesn't map to a vehicle
        """
        class_id_map = {
            2: cls.CAR,
            3: cls.MOTORCYCLE,
            5: cls.BUS,
            7: cls.TRUCK
        }
        
        if class_id not in class_id_map:
            raise ValueError(
                f"Class ID {class_id} is not a recognized vehicle type. "
                f"Valid IDs: {list(class_id_map.keys())}"
            )
        
        return class_id_map[class_id]
    
    @classmethod
    def get_all_class_ids(cls) -> list:
        """Get all COCO class IDs that represent vehicles."""
        return [2, 3, 5, 7]


class VehicleSize(Enum):
    """
    Vehicle size classification.
    
    Based on bounding box area and vehicle type.
    
    Why size matters:
    - LARGE vehicles (trucks, buses): Can carry cargo, equipment, personnel
    - MEDIUM vehicles (SUVs, vans): Standard patrol vehicles
    - SMALL vehicles (motorcycles, compact cars): Limited capacity, high mobility
    """
    
    SMALL = 1   # Motorcycles, compact cars
    MEDIUM = 2  # Sedans, SUVs, pickups
    LARGE = 3   # Trucks, buses, heavy vehicles
    
    @property
    def cargo_capacity(self) -> str:
        """Estimated cargo capacity."""
        capacity_map = {
            VehicleSize.SMALL: "Minimal (personal items only)",
            VehicleSize.MEDIUM: "Moderate (trunk, backseat)",
            VehicleSize.LARGE: "High (cargo bed, large interior)"
        }
        return capacity_map[self]
    
    @property
    def threat_modifier(self) -> float:
        """
        Threat score modifier based on size.
        
        Larger vehicles = greater cargo capacity = potentially higher threat.
        
        Used in threat scoring calculation (Day 69+).
        """
        modifier_map = {
            VehicleSize.SMALL: 1.0,   # Baseline
            VehicleSize.MEDIUM: 1.2,  # +20% threat consideration
            VehicleSize.LARGE: 1.5    # +50% threat consideration
        }
        return modifier_map[self]


@dataclass
class VehicleCharacteristics:
    """
    Complete vehicle characteristics for tactical analysis.
    
    This information feeds into:
    - Threat scoring (Day 69+)
    - Vehicle-person correlation (Day 56)
    - Authorization checks (Day 53-55)
    - Behavior analysis (Day 23-35)
    """
    
    vehicle_type: VehicleType
    size: VehicleSize
    confidence: float
    
    # Physical characteristics
    bbox_area: int
    aspect_ratio: float  # width/height
    
    # Tactical flags
    is_stationary: bool = False
    is_oversized: bool = False  # Unusually large for type
    has_visible_cargo: bool = False  # Future: computer vision analysis
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return f"{self.size.name.title()} {self.vehicle_type.display_name}"
    
    @property
    def base_threat_level(self) -> int:
        """
        Base threat level (0-100) based on vehicle characteristics alone.
        
        This is just ONE component of the overall threat score.
        
        Factors:
        - Vehicle type tactical priority
        - Size modifier
        - Confidence in detection
        
        Real threat score will add:
        - Time of day
        - Authorization status
        - Associated persons
        - Location
        - Behavior
        - etc.
        """
        base = self.vehicle_type.tactical_priority * 10  # 30-50
        size_addition = (self.size.value - 1) * 5  # 0-10
        confidence_factor = self.confidence  # 0.45-1.0
        
        threat = (base + size_addition) * confidence_factor
        
        # Modifiers
        if self.is_oversized:
            threat *= 1.3  # Unusual size = higher suspicion
        
        return min(int(threat), 100)
    
    def matches_occupant_count(self, person_count: int) -> bool:
        """
        Check if person count matches expected range for this vehicle.
        
        Args:
            person_count: Number of persons detected near vehicle
            
        Returns:
            True if count is within expected range, False if suspicious
            
        Example:
            car = VehicleCharacteristics(VehicleType.CAR, ...)
            car.matches_occupant_count(3)  # True (1-5 expected)
            car.matches_occupant_count(10) # False (unusual)
        """
        min_occ, max_occ = self.vehicle_type.typical_occupants
        return min_occ <= person_count <= max_occ
