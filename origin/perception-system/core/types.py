"""
Core Data Types
===============

Data structures used throughout the perception system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import time


# =============================================================================
# Detection & Tracking Types
# =============================================================================

@dataclass
class BoundingBox:
    """2D bounding box in pixel coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BoundingBox':
        return cls(arr[0], arr[1], arr[2], arr[3])
    
    def iou(self, other: 'BoundingBox') -> float:
        """Compute Intersection over Union"""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = self.area + other.area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedObject:
    """A tracked object with state estimation"""
    id: int
    class_id: int = 0
    label: str = "object"
    confidence: float = 0.0
    
    # 2D bounding box [x1, y1, x2, y2]
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # Estimated center position
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Velocity in pixels/frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Full state vector [x, y, vx, vy, w, h] for UKF
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    
    # 3D position in world frame (if depth available)
    position_3d: Optional[np.ndarray] = None
    
    # Distance from camera (meters)
    distance: float = 0.0
    
    # World velocity (m/s)
    velocity_world: Optional[np.ndarray] = None
    
    # Tracking metadata
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    
    # Motion direction ('left', 'right', 'up', 'down', 'stationary')
    direction: str = "stationary"
    
    # Predicted future positions
    predicted_positions: List[np.ndarray] = field(default_factory=list)
    
    # Trajectory history
    trajectory: List[np.ndarray] = field(default_factory=list)
    
    # Time to collision (seconds)
    time_to_collision: Optional[float] = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def update_direction(self, speed_threshold: float = 2.0):
        """Update movement direction from velocity"""
        vx, vy = self.velocity[0], self.velocity[1]
        speed = np.linalg.norm(self.velocity)
        
        if speed < speed_threshold:
            self.direction = "stationary"
        elif abs(vx) > abs(vy):
            self.direction = "right" if vx > 0 else "left"
        else:
            self.direction = "down" if vy > 0 else "up"


# =============================================================================
# Depth Types
# =============================================================================

@dataclass
class DepthResult:
    """Result from depth estimation"""
    depth: np.ndarray  # HxW depth map in meters
    confidence: Optional[np.ndarray] = None  # HxW confidence map
    colored: Optional[np.ndarray] = None  # HxWx3 visualization
    min_depth: float = 0.0
    max_depth: float = 10.0
    valid: bool = True
    
    def get_depth_at(self, x: int, y: int, window: int = 5) -> float:
        """Get median depth in window around point"""
        h, w = self.depth.shape
        x = np.clip(x, window, w - window - 1)
        y = np.clip(y, window, h - window - 1)
        
        roi = self.depth[y-window:y+window+1, x-window:x+window+1]
        valid = (roi > self.min_depth) & (roi < self.max_depth)
        
        if np.any(valid):
            return float(np.median(roi[valid]))
        return 0.0


# =============================================================================
# Feature Types
# =============================================================================

@dataclass
class FeatureResult:
    """Result from feature extraction"""
    keypoints: np.ndarray  # Nx2 array of (x, y)
    descriptors: np.ndarray  # NxD descriptors
    scores: Optional[np.ndarray] = None  # N scores
    
    @property
    def num_features(self) -> int:
        return len(self.keypoints) if self.keypoints is not None else 0


@dataclass
class SemanticResult:
    """Result from semantic feature extraction (DINOv2)"""
    features: np.ndarray  # HxWxD or global descriptor
    pca_visualization: Optional[np.ndarray] = None  # RGB visualization
    patch_tokens: Optional[np.ndarray] = None  # Per-patch features


# =============================================================================
# Optical Flow Types
# =============================================================================

@dataclass
class FlowResult:
    """Result from optical flow computation"""
    flow: np.ndarray  # HxWx2 flow field (dx, dy)
    magnitude: Optional[np.ndarray] = None  # HxW magnitude
    visualization: Optional[np.ndarray] = None  # HxWx3 color visualization
    valid: bool = True
    
    def get_flow_at(self, x: int, y: int) -> Tuple[float, float]:
        """Get flow vector at point"""
        h, w = self.flow.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return float(self.flow[y, x, 0]), float(self.flow[y, x, 1])
        return 0.0, 0.0
    
    def get_flow_in_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Get median flow vector in bounding box"""
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = self.flow.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            roi = self.flow[y1:y2, x1:x2]
            return np.array([np.median(roi[..., 0]), np.median(roi[..., 1])])
        return np.zeros(2)


# =============================================================================
# Visual Odometry Types
# =============================================================================

@dataclass
class VOState:
    """Visual odometry state"""
    # 4x4 pose matrix (camera to world)
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    # Position in world frame (extracted from pose)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Orientation (roll, pitch, yaw) in degrees
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Velocity estimate (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Quality metrics
    num_inliers: int = 0
    num_matches: int = 0
    confidence: float = 0.0
    scale: float = 1.0
    
    # Status flags
    is_lost: bool = False
    is_keyframe: bool = False
    is_loop_closure: bool = False
    
    # Frame info
    frame_id: int = 0
    timestamp: float = 0.0
    
    def update_from_pose(self):
        """Extract position and orientation from pose matrix"""
        self.position = self.pose[:3, 3].copy()
        self.orientation = self._rotation_to_euler(self.pose[:3, :3])
    
    @staticmethod
    def _rotation_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (degrees)"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.degrees(np.array([roll, pitch, yaw]))


# =============================================================================
# Mapping Types
# =============================================================================

@dataclass
class Place:
    """A remembered place for loop closure and mapping"""
    id: int
    pose: np.ndarray  # 4x4 transformation
    descriptor: Optional[np.ndarray] = None  # Global scene descriptor
    keypoints: Optional[np.ndarray] = None
    local_descriptors: Optional[np.ndarray] = None
    thumbnail: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    neighbors: List[int] = field(default_factory=list)


@dataclass
class MapPoint:
    """A 3D point in the map"""
    position: np.ndarray  # 3D position
    color: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: Optional[np.ndarray] = None
    descriptor: Optional[np.ndarray] = None
    observations: int = 1
    semantic_class: int = -1


# =============================================================================
# Complete Perception Output
# =============================================================================

@dataclass
class PerceptionResult:
    """Complete output from perception pipeline"""
    # Timing
    timestamp: float = 0.0
    frame_id: int = 0
    process_time_ms: float = 0.0
    
    # Original and preprocessed image
    image: Optional[np.ndarray] = None
    image_rgb: Optional[np.ndarray] = None
    
    # Depth
    depth: Optional[DepthResult] = None
    
    # Detections and tracked objects
    detections: List[TrackedObject] = field(default_factory=list)
    raw_boxes: Optional[np.ndarray] = None
    raw_scores: Optional[np.ndarray] = None
    raw_classes: Optional[np.ndarray] = None
    
    # Features
    features: Optional[FeatureResult] = None
    semantics: Optional[SemanticResult] = None
    
    # Optical flow
    flow: Optional[FlowResult] = None
    
    # Visual odometry
    odometry: Optional[VOState] = None
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    @property
    def num_features(self) -> int:
        return self.features.num_features if self.features else 0
    
    @property
    def depth_valid(self) -> bool:
        return self.depth is not None and self.depth.valid
