"""
Configuration Management
========================

Centralized configuration for all perception modules.
Supports loading from YAML and environment variables.
"""

import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera intrinsic and extrinsic parameters"""
    # Intrinsics
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    
    # Distortion coefficients (k1, k2, p1, p2, k3)
    distortion: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Extrinsics (camera to robot transform)
    camera_height: float = 0.5  # Height above ground/water surface (meters)
    camera_pitch: float = 0.0   # Pitch angle (radians, positive = looking down)
    camera_roll: float = 0.0
    camera_yaw: float = 0.0
    
    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def K_inv(self) -> np.ndarray:
        """Inverse of intrinsic matrix"""
        return np.linalg.inv(self.K)
    
    def from_calibration_file(self, path: str) -> 'CameraConfig':
        """Load camera calibration from OpenCV XML/YAML file"""
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode('camera_matrix').mat()
        dist = fs.getNode('distortion_coefficients').mat()
        fs.release()
        
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        if dist is not None:
            self.distortion = tuple(dist.flatten())
        
        return self


@dataclass
class DepthConfig:
    """Depth estimation configuration"""
    enabled: bool = True
    model: str = 'small'  # 'small', 'base', 'large'
    min_depth: float = 0.1  # meters
    max_depth: float = 20.0  # meters (underwater visibility)
    skip_frames: int = 2  # Process every N frames
    scale_factor: float = 1.0  # Manual depth scale adjustment


@dataclass
class DetectionConfig:
    """Object detection configuration"""
    enabled: bool = True
    model_path: str = 'yolo11n.pt'
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    classes: Tuple[int, ...] = (0,)  # COCO class IDs to detect
    image_size: int = 640


@dataclass
class TrackingConfig:
    """Multi-object tracking configuration"""
    enabled: bool = True
    tracker_type: str = 'ukf'  # 'ukf' or 'kalman'
    max_age: int = 30  # Frames before track deletion
    min_hits: int = 3   # Minimum detections before confirmed
    iou_threshold: float = 0.3
    distance_threshold: float = 50.0  # Pixels


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    enabled: bool = True
    extractor: str = 'xfeat'  # 'xfeat', 'orb', 'sift'
    max_features: int = 2000
    # DINOv2 for semantic features
    enable_semantics: bool = False
    dino_model: str = 'vits14'  # 'vits14', 'vitb14', 'vitl14'


@dataclass
class FlowConfig:
    """Optical flow configuration"""
    enabled: bool = True
    method: str = 'raft'  # 'raft', 'farneback'
    raft_model: str = 'raft-small.pth'
    skip_frames: int = 2
    scale: float = 0.5  # Processing scale


@dataclass 
class OdometryConfig:
    """Visual odometry configuration"""
    enabled: bool = True
    min_features: int = 80
    ransac_threshold: float = 1.5
    keyframe_translation: float = 0.3  # meters
    keyframe_rotation: float = 15.0    # degrees
    enable_loop_closure: bool = False
    loop_threshold: float = 0.85


@dataclass
class MappingConfig:
    """3D mapping configuration"""
    enabled: bool = False
    voxel_resolution: float = 0.05  # meters
    map_size: Tuple[float, float, float] = (20.0, 20.0, 5.0)  # XYZ in meters
    max_points: int = 100000


@dataclass
class PlanningConfig:
    """Path planning configuration"""
    enabled: bool = False
    grid_resolution: float = 0.1
    grid_size: Tuple[int, int] = (200, 200)
    robot_radius: float = 0.3
    max_speed: float = 0.8
    max_yaw_rate: float = 1.0


@dataclass
class PerceptionConfig:
    """Complete perception pipeline configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)


class Config:
    """
    Main configuration manager.
    
    Usage:
        config = Config()  # Default config
        config = Config.from_yaml('config.yaml')
        config = Config.from_dict({'camera': {'fx': 500}})
    """
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        config = PerceptionConfig()
        
        if 'camera' in data:
            for k, v in data['camera'].items():
                if hasattr(config.camera, k):
                    setattr(config.camera, k, v)
        
        if 'depth' in data:
            for k, v in data['depth'].items():
                if hasattr(config.depth, k):
                    setattr(config.depth, k, v)
        
        if 'detection' in data:
            for k, v in data['detection'].items():
                if hasattr(config.detection, k):
                    setattr(config.detection, k, v)
        
        if 'tracking' in data:
            for k, v in data['tracking'].items():
                if hasattr(config.tracking, k):
                    setattr(config.tracking, k, v)
        
        if 'features' in data:
            for k, v in data['features'].items():
                if hasattr(config.features, k):
                    setattr(config.features, k, v)
        
        if 'flow' in data:
            for k, v in data['flow'].items():
                if hasattr(config.flow, k):
                    setattr(config.flow, k, v)
        
        if 'odometry' in data:
            for k, v in data['odometry'].items():
                if hasattr(config.odometry, k):
                    setattr(config.odometry, k, v)
        
        if 'mapping' in data:
            for k, v in data['mapping'].items():
                if hasattr(config.mapping, k):
                    setattr(config.mapping, k, v)
        
        if 'planning' in data:
            for k, v in data['planning'].items():
                if hasattr(config.planning, k):
                    setattr(config.planning, k, v)
        
        return cls(config)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        data = {
            'camera': self.config.camera.__dict__,
            'depth': self.config.depth.__dict__,
            'detection': self.config.detection.__dict__,
            'tracking': self.config.tracking.__dict__,
            'features': self.config.features.__dict__,
            'flow': self.config.flow.__dict__,
            'odometry': self.config.odometry.__dict__,
            'mapping': self.config.mapping.__dict__,
            'planning': self.config.planning.__dict__,
        }
        
        # Convert tuples to lists for YAML
        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        data = convert_tuples(data)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    # Convenience accessors
    @property
    def camera(self) -> CameraConfig:
        return self.config.camera
    
    @property
    def depth(self) -> DepthConfig:
        return self.config.depth
    
    @property
    def detection(self) -> DetectionConfig:
        return self.config.detection
    
    @property
    def tracking(self) -> TrackingConfig:
        return self.config.tracking
    
    @property
    def features(self) -> FeatureConfig:
        return self.config.features
    
    @property
    def flow(self) -> FlowConfig:
        return self.config.flow
    
    @property
    def odometry(self) -> OdometryConfig:
        return self.config.odometry
    
    @property
    def mapping(self) -> MappingConfig:
        return self.config.mapping
    
    @property
    def planning(self) -> PlanningConfig:
        return self.config.planning


# AUV-specific presets
def get_robosub_config() -> Config:
    """Configuration preset for RoboSub competition"""
    config = PerceptionConfig()
    
    # Underwater camera settings (adjust based on your camera)
    config.camera.fx = 500.0
    config.camera.fy = 500.0
    config.camera.cx = 320.0
    config.camera.cy = 240.0
    config.camera.camera_pitch = 0.0  # Forward-facing
    
    # Underwater depth settings
    config.depth.max_depth = 15.0  # Pool depth
    config.depth.model = 'small'  # Fast for real-time
    
    # Detection for RoboSub objects
    config.detection.confidence_threshold = 0.6
    
    # Faster tracking for underwater motion
    config.tracking.max_age = 20
    config.tracking.min_hits = 2
    
    # Visual odometry
    config.odometry.min_features = 100
    config.odometry.enable_loop_closure = True
    
    return Config(config)
