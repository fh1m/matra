"""
Visual Odometry with Depth Scale Recovery
==========================================

Monocular visual odometry that recovers absolute scale using depth estimates.

Key approach: scale = median(depth_measured) / median(Z_triangulated)

Features:
- Essential matrix + RANSAC for robust pose estimation
- Triangulation-based scale recovery
- Keyframe management
- Loop closure detection with DINOv2
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
import torch

from core.config import Config
from core.types import VOState, Place
from features.xfeat_extractor import XFeatExtractor
from features.dino_extractor import DINOv2Extractor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class VisualOdometry:
    """
    Visual Odometry with depth-based scale recovery.
    
    Usage:
        config = Config()
        vo = VisualOdometry(config)
        
        for frame, depth in video:
            state = vo.process(frame, depth)
            print(f"Position: {state.position}")
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.vo_config = config.odometry
        self.camera_config = config.camera
        
        # Camera intrinsics
        self.K = self.camera_config.K
        
        # Feature extractor
        self.feature_extractor = XFeatExtractor(config)
        
        # ORB fallback
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # State
        self.state = VOState()
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None
        
        # Keyframe management
        self.keyframe_pose = np.eye(4)
        self.keyframe_points_3d = None
        self.keyframe_descriptors = None
        
        # History for smoothing
        self.pose_history: deque = deque(maxlen=10)
        self.scale_history: deque = deque(maxlen=30)
        
        # Loop closure
        self.place_database: List[Place] = []
        self.dino_extractor = None
        if self.vo_config.enable_loop_closure:
            self.dino_extractor = DINOv2Extractor(config)
        
        self.frame_id = 0
    
    def process(self, frame: np.ndarray, 
                depth: Optional[np.ndarray] = None) -> VOState:
        """
        Process frame and update odometry.
        
        Args:
            frame: BGR image
            depth: Optional depth map (same size as frame)
            
        Returns:
            VOState with current pose estimate
        """
        self.frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        feature_result = self.feature_extractor.extract(frame)
        keypoints = feature_result.keypoints
        descriptors = feature_result.descriptors
        
        # Convert to cv2.KeyPoint format for compatibility
        kp_cv = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints]
        
        if len(keypoints) < self.vo_config.min_features:
            self.state.is_lost = True
            self.state.confidence = 0.0
            return self.state
        
        # First frame - initialize
        if self.prev_frame is None:
            self._initialize(gray, depth, kp_cv, descriptors)
            return self.state
        
        # Match features
        matches = self._match_features(self.prev_descriptors, descriptors)
        
        if len(matches) < self.vo_config.min_features:
            self.state.is_lost = True
            self.state.confidence = len(matches) / self.vo_config.min_features
            return self.state
        
        # Get matched points
        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp_cv[m.trainIdx].pt for m in matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.vo_config.ransac_threshold
        )
        
        if E is None:
            self.state.is_lost = True
            return self.state
        
        # Recover pose
        inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        self.state.num_inliers = inliers
        self.state.num_matches = len(matches)
        self.state.confidence = inliers / len(matches) if len(matches) > 0 else 0
        
        if inliers < self.vo_config.min_features // 2:
            self.state.is_lost = True
            return self.state
        
        # Get scale from depth
        scale = self._estimate_scale(pts1, pts2, mask_pose, self.prev_depth, depth, R, t)
        self.state.scale = scale
        
        # Apply scale
        t_scaled = t * scale
        
        # Build transformation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t_scaled.flatten()
        
        # Update pose
        self.state.pose = self.state.pose @ T
        self.state.update_from_pose()
        
        # Estimate velocity
        if len(self.pose_history) > 0:
            prev_pos = self.pose_history[-1][:3, 3]
            self.state.velocity = self.state.position - prev_pos
        
        self.pose_history.append(self.state.pose.copy())
        
        # Check for keyframe
        self.state.is_keyframe = self._check_keyframe()
        if self.state.is_keyframe:
            self._add_keyframe(gray, depth, kp_cv, descriptors)
            
            # Check loop closure
            if self.dino_extractor is not None:
                self.state.is_loop_closure = self._check_loop_closure(frame, kp_cv, descriptors)
        
        # Update previous
        self.prev_frame = gray
        self.prev_keypoints = kp_cv
        self.prev_descriptors = descriptors
        self.prev_depth = depth
        
        self.state.is_lost = False
        self.state.frame_id = self.frame_id
        
        return self.state
    
    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features with ratio test"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Choose matcher based on descriptor type
        if desc1.dtype == np.float32:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = self.bf
        
        try:
            matches = matcher.knnMatch(desc1, desc2, k=2)
        except Exception:
            return []
        
        # Ratio test
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        return good
    
    def _estimate_scale(self, pts1: np.ndarray, pts2: np.ndarray,
                       mask: np.ndarray, depth1: Optional[np.ndarray],
                       depth2: Optional[np.ndarray], R: np.ndarray, 
                       t: np.ndarray) -> float:
        """
        Estimate scale using triangulation vs measured depth.
        
        scale = median(depth_measured) / median(Z_triangulated)
        """
        if depth1 is None:
            if len(self.scale_history) > 0:
                return np.median(list(self.scale_history))
            return 1.0
        
        # Get inlier points
        inlier_mask = mask.ravel() > 0
        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]
        
        if len(pts1_in) < 10:
            return 1.0
        
        h, w = depth1.shape[:2]
        
        # Build projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        depths_measured = []
        depths_triangulated = []
        
        for p1, p2 in zip(pts1_in, pts2_in):
            x1, y1 = int(p1[0]), int(p1[1])
            
            if 0 <= x1 < w and 0 <= y1 < h:
                d_measured = depth1[y1, x1]
                
                if 0.1 < d_measured < 50:
                    # Triangulate
                    pt1_h = np.array([[p1[0]], [p1[1]]], dtype=np.float64)
                    pt2_h = np.array([[p2[0]], [p2[1]]], dtype=np.float64)
                    
                    point_4d = cv2.triangulatePoints(P1, P2, pt1_h, pt2_h)
                    point_3d = point_4d[:3] / point_4d[3]
                    z_triangulated = point_3d[2, 0]
                    
                    if 0.1 < z_triangulated < 100:
                        depths_measured.append(d_measured)
                        depths_triangulated.append(z_triangulated)
        
        if len(depths_measured) < 5:
            if len(self.scale_history) > 0:
                return np.median(list(self.scale_history))
            return 1.0
        
        # Compute scale
        scale = np.median(depths_measured) / np.median(depths_triangulated)
        scale = np.clip(scale, 0.01, 50.0)
        
        # Smooth
        self.scale_history.append(scale)
        return np.median(list(self.scale_history))
    
    def _check_keyframe(self) -> bool:
        """Check if current frame should be keyframe"""
        if self.keyframe_points_3d is None:
            return True
        
        rel_pose = np.linalg.inv(self.keyframe_pose) @ self.state.pose
        
        trans = np.linalg.norm(rel_pose[:3, 3])
        R = rel_pose[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle)
        
        return (trans > self.vo_config.keyframe_translation or 
                angle_deg > self.vo_config.keyframe_rotation)
    
    def _add_keyframe(self, gray, depth, keypoints, descriptors):
        """Add current frame as keyframe"""
        self.keyframe_pose = self.state.pose.copy()
        self.keyframe_descriptors = descriptors
        
        if depth is not None and keypoints is not None:
            points_3d = []
            h, w = depth.shape[:2]
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= x < w and 0 <= y < h:
                    d = depth[y, x]
                    if 0.1 < d < 100:
                        X = (x - self.K[0, 2]) * d / self.K[0, 0]
                        Y = (y - self.K[1, 2]) * d / self.K[1, 1]
                        points_3d.append([X, Y, d])
            
            self.keyframe_points_3d = np.array(points_3d) if points_3d else None
    
    @torch.inference_mode()
    def _check_loop_closure(self, frame: np.ndarray, keypoints, descriptors) -> bool:
        """Check for loop closure using DINOv2"""
        if self.dino_extractor is None or not self.dino_extractor.available:
            return False
        
        min_frames = 30
        
        if len(self.place_database) < min_frames:
            # Add to database
            descriptor = self.dino_extractor.extract_global(frame)
            if descriptor is not None:
                self.place_database.append(Place(
                    id=self.frame_id,
                    pose=self.state.pose.copy(),
                    descriptor=descriptor
                ))
            return False
        
        # Get current descriptor
        current_desc = self.dino_extractor.extract_global(frame)
        if current_desc is None:
            return False
        
        # Search for matches
        best_sim = 0.0
        best_place = None
        
        for place in self.place_database[:-min_frames]:
            sim = self.dino_extractor.compute_similarity(current_desc, place.descriptor)
            if sim > best_sim:
                best_sim = sim
                best_place = place
        
        # Add current to database
        self.place_database.append(Place(
            id=self.frame_id,
            pose=self.state.pose.copy(),
            descriptor=current_desc
        ))
        
        # Check threshold
        if best_sim > self.vo_config.loop_threshold and best_place is not None:
            print(f"[VO] Loop closure detected! Frame {self.frame_id} -> {best_place.id} "
                  f"(similarity: {best_sim:.3f})")
            return True
        
        return False
    
    def _initialize(self, gray, depth, keypoints, descriptors):
        """Initialize VO with first frame"""
        self.state = VOState()
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = depth
        self._add_keyframe(gray, depth, keypoints, descriptors)
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as Nx3 array"""
        if len(self.pose_history) == 0:
            return np.zeros((1, 3))
        return np.array([p[:3, 3] for p in self.pose_history])
    
    def reset(self):
        """Reset odometry"""
        self.state = VOState()
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None
        self.keyframe_pose = np.eye(4)
        self.keyframe_points_3d = None
        self.pose_history.clear()
        self.scale_history.clear()
        self.place_database.clear()
        self.frame_id = 0
