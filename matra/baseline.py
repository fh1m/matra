"""Unchanged archived method bodies; complete source modules live in origin/."""
from typing import Optional
import cv2
import numpy as np

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

def _to_metric_depth(self, depth_raw: np.ndarray) -> np.ndarray:
    """Convert relative depth to metric depth (approximate)"""
    # Normalize to [0, 1]
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min > 1e-8:
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_raw)

    # Map to metric range
    metric_depth = self.config.min_depth + depth_norm * (
        self.config.max_depth - self.config.min_depth
    )

    # Apply scale factor
    metric_depth *= self.config.scale_factor

    return metric_depth.astype(np.float32)
