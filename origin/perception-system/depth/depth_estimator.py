"""
Depth Estimator
===============

Multi-model depth estimation with fallbacks.

Model Priority:
1. Depth Anything V2 - Best quality, fast on GPU
2. MiDaS (v2.1 small) - Good quality fallback  
3. Gradient-based - CPU fallback when no models available
"""

import torch
import numpy as np
import cv2
from typing import Optional
import os

from core.config import Config, DepthConfig
from core.types import DepthResult

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DepthEstimator:
    """
    Monocular depth estimation with multiple model support.
    
    Usage:
        config = Config()
        estimator = DepthEstimator(config)
        
        result = estimator.estimate(image)
        print(f"Depth range: {result.min_depth:.2f} - {result.max_depth:.2f}m")
    """
    
    def __init__(self, config: Config):
        self.config = config.depth
        self.camera_config = config.camera
        
        self.model = None
        self.model_name = None
        self.available = False
        self.model_type = None  # 'depth_anything', 'midas', 'transformers'
        
        # Caching
        self.cache: Optional[DepthResult] = None
        self.frame_count = 0
        
        # MiDaS transform (if used)
        self._midas_transform = None
        
        # Transformers pipeline (if used)
        self._pipe = None
        
        self._load_model()
    
    def _load_model(self):
        """Load depth model with fallbacks"""
        
        # Try Depth Anything V2 via transformers (most reliable)
        if self._try_load_transformers():
            return
        
        # Try Depth Anything V2 via torch.hub
        if self._try_load_depth_anything():
            return
        
        # Try ZoeDepth (another option)
        if self._try_load_zoedepth():
            return
        
        # Try MiDaS v2.1 small (most compatible)
        if self._try_load_midas_v21():
            return
        
        # Try MiDaS DPT
        if self._try_load_midas_dpt():
            return
        
        # No model available - use fallback
        print("[Depth] ⚠ Using gradient-based fallback (install 'transformers' for real depth)")
        self.model_name = 'gradient-fallback'
        self.available = False
    
    def _try_load_transformers(self) -> bool:
        """Try loading Depth Anything via transformers library"""
        try:
            from transformers import pipeline
            
            model_id = f"depth-anything/Depth-Anything-V2-Small-hf"
            
            self._pipe = pipeline(
                task="depth-estimation",
                model=model_id,
                device=0 if DEVICE == 'cuda' else -1
            )
            
            self.model_name = 'DepthAnythingV2-transformers'
            self.model_type = 'transformers'
            self.available = True
            print(f"[Depth] ✓ {self.model_name} loaded on {DEVICE}")
            return True
        except Exception as e:
            print(f"[Depth] Transformers pipeline failed: {e}")
            return False
    
    def _try_load_depth_anything(self) -> bool:
        """Try loading Depth Anything V2 via torch.hub"""
        try:
            # Set environment to skip auth if possible
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '0'
            
            self.model = torch.hub.load(
                'LiheYoung/Depth-Anything-V2',
                f'depth_anything_v2_{self.config.model}',
                pretrained=True,
                trust_repo=True
            )
            self.model.to(DEVICE).eval()
            if DEVICE == 'cuda':
                self.model.half()
            
            self.model_name = f'DepthAnythingV2-{self.config.model}'
            self.model_type = 'depth_anything'
            self.available = True
            print(f"[Depth] ✓ {self.model_name} loaded on {DEVICE}")
            return True
        except Exception as e:
            print(f"[Depth] Depth-Anything-V2 failed: {e}")
            return False
    
    def _try_load_zoedepth(self) -> bool:
        """Try loading ZoeDepth (another depth option)"""
        try:
            self.model = torch.hub.load(
                'isl-org/ZoeDepth',
                'ZoeD_NK',
                pretrained=True,
                trust_repo=True
            )
            self.model.to(DEVICE).eval()
            
            self.model_name = 'ZoeDepth'
            self.model_type = 'zoedepth'
            self.available = True
            print(f"[Depth] ✓ {self.model_name} loaded on {DEVICE}")
            return True
        except Exception as e:
            print(f"[Depth] ZoeDepth failed: {e}")
            return False
    
    def _try_load_midas_v21(self) -> bool:
        """Try loading MiDaS v2.1 small (most compatible)"""
        try:
            # Suppress timm import issues
            import warnings
            warnings.filterwarnings('ignore')
            
            # Try loading just the model weights directly
            self.model = torch.hub.load(
                'intel-isl/MiDaS', 
                'MiDaS_small',
                trust_repo=True,
                skip_validation=True
            )
            self.model.to(DEVICE).eval()
            
            # Simple transform for MiDaS small
            self._midas_transform = self._create_simple_midas_transform(256)
            
            self.model_name = 'MiDaS-small'
            self.model_type = 'midas'
            self.available = True
            print(f"[Depth] ✓ {self.model_name} loaded on {DEVICE}")
            return True
        except Exception as e:
            print(f"[Depth] MiDaS small failed: {e}")
            return False
    
    def _try_load_midas_dpt(self) -> bool:
        """Try loading MiDaS DPT models with custom transforms"""
        try:
            import warnings
            warnings.filterwarnings('ignore')
            
            model_type = {
                'small': 'DPT_Hybrid',
                'base': 'DPT_Hybrid', 
                'large': 'DPT_Large'
            }.get(self.config.model, 'DPT_Hybrid')
            
            self.model = torch.hub.load(
                'intel-isl/MiDaS', 
                model_type, 
                trust_repo=True,
                skip_validation=True
            )
            self.model.to(DEVICE).eval()
            
            # Use simple custom transform instead of hub transforms
            input_size = 384 if model_type != 'DPT_Large' else 384
            self._midas_transform = self._create_simple_midas_transform(input_size)
            
            self.model_name = f'MiDaS-{model_type}'
            self.model_type = 'midas'
            self.available = True
            print(f"[Depth] ✓ {self.model_name} loaded on {DEVICE}")
            return True
        except Exception as e:
            print(f"[Depth] MiDaS DPT failed: {e}")
            return False
    
    def _create_simple_midas_transform(self, input_size: int = 384):
        """Create a simple transform for MiDaS without timm dependency"""
        def transform(img):
            # Resize 
            img_resized = cv2.resize(img, (input_size, input_size))
            # Normalize with ImageNet stats
            img_norm = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_norm = (img_norm - mean) / std
            # To tensor [1, 3, H, W]
            tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)
            return tensor.to(DEVICE)
        return transform
    
    @torch.inference_mode()
    def estimate(self, image: np.ndarray, use_cache: bool = True) -> DepthResult:
        """
        Estimate depth from single image.
        
        Args:
            image: BGR image (H, W, 3)
            use_cache: Return cached result if within skip_frames
            
        Returns:
            DepthResult with depth map in meters
        """
        self.frame_count += 1
        
        # Check cache
        if use_cache and self.config.skip_frames > 1:
            if self.frame_count % self.config.skip_frames != 0 and self.cache is not None:
                return self.cache
        
        if not self.available:
            result = self._fallback_depth(image)
            self.cache = result
            return result
        
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            if self.model_type == 'transformers':
                depth_raw = self._estimate_transformers(rgb)
            elif self.model_type == 'depth_anything':
                depth_raw = self._estimate_depth_anything(rgb)
            elif self.model_type == 'zoedepth':
                depth_raw = self._estimate_zoedepth(rgb)
            elif self.model_type == 'midas':
                depth_raw = self._estimate_midas(rgb)
            else:
                return self._fallback_depth(image)
            
            # Resize to original
            if depth_raw.shape != (h, w):
                depth_raw = cv2.resize(depth_raw, (w, h))
            
            # Convert to metric depth
            depth = self._to_metric_depth(depth_raw)
            
            # Create result
            result = DepthResult(
                depth=depth,
                colored=self._colorize(depth),
                min_depth=float(np.percentile(depth, 5)),
                max_depth=float(np.percentile(depth, 95)),
                valid=True
            )
            
            self.cache = result
            return result
            
        except Exception as e:
            print(f"[Depth] Estimation error: {e}, using fallback")
            return self._fallback_depth(image)
    
    def _estimate_transformers(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth using transformers pipeline"""
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        result = self._pipe(pil_image)
        return np.array(result['depth'])
    
    def _estimate_depth_anything(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth using Depth Anything V2"""
        input_size = 518
        img_resized = cv2.resize(rgb, (input_size, input_size))
        
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        if DEVICE == 'cuda':
            img_tensor = img_tensor.half()
        
        return self.model(img_tensor).squeeze().cpu().float().numpy()
    
    def _estimate_midas(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS"""
        input_batch = self._midas_transform(rgb)
        prediction = self.model(input_batch)
        return prediction.squeeze().cpu().numpy()
    
    def _estimate_zoedepth(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth using ZoeDepth"""
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        depth = self.model.infer_pil(pil_image)
        return np.array(depth)
    
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
    
    def _colorize(self, depth: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
        """Create colored visualization of depth map"""
        # Clip outliers for better visualization
        d_min = np.percentile(depth, 2)
        d_max = np.percentile(depth, 98)
        
        depth_clipped = np.clip(depth, d_min, d_max)
        depth_norm = ((depth_clipped - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        
        # Invert so closer = warmer colors
        return cv2.applyColorMap(255 - depth_norm, colormap)
    
    def _fallback_depth(self, image: np.ndarray) -> DepthResult:
        """Gradient-based depth estimation fallback"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Sobel instead of Laplacian (more compatible)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Blur for smooth depth-like map
        depth_raw = cv2.GaussianBlur(gradient, (31, 31), 0)
        
        # Create simple depth gradient based on vertical position
        # (lower in frame = closer, typical for ground-level camera)
        y_gradient = np.linspace(0, 1, h).reshape(-1, 1).repeat(w, axis=1)
        depth_raw = depth_raw + y_gradient * depth_raw.max() * 0.5
        
        # Invert: high gradient = closer (textured objects)
        depth_raw = depth_raw.max() - depth_raw + 1.0
        
        # Normalize to metric range
        depth = self._to_metric_depth(depth_raw.astype(np.float32))
        
        return DepthResult(
            depth=depth,
            colored=self._colorize(depth),
            min_depth=float(depth.min()),
            max_depth=float(depth.max()),
            valid=True
        )
    
    def get_point_cloud(self, depth_result: DepthResult, 
                       image: np.ndarray,
                       subsample: int = 4) -> tuple:
        """
        Convert depth map to colored point cloud.
        
        Args:
            depth_result: DepthResult from estimate()
            image: Original BGR image
            subsample: Subsampling factor for performance
            
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
        """
        depth = depth_result.depth
        h, w = depth.shape
        K = self.camera_config.K
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create pixel grid
        u, v = np.meshgrid(
            np.arange(0, w, subsample),
            np.arange(0, h, subsample)
        )
        u = u.flatten().astype(np.float64)
        v = v.flatten().astype(np.float64)
        z = depth[::subsample, ::subsample].flatten()
        
        # Filter valid depths
        valid = (z > self.config.min_depth) & (z < self.config.max_depth)
        u, v, z = u[valid], v[valid], z[valid]
        
        # Backproject to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=1)
        
        # Get colors
        u_idx = (u / subsample).astype(int)
        v_idx = (v / subsample).astype(int)
        
        img_subsampled = image[::subsample, ::subsample]
        colors = img_subsampled[v_idx, u_idx][:, ::-1]  # BGR to RGB
        
        return points, colors
    
    def reset(self):
        """Reset estimator state"""
        self.cache = None
        self.frame_count = 0
