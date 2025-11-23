# Face Tracking Module
# Tracks faces between frames using OpenCV trackers

import cv2
import numpy as np
from typing import Optional, Tuple

from config import TRACKER_TYPE


class FaceTracker:
    """
    Wrapper for OpenCV trackers to follow faces between heavy recognition runs.
    """
    
    def __init__(self, tracker_type: str = TRACKER_TYPE):
        self.tracker_type = tracker_type.upper()
        self.tracker = None
        self.bbox = None
        self._warned = False
        
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with a frame and bounding box.
        
        Args:
            frame: BGR image
            bbox: (x1, y1, x2, y2) format
            
        Returns:
            True if successful
        """
        self.tracker = self._create_tracker()
        
        if self.tracker is None:
            return False
        
        # Convert to (x, y, w, h) format for OpenCV
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        self.bbox = (x1, y1, w, h)
        
        try:
            self.tracker.init(frame, self.bbox)
            return True
        except Exception as e:
            print(f"[TRACKER] Init failed: {e}")
            self.tracker = None
            self.bbox = None
            return False
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Update tracker with new frame.
        
        Returns:
            (success, bbox) where bbox is (x1, y1, x2, y2) or None
        """
        if self.tracker is None or self.bbox is None:
            return False, None
        
        try:
            ok, tracked_bbox = self.tracker.update(frame)
            
            if ok:
                x, y, w, h = map(int, tracked_bbox)
                # Convert back to (x1, y1, x2, y2) format
                bbox = (x, y, x + w, y + h)
                return True, bbox
            else:
                return False, None
                
        except Exception as e:
            print(f"[TRACKER] Update failed: {e}")
            return False, None
    
    def reset(self):
        """Reset/clear the tracker."""
        self.tracker = None
        self.bbox = None
    
    def _create_tracker(self):
        """Create an OpenCV tracker instance."""
        kind = self.tracker_type
        
        # 1) Try legacy module if it really has the tracker
        if hasattr(cv2, "legacy"):
            if kind == "KCF" and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()
            if kind == "CSRT" and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
        
        # 2) Try non-legacy API if available
        if kind == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if kind == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        
        # 3) Fallbacks: MIL / MOSSE if present
        if hasattr(cv2, "TrackerMIL_create"):
            if not self._warned:
                print(f"[WARN] {kind} not available, falling back to MIL tracker.")
                self._warned = True
            return cv2.TrackerMIL_create()
        
        if hasattr(cv2, "TrackerMOSSE_create"):
            if not self._warned:
                print(f"[WARN] {kind} not available, falling back to MOSSE tracker.")
                self._warned = True
            return cv2.TrackerMOSSE_create()
        
        # 4) Nothing available → disable tracker
        if not self._warned:
            print("[WARN] No suitable OpenCV tracker found; tracker will be disabled.")
            self._warned = True
        return None


def ema(prev: Optional[float], new: float, alpha: float) -> float:
    """Exponential moving average for scalar values."""
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev


def ema_bbox(
    prev: Optional[Tuple[float, float, float, float]], 
    new: Tuple[int, int, int, int], 
    alpha: float
) -> Tuple[float, float, float, float]:
    """Exponential moving average for bounding boxes."""
    nx1, ny1, nx2, ny2 = map(float, new)
    
    if prev is None:
        return (nx1, ny1, nx2, ny2)
    
    px1, py1, px2, py2 = prev
    return (
        alpha * nx1 + (1 - alpha) * px1,
        alpha * ny1 + (1 - alpha) * py1,
        alpha * nx2 + (1 - alpha) * px2,
        alpha * ny2 + (1 - alpha) * py2
    )
