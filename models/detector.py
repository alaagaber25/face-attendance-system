# Face Detection Module
# Abstracts face detection - easy to swap cascade for YOLO, MediaPipe, etc.

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import MIN_BBOX_AREA, MIN_LAPLACE_VAR


class FaceDetector:
    """
    Abstract face detection. Currently uses Haar cascades,
    but can be easily replaced with YOLO, MediaPipe, RetinaFace, etc.
    """
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades
        print(f"[DEBUG] Haar cascades path: {cascade_path}")
        
        frontal_xml = cascade_path + "haarcascade_frontalface_default.xml"
        profile_xml = cascade_path + "haarcascade_profileface.xml"
        
        print(f"[DEBUG] Loading frontal cascade from: {frontal_xml}")
        self.frontal_cascade = cv2.CascadeClassifier(frontal_xml)
        print(f"[DEBUG] Frontal cascade empty: {self.frontal_cascade.empty()}")
        
        print(f"[DEBUG] Loading profile cascade from: {profile_xml}")
        self.profile_cascade = cv2.CascadeClassifier(profile_xml)
        print(f"[DEBUG] Profile cascade empty: {self.profile_cascade.empty()}")
        
    def detect_largest_face(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in the image.
        Returns: (x1, y1, x2, y2) or None
        """
        if bgr is None or bgr.size == 0:
            return None
            
        h, w = bgr.shape[:2]
        if w < 32 or h < 32:
            return None
            
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
            
        min_size = self._dynamic_min_size(w, h)
        
        # Try frontal detection first
        @staticmethod
        def _detect_with_cascade(gray, cascade, min_size):
            """Run cascade detection with error handling."""
            try:
                h, w = gray.shape[:2]
                
                # Validate image and min_size
                if w < min_size[0] or h < min_size[1]:
                    print(f"[DETECT] Image {w}x{h} smaller than min_size {min_size}, skipping")
                    return ()
                
                # Ensure min_size is not too large (max 1/4 of image size)
                safe_min_w = min(min_size[0], w // 4)
                safe_min_h = min(min_size[1], h // 4)
                safe_min_size = (max(safe_min_w, 20), max(safe_min_h, 20))
                
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=5,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    minSize=safe_min_size
                )
                return faces
            except Exception as e:
                print(f"[DETECT] error: {e}")
                return ()
        
        # Try profile detection
        faces = self._detect_with_cascade(gray, self.profile_cascade, min_size)
        if len(faces):
            return self._bbox_from_faces(faces)
        
        # Try flipped profile (for left-facing profiles)
        gray_flipped = cv2.flip(gray, 1)
        faces = self._detect_with_cascade(gray_flipped, self.profile_cascade, min_size)
        if len(faces):
            x, y, ww, hh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            x1 = w - (x + ww)
            y1 = y
            x2 = w - x
            y2 = y + hh
            return (x1, y1, x2, y2)
        
        return None
    
    def detect_multiple_faces(
        self, 
        bgr: np.ndarray, 
        max_faces: int = 5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect up to max_faces faces.
        Returns: List of (x1, y1, x2, y2) bounding boxes
        """
        if bgr is None or bgr.size == 0:
            return []
            
        h, w = bgr.shape[:2]
        if w < 32 or h < 32:
            return []
            
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"[DEBUG] Color conversion failed: {e}")
            return []
            
        min_size = self._dynamic_min_size(w, h)
        print(f"[DEBUG] Image size: {w}x{h}, min_size: {min_size}")
        
        boxes: List[Tuple[int, int, int, int]] = []
        
        # Main: frontal faces
        faces = self._detect_with_cascade(gray, self.frontal_cascade, min_size)
        print(f"[DEBUG] Frontal cascade found {len(faces)} faces")
        for (x, y, ww, hh) in faces:
            boxes.append((x, y, x + ww, y + hh))
        
        # Fallback: if no frontal faces, use single-face logic
        if not boxes:
            print(f"[DEBUG] No frontal faces, trying fallback detection")
            one = self.detect_largest_face(bgr)
            if one is not None:
                boxes.append(one)
                print(f"[DEBUG] Fallback found 1 face")
        
        # Sort by area (largest first) and limit
        boxes = sorted(
            boxes, 
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), 
            reverse=True
        )
        
        if max_faces > 0 and len(boxes) > max_faces:
            boxes = boxes[:max_faces]
        
        print(f"[DEBUG] Returning {len(boxes)} boxes")
        return boxes
    
    def detect_for_stage(
        self, 
        bgr: np.ndarray, 
        stage_name: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Stage-aware detection for enrollment (front/right_side/left_side).
        Returns: (x1, y1, x2, y2) or None
        """
        if bgr is None or bgr.size == 0:
            return None
            
        h, w = bgr.shape[:2]
        if w < 32 or h < 32:
            return None
            
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
            
        min_size = self._dynamic_min_size(w, h)
        
        if stage_name == "front":
            faces = self._detect_with_cascade(gray, self.frontal_cascade, min_size)
            if len(faces):
                return self._bbox_from_faces(faces)
            return self.detect_largest_face(bgr)
        
        if stage_name == "right_side":
            faces = self._detect_with_cascade(gray, self.profile_cascade, min_size)
            if len(faces):
                return self._bbox_from_faces(faces)
            return self.detect_largest_face(bgr)
        
        if stage_name == "left_side":
            gray_flipped = cv2.flip(gray, 1)
            faces = self._detect_with_cascade(gray_flipped, self.profile_cascade, min_size)
            if len(faces):
                x, y, ww, hh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                x1 = w - (x + ww)
                y1 = y
                x2 = w - x
                y2 = y + hh
                return (x1, y1, x2, y2)
            return self.detect_largest_face(bgr)
        
        return self.detect_largest_face(bgr)
    
    @staticmethod
    def is_crop_usable(bgr_crop: np.ndarray) -> bool:
        """Check if a face crop is usable (quality checks)."""
        if bgr_crop is None or bgr_crop.size == 0:
            print(f"[DEBUG QUALITY] FAIL: crop is None or empty")
            return False
            
        h, w = bgr_crop.shape[:2]
        area = w * h
        print(f"[DEBUG QUALITY] Crop size: {w}x{h}, area={area}, min={MIN_BBOX_AREA}")
        
        if area < MIN_BBOX_AREA:
            print(f"[DEBUG QUALITY] FAIL: area {area} < {MIN_BBOX_AREA}")
            return False
            
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        laplace_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"[DEBUG QUALITY] Laplace variance: {laplace_var:.2f}, min={MIN_LAPLACE_VAR}")
        
        if laplace_var < MIN_LAPLACE_VAR:
            print(f"[DEBUG QUALITY] FAIL: blur {laplace_var:.2f} < {MIN_LAPLACE_VAR}")
            return False
        
        print(f"[DEBUG QUALITY] PASS: all checks passed")
        return True
    
    # ========== PRIVATE METHODS ==========
    
    @staticmethod
    def _dynamic_min_size(w: int, h: int) -> Tuple[int, int]:
        """Calculate dynamic minimum face size based on image dimensions."""
        m = max(32, int(min(w, h) * 0.15))
        m = min(m, max(32, min(w, h)))
        return (m, m)
    
    @staticmethod
    def _detect_with_cascade(gray, cascade, min_size):
        """Run cascade detection with error handling."""
        try:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=min_size
            )
            return faces
        except Exception as e:
            print(f"[DETECT] error: {e}")
            return ()
    
    @staticmethod
    def _bbox_from_faces(faces):
        """Extract the largest bounding box from detected faces."""
        if len(faces) == 0:
            return None
        x, y, ww, hh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        return (x, y, x + ww, y + hh)