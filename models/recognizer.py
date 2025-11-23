# Face Recognition Module
# Abstracts face recognition/embedding - easy to swap DeepFace for other models

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple

# Try to import DeepFace
try:
    from deepface import DeepFace
except Exception as e:
    DeepFace = None
    print(f"[WARN] DeepFace is not available: {e}")

from config import (
    MODEL_NAME, 
    MAX_TEMPLATES_PER_ID,
    ACCEPT_DIST_THRESH,
    TOP2_MARGIN_MIN,
    TOP2_RATIO_MAX
)


class FaceRecognizer:
    """
    Abstract face recognition. Currently uses DeepFace,
    but can be easily replaced with FaceNet, ArcFace, InsightFace, etc.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        if DeepFace is None:
            raise RuntimeError(
                "DeepFace not installed. Please install: pip install deepface"
            )
        
        self.model_name = model_name
        self.model = self._build_model()
        
    def _build_model(self):
        """Build/load the recognition model."""
        return DeepFace.build_model(self.model_name)
    
    def extract_embedding(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a BGR image crop.
        Returns: Normalized embedding vector or None
        """
        try:
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            reps = DeepFace.represent(
                img_path=rgb,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False
            )
            
            if not reps:
                return None
                
            embedding = np.array(reps[0]["embedding"], dtype=np.float32)
            return self._l2_normalize(embedding)
            
        except Exception as e:
            print(f"[WARN] Embedding extraction failed: {e}")
            return None
    
    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance between two embeddings."""
        a = FaceRecognizer._l2_normalize(a)
        b = FaceRecognizer._l2_normalize(b)
        return float(1.0 - np.dot(a, b))
    
    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        return v / (np.linalg.norm(v) + 1e-9)
    
    def match_identity(
        self, 
        embedding: np.ndarray, 
        database: Dict[str, List[np.ndarray]]
    ) -> Optional[Tuple[str, float, float]]:
        """
        Match an embedding against a database of known identities.
        
        Args:
            embedding: Query embedding
            database: Dict mapping identity names to list of template embeddings
            
        Returns:
            (best_name, best_distance, second_best_distance) or None
        """
        pairs = []
        
        for name, templates in database.items():
            if not templates:
                continue
                
            # Find minimum distance to any template for this identity
            min_dist = min(
                self.cosine_distance(embedding, template) 
                for template in templates
            )
            pairs.append((name, min_dist))
        
        if not pairs:
            return None
        
        # Sort by distance (ascending)
        pairs.sort(key=lambda t: t[1])
        
        best_name, best_dist = pairs[0]
        second_dist = pairs[1][1] if len(pairs) > 1 else 1.0
        
        return best_name, best_dist, second_dist
    
    def is_confident_match(
        self, 
        best_dist: float, 
        second_dist: float
    ) -> bool:
        """
        Determine if a match is confident based on distance thresholds.
        """
        clear_margin = (second_dist - best_dist) >= TOP2_MARGIN_MIN
        clear_ratio = (best_dist / max(second_dist, 1e-9)) <= TOP2_RATIO_MAX
        accept = best_dist <= ACCEPT_DIST_THRESH
        
        return accept and clear_margin and clear_ratio


class RecognitionWorker:
    """
    Threaded recognition worker for non-blocking recognition.
    """
    
    def __init__(self, recognizer: FaceRecognizer, detector):
        self.recognizer = recognizer
        self.detector = detector
        self.lock = threading.Lock()
        self.busy = False
        self.frame = None
        self.result = None
        self.stop_flag = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def submit(self, frame: np.ndarray) -> bool:
        """
        Submit a frame for recognition.
        Returns: True if accepted, False if worker is busy
        """
        if self.busy:
            return False
            
        with self.lock:
            self.frame = frame.copy()
            self.busy = True
            
        return True
    
    def get_result(self):
        """Get the latest recognition result."""
        return self.result
    
    def _loop(self):
        """Main worker loop."""
        while not self.stop_flag:
            frame_to_process = None
            
            with self.lock:
                if self.frame is not None:
                    frame_to_process = self.frame
                    self.frame = None
            
            if frame_to_process is None:
                time.sleep(0.01)
                continue
            
            try:
                self.result = self._recognize_multi_face(frame_to_process)
            except Exception as e:
                print(f"[WARN] Recognition worker failed: {e}")
                self.result = None
            finally:
                with self.lock:
                    self.busy = False
    
    def _recognize_multi_face(
        self, 
        bgr: np.ndarray, 
        database: Dict[str, List[np.ndarray]] = None,
        max_faces: int = 5
    ) -> List[Tuple[Tuple[int, int, int, int], str, float, bool, float]]:
        """
        Multi-face recognition.
        
        Returns:
            List of (bbox, identity, distance, is_confident, elapsed_time)
        """
        if database is None:
            # Import here to avoid circular dependency
            from utils.state import get_db_templates
            database = get_db_templates()
        
        t0 = time.perf_counter()
        
        # Detect faces
        boxes = self.detector.detect_multiple_faces(bgr, max_faces=max_faces)
        print(f"[DEBUG RECOG] Detected {len(boxes)} boxes")
        if not boxes:
            return []
        
        results = []
        
        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            crop = bgr[y1:y2, x1:x2]
            
            print(f"[DEBUG RECOG] Face {i+1}: bbox={bbox}, crop_size={crop.shape if crop.size else 'empty'}")
            
            # Quality check
            if not self.detector.is_crop_usable(crop):
                print(f"[DEBUG RECOG] Face {i+1}: FAILED quality check")
                continue
            
            print(f"[DEBUG RECOG] Face {i+1}: PASSED quality check")
            
            # Extract embedding
            embedding = self.recognizer.extract_embedding(crop)
            if embedding is None:
                print(f"[DEBUG RECOG] Face {i+1}: FAILED embedding extraction")
                continue
            
            print(f"[DEBUG RECOG] Face {i+1}: PASSED embedding extraction, shape={embedding.shape}")
            
            # Match against database
            match_result = self.recognizer.match_identity(embedding, database)
            if match_result is None:
                print(f"[DEBUG RECOG] Face {i+1}: No match (empty database), marking as Unknown")
                # No database or no match - mark as Unknown with high distance
                best_name = "Unknown"
                best_dist = 1.0
                second_dist = 1.0
                is_confident = False
            else:
                best_name, best_dist, second_dist = match_result
                # Check confidence
                is_confident = self.recognizer.is_confident_match(best_dist, second_dist)
                print(f"[DEBUG RECOG] Face {i+1}: Match='{best_name}', dist={best_dist:.3f}, confident={is_confident}")
            
            elapsed = time.perf_counter() - t0
            results.append((bbox, best_name, best_dist, is_confident, elapsed))
        
        print(f"[DEBUG RECOG] Returning {len(results)} results")
        return results
    
    def shutdown(self):
        """Shutdown the worker thread."""
        self.stop_flag = True
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass