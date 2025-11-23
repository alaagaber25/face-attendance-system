# Model Swapping Guide

This guide shows exactly how to replace the detection and recognition models with popular alternatives.

## 🎯 Quick Reference

| Component | Current | Easy Alternatives |
|-----------|---------|-------------------|
| **Detector** | Haar Cascades | YOLO, RetinaFace, MediaPipe, MTCNN, Dlib |
| **Recognizer** | DeepFace (Facenet) | InsightFace, FaceNet, ArcFace, CosFace |
| **Tracker** | KCF/CSRT | SORT, DeepSORT, ByteTrack |

## 1. Replace Face Detector

### Option A: YOLO (YOLOv8-Face)

**Why**: Faster and more accurate than cascades

```python
# models/detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self):
        # Load YOLO face detection model
        self.model = YOLO('yolov8n-face.pt')
        self.conf_threshold = 0.5
    
    def detect_largest_face(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the largest face using YOLO."""
        results = self.model(bgr, conf=self.conf_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Get all boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Find largest by area
        areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
        largest_idx = np.argmax(areas)
        
        x1, y1, x2, y2 = map(int, boxes[largest_idx])
        return (x1, y1, x2, y2)
    
    def detect_multiple_faces(self, bgr: np.ndarray, max_faces: int = 5) -> List[Tuple[int, int, int, int]]:
        """Detect multiple faces using YOLO."""
        results = self.model(bgr, conf=self.conf_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Convert to integer tuples
        boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]
        
        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        
        return boxes[:max_faces]
    
    def detect_for_stage(self, bgr: np.ndarray, stage_name: str) -> Optional[Tuple[int, int, int, int]]:
        """Stage-aware detection (YOLO handles all angles)."""
        return self.detect_largest_face(bgr)
    
    @staticmethod
    def is_crop_usable(bgr_crop: np.ndarray) -> bool:
        """Quality check for face crops."""
        if bgr_crop is None or bgr_crop.size == 0:
            return False
        
        h, w = bgr_crop.shape[:2]
        if (w * h) < 70 * 70:  # Minimum size
            return False
        
        # Blur check
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 40.0:
            return False
        
        return True
```

**Installation**:
```bash
pip install ultralytics
# Download model: yolov8n-face.pt (from GitHub or Ultralytics)
```

### Option B: RetinaFace

**Why**: Very accurate, handles small faces well

```python
# models/detector.py

from retinaface import RetinaFace
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self):
        # RetinaFace initialized on first use
        self.initialized = False
    
    def detect_largest_face(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the largest face using RetinaFace."""
        faces = RetinaFace.detect_faces(bgr)
        
        if not isinstance(faces, dict) or len(faces) == 0:
            return None
        
        # RetinaFace returns dict: {'face_1': {...}, 'face_2': {...}}
        largest_face = None
        largest_area = 0
        
        for key, face_data in faces.items():
            bbox = face_data['facial_area']  # [x1, y1, x2, y2]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if area > largest_area:
                largest_area = area
                largest_face = bbox
        
        if largest_face is None:
            return None
        
        return tuple(map(int, largest_face))
    
    def detect_multiple_faces(self, bgr: np.ndarray, max_faces: int = 5) -> List[Tuple[int, int, int, int]]:
        """Detect multiple faces using RetinaFace."""
        faces = RetinaFace.detect_faces(bgr)
        
        if not isinstance(faces, dict) or len(faces) == 0:
            return []
        
        boxes = []
        for key, face_data in faces.items():
            bbox = face_data['facial_area']
            boxes.append(tuple(map(int, bbox)))
        
        # Sort by area
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        
        return boxes[:max_faces]
    
    # ... same is_crop_usable, detect_for_stage as YOLO example
```

**Installation**:
```bash
pip install retina-face
```

### Option C: MediaPipe

**Why**: Fast, runs on CPU, good for real-time

```python
# models/detector.py

import mediapipe as mp
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short range, 1=full range
            min_detection_confidence=0.5
        )
    
    def detect_largest_face(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the largest face using MediaPipe."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            return None
        
        h, w = bgr.shape[:2]
        largest_bbox = None
        largest_area = 0
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative to absolute coordinates
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_bbox = (x1, y1, x2, y2)
        
        return largest_bbox
    
    # ... similar implementation for detect_multiple_faces
```

**Installation**:
```bash
pip install mediapipe
```

## 2. Replace Face Recognizer

### Option A: InsightFace

**Why**: State-of-the-art accuracy, fast inference

```python
# models/recognizer.py

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from insightface.app import FaceAnalysis

from ..config import MAX_TEMPLATES_PER_ID, ACCEPT_DIST_THRESH, TOP2_MARGIN_MIN, TOP2_RATIO_MAX

class FaceRecognizer:
    def __init__(self, model_name: str = "buffalo_l"):
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def extract_embedding(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using InsightFace."""
        try:
            faces = self.app.get(bgr_crop)
            
            if len(faces) == 0:
                return None
            
            # Use the first (largest) face
            embedding = faces[0].embedding
            
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
        """Match an embedding against a database of known identities."""
        pairs = []
        
        for name, templates in database.items():
            if not templates:
                continue
            
            min_dist = min(
                self.cosine_distance(embedding, template) 
                for template in templates
            )
            pairs.append((name, min_dist))
        
        if not pairs:
            return None
        
        pairs.sort(key=lambda t: t[1])
        
        best_name, best_dist = pairs[0]
        second_dist = pairs[1][1] if len(pairs) > 1 else 1.0
        
        return best_name, best_dist, second_dist
    
    def is_confident_match(self, best_dist: float, second_dist: float) -> bool:
        """Determine if a match is confident based on distance thresholds."""
        clear_margin = (second_dist - best_dist) >= TOP2_MARGIN_MIN
        clear_ratio = (best_dist / max(second_dist, 1e-9)) <= TOP2_RATIO_MAX
        accept = best_dist <= ACCEPT_DIST_THRESH
        
        return accept and clear_margin and clear_ratio
```

**Installation**:
```bash
pip install insightface onnxruntime
```

### Option B: FaceNet (Pure Implementation)

**Why**: Well-established, good balance of speed/accuracy

```python
# models/recognizer.py

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from keras_facenet import FaceNet

class FaceRecognizer:
    def __init__(self):
        self.model = FaceNet()
        self.target_size = (160, 160)  # FaceNet input size
    
    def extract_embedding(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using FaceNet."""
        try:
            # Resize to FaceNet input size
            face = cv2.resize(bgr_crop, self.target_size)
            
            # Convert BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Normalize
            face = (face - 127.5) / 128.0
            
            # Add batch dimension
            face = np.expand_dims(face, axis=0)
            
            # Get embedding
            embedding = self.model.embeddings(face)[0]
            
            return self._l2_normalize(embedding)
            
        except Exception as e:
            print(f"[WARN] Embedding extraction failed: {e}")
            return None
    
    # ... same cosine_distance, match_identity, etc. as before
```

**Installation**:
```bash
pip install keras-facenet
```

### Option C: ArcFace

**Why**: High accuracy, especially for large datasets

```python
# models/recognizer.py

import cv2
import numpy as np
import onnxruntime
from typing import Dict, List, Optional, Tuple

class FaceRecognizer:
    def __init__(self, model_path: str = "arcface_r100_v1.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (112, 112)
    
    def extract_embedding(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using ArcFace."""
        try:
            # Resize to model input size
            face = cv2.resize(bgr_crop, self.input_size)
            
            # Preprocess
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, axis=0)
            face = face.astype(np.float32)
            
            # Normalize
            face = (face - 127.5) / 128.0
            
            # Run inference
            embedding = self.session.run(
                [self.output_name],
                {self.input_name: face}
            )[0][0]
            
            return self._l2_normalize(embedding)
            
        except Exception as e:
            print(f"[WARN] Embedding extraction failed: {e}")
            return None
    
    # ... same methods as before
```

**Installation**:
```bash
pip install onnxruntime
# Download model: arcface_r100_v1.onnx
```

## 3. Configuration Updates

After swapping models, update `config.py`:

```python
# config.py

# ============= MODEL SELECTION =============
DETECTOR_BACKEND = "yolo"  # Options: "haar", "yolo", "retinaface", "mediapipe"
RECOGNIZER_BACKEND = "insightface"  # Options: "deepface", "insightface", "facenet", "arcface"
TRACKER_TYPE = "KCF"  # No change needed

# ============= MODEL-SPECIFIC SETTINGS =============
if RECOGNIZER_BACKEND == "insightface":
    MODEL_NAME = "buffalo_l"
    EMBEDDING_DIM = 512
elif RECOGNIZER_BACKEND == "facenet":
    MODEL_NAME = "facenet"
    EMBEDDING_DIM = 512
elif RECOGNIZER_BACKEND == "deepface":
    MODEL_NAME = "Facenet"
    EMBEDDING_DIM = 128

# ============= THRESHOLDS =============
# May need to adjust based on model
if RECOGNIZER_BACKEND == "insightface":
    ACCEPT_DIST_THRESH = 0.40  # InsightFace is more confident
    REVOKE_DIST_THRESH = 0.50
else:
    ACCEPT_DIST_THRESH = 0.35
    REVOKE_DIST_THRESH = 0.45
```

## 4. Testing After Model Swap

```python
# test_new_models.py

import cv2
from models import FaceDetector, FaceRecognizer

def test_detector():
    """Test new detector."""
    detector = FaceDetector()
    
    # Load test image
    img = cv2.imread("test_face.jpg")
    
    # Test single face detection
    bbox = detector.detect_largest_face(img)
    print(f"Detected face: {bbox}")
    
    # Test multiple faces
    faces = detector.detect_multiple_faces(img, max_faces=5)
    print(f"Detected {len(faces)} faces")
    
    # Visualize
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Detection Test", img)
    cv2.waitKey(0)

def test_recognizer():
    """Test new recognizer."""
    recognizer = FaceRecognizer()
    
    # Load test face crop
    face_crop = cv2.imread("face_crop.jpg")
    
    # Extract embedding
    embedding = recognizer.extract_embedding(face_crop)
    
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print("✓ Recognition test passed")
    else:
        print("✗ Recognition test failed")

if __name__ == "__main__":
    print("Testing detector...")
    test_detector()
    
    print("\nTesting recognizer...")
    test_recognizer()
```

## 5. Performance Comparison

| Model | Speed (FPS) | Accuracy | Notes |
|-------|-------------|----------|-------|
| **Detection** ||||
| Haar Cascade | ~30 | Good | Fast but less accurate |
| YOLO v8 | ~25 | Excellent | Best balance |
| RetinaFace | ~15 | Excellent | Most accurate |
| MediaPipe | ~40 | Good | Fastest |
| **Recognition** ||||
| DeepFace (Facenet) | ~20 | Good | Original |
| InsightFace | ~30 | Excellent | Recommended |
| FaceNet (pure) | ~25 | Good | Lightweight |
| ArcFace | ~20 | Excellent | Large datasets |

## Summary

The modular structure makes model swapping straightforward:

1. **Choose your model** from the options above
2. **Replace the class** in `models/detector.py` or `models/recognizer.py`
3. **Update config.py** with model-specific settings
4. **Test** with `test_new_models.py`
5. **Adjust thresholds** if needed in `config.py`

All without touching the rest of the codebase! 🎉
