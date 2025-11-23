"""
HR Module - Face Recognition Attendance System
For HR personnel to enroll and manage employees.
No attendance tracking - only user management.
"""

import cv2
import time

from config import (
    CAM_INDEX,
    FRAME_DOWNSCALE,
    MIRROR_WEBCAM,
    DETECT_EVERY_N_BASE,
    HEAVY_MIN_PERIOD_SEC,
    NO_FACE_BACKOFF_MAX_N,
    NO_FACE_BACKOFF_STEP,
    ACCEPT_DIST_THRESH,
    REVOKE_DIST_THRESH,
    CAPTURE_TRIGGER_COOLDOWN_SEC,
    BBOX_SMOOTH_ALPHA,
    DIST_SMOOTH_ALPHA,
    COLOR_HOLD_FRAMES
)
from models import FaceDetector, FaceRecognizer, RecognitionWorker, FaceTracker, ema, ema_bbox
from database.db_manager import DatabaseManager
from services.tts_service import TTSService
from services.enrollment_service import EnrollmentService
from ui.display import DisplayManager
from utils.state import get_state


class HRModule:
    """HR Module for employee enrollment and management."""
    
    def __init__(self):
        print("=" * 60)
        print("HR MODULE - Employee Enrollment & Management")
        print("=" * 60)
        
        # State
        self.state = get_state()
        self.state.detect_every_n = DETECT_EVERY_N_BASE
        
        # Initialize components
        print("[INIT] Initializing face detector...")
        self.detector = FaceDetector()
        
        print("[INIT] Initializing face recognizer...")
        self.recognizer = FaceRecognizer()
        
        print("[INIT] Initializing database...")
        self.db_manager = DatabaseManager()
        
        print("[INIT] Initializing TTS...")
        self.tts = TTSService()
        
        print("[INIT] Initializing enrollment service...")
        self.enrollment = EnrollmentService(
            self.detector,
            self.recognizer,
            self.db_manager,
            self.tts,
            self.state
        )
        
        print("[INIT] Initializing tracker...")
        self.tracker = FaceTracker()
        
        print("[INIT] Initializing display...")
        self.display = DisplayManager(self.state)
        
        print("[INIT] Initializing recognition worker...")
        self.recognition_worker = RecognitionWorker(self.recognizer, self.detector)
        
        # Load database templates
        print("[INIT] Loading employee database...")
        self.state.set_db_templates(self.db_manager.get_all_templates())
        print(f"[INIT] Loaded {len(self.state.get_db_templates())} employees")
        
        # Camera
        print("[INIT] Opening camera...")
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        print("[INIT] HR Module ready!")
        print("\nControls:")
        print("  'g' - Enroll new employee")
        print("  'q' - Quit")
        print("=" * 60)
        
        self.tts.speak("وحدة الموارد البشرية جاهزة. اضغط جي لتسجيل موظف جديد.")
    
    def run(self):
        """Main application loop."""
        loop_start = time.perf_counter()
        frame_idx = 0
        
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[ERROR] Camera read failed")
                    break
                
                # Calculate FPS
                now = time.perf_counter()
                dt = now - loop_start
                loop_start = now
                fps = (1.0 / dt) if dt > 0 else 0.0
                
                # Prepare frame
                frame_disp = self._prepare_frame(frame)
                frame_idx += 1
                
                # Store frame for enrollment
                self.state.set_latest_frame(frame)
                
                # Process frame (detection and recognition)
                self._process_frame(frame_disp, frame_idx)
                
                # Draw overlays
                self._draw_overlays(frame_disp, fps)
                
                # Display
                cv2.imshow("HR Module - Employee Enrollment (press 'g' to enroll, 'q' to quit)", frame_disp)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame_idx):
                    break
                
                # Process pending enrollments
                self.enrollment.process_pending_enrollments()
                
                # Reload database if requested
                if self.state.check_and_clear_db_reload():
                    print("[DB] Reloading employee database...")
                    self.state.set_db_templates(self.db_manager.get_all_templates())
        
        finally:
            self._cleanup()
    
    def _prepare_frame(self, frame):
        """Prepare frame for display."""
        if FRAME_DOWNSCALE != 1.0:
            frame = cv2.resize(frame, None, fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)
        
        if MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)
        
        return frame
    
    def _process_frame(self, frame, frame_idx):
        """Process frame for face detection and recognition."""
        run_due_to_cadence = (frame_idx % self.state.detect_every_n) == 0
        run_due_to_period = (
            time.time() - self.state.last_heavy_submit_timestamp
        ) >= HEAVY_MIN_PERIOD_SEC
        
        if run_due_to_cadence and run_due_to_period:
            self._run_recognition(frame, frame_idx)
        else:
            self._run_tracker(frame)
    
    def _run_recognition(self, frame, frame_idx):
        """Run detection and recognition."""
        self.state.last_heavy_submit_timestamp = time.time()
        
        self.recognition_worker.submit(frame)
        results = self.recognition_worker.get_result()
        
        if not results:
            self._handle_no_faces(frame)
        else:
            self._handle_faces(results, frame)
    
    def _handle_no_faces(self, frame):
        """Handle no faces detected."""
        self.state.hold_counter = max(0, self.state.hold_counter - 1)
        self.state.no_face_cycles = min(self.state.no_face_cycles + 1, 50)
        
        self.state.detect_every_n = min(
            DETECT_EVERY_N_BASE + self.state.no_face_cycles * NO_FACE_BACKOFF_STEP,
            NO_FACE_BACKOFF_MAX_N
        )
        
        self.state.current_identity = None
        self.state.auth_streak = 0
    
    def _handle_faces(self, results, frame):
        """Handle detected faces - show employee names or 'Unknown'."""
        self.state.no_face_cycles = 0
        self.state.detect_every_n = DETECT_EVERY_N_BASE
        
        # Process primary face
        bbox, identity, dist_raw, is_auth_raw, elapsed = results[0]
        
        self.state.last_detect_recog_seconds = elapsed
        
        # Smooth distance and bbox
        self.state.smoothed_distance = ema(
            self.state.smoothed_distance,
            float(dist_raw),
            DIST_SMOOTH_ALPHA
        )
        
        x1, y1, x2, y2 = bbox
        self.state.smoothed_bbox = ema_bbox(
            self.state.smoothed_bbox,
            (x1, y1, x2, y2),
            BBOX_SMOOTH_ALPHA
        )
        x1, y1, x2, y2 = map(int, map(round, self.state.smoothed_bbox))
        
        # Determine if recognized
        is_recognized = (
            is_auth_raw
            and identity is not None
            and identity != "Unknown"
            and self.state.smoothed_distance is not None
            and self.state.smoothed_distance <= ACCEPT_DIST_THRESH
        )
        
        if is_recognized:
            color = (0, 200, 0)  # Green
            label = f"{identity} (Employee)"
        else:
            color = (255, 165, 0)  # Orange
            label = "Unknown - Press 'g' to enroll"
        
        # Hold logic
        if color != self.state.last_draw_color or label != self.state.last_draw_label:
            if self.state.hold_counter <= 0:
                self.state.last_draw_color = color
                self.state.last_draw_label = label
                self.state.hold_counter = COLOR_HOLD_FRAMES
        else:
            self.state.hold_counter = max(0, self.state.hold_counter - 1)
        
        # Draw
        self.display.draw_primary_face_box(
            frame,
            (x1, y1, x2, y2),
            self.state.last_draw_label,
            self.state.last_draw_color
        )
        
        # Update tracker
        self.tracker.init(frame, (x1, y1, x2, y2))
    
    def _run_tracker(self, frame):
        """Run tracker between recognition runs."""
        ok, bbox = self.tracker.update(frame)
        
        if ok and bbox is not None:
            self.state.smoothed_bbox = ema_bbox(
                self.state.smoothed_bbox,
                bbox,
                BBOX_SMOOTH_ALPHA
            )
            sx1, sy1, sx2, sy2 = map(int, map(round, self.state.smoothed_bbox))
            
            self.display.draw_primary_face_box(
                frame,
                (sx1, sy1, sx2, sy2),
                self.state.last_draw_label,
                self.state.last_draw_color
            )
            
            self.state.hold_counter = max(0, self.state.hold_counter - 1)
        else:
            self.tracker.reset()
    
    def _draw_overlays(self, frame, fps):
        """Draw UI overlays."""
        # Stats
        y = 24
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22
        cv2.putText(frame, f"Employees: {len(self.state.get_db_templates())}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Help text
        cv2.putText(
            frame,
            "Press 'g' to enroll new employee",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # UI overlays (banners, countdowns, etc.)
        self.display.draw_ui_overlays(frame)
    
    def _handle_key(self, key, frame_idx) -> bool:
        """Handle keyboard input."""
        if key == ord('q'):
            return False
        
        if key == ord('g'):
            self._handle_enrollment()
        
        return True
    
    def _handle_enrollment(self):
        """Handle enrollment trigger."""
        if self.state.is_capture_in_progress():
            print("[HR] Enrollment already in progress")
            self.tts.speak("عملية التسجيل قيد التنفيذ بالفعل.")
            return
        
        if self.enrollment.start_capture():
            self.state.last_capture_trigger_timestamp = time.time()
            print("[HR] Starting employee enrollment")
            self.tts.speak("بدء تسجيل موظف جديد.")
        else:
            print("[HR] Enrollment worker busy")
            self.tts.speak("لا يمكن بدء التسجيل الآن.")
    
    def _cleanup(self):
        """Cleanup resources."""
        print("[HR] Shutting down...")
        
        try:
            self.recognition_worker.shutdown()
        except Exception:
            pass
        
        try:
            self.tts.shutdown()
        except Exception:
            pass
        
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Entry point."""
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass
    
    module = HRModule()
    module.run()


if __name__ == "__main__":
    main()