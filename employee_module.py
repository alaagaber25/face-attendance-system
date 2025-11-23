"""
Employee Module - Face Recognition Attendance System
For employees to login/logout with automatic time tracking.
No enrollment capability - read-only user database.
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
    BBOX_SMOOTH_ALPHA,
    DIST_SMOOTH_ALPHA,
    COLOR_HOLD_FRAMES
)
from models import FaceDetector, FaceRecognizer, RecognitionWorker, FaceTracker, ema, ema_bbox
from database.db_manager import DatabaseManager
from services.tts_service import TTSService
from services.login_logout_service import LoginLogoutService
from ui.display import DisplayManager
from utils.state import get_state


class EmployeeModule:
    """Employee Module for attendance tracking (login/logout)."""
    
    def __init__(self):
        print("=" * 60)
        print("EMPLOYEE MODULE - Attendance Tracking")
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
        
        print("[INIT] Initializing login/logout service...")
        self.login_logout = LoginLogoutService(
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
        
        print("[INIT] Employee Module ready!")
        print("\nInstructions:")
        print("  - Stand in front of camera to login/logout")
        print("  - System will automatically track your attendance")
        print("  - Press 'q' to quit")
        print("=" * 60)
        
        self.tts.speak("نظام الحضور جاهز. قف أمام الكاميرا لتسجيل الدخول أو الخروج.")
    
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
                
                # Process frame
                self._process_frame(frame_disp, frame_idx)
                
                # Draw overlays
                self._draw_overlays(frame_disp, fps)
                
                # Display
                cv2.imshow("Employee Module - Attendance Tracking (press 'q' to quit)", frame_disp)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
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
        """Process frame for attendance tracking."""
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
        self.state.last_secondary_overlays = []
    
    def _handle_faces(self, results, frame):
        """Handle detected faces - process login/logout."""
        self.state.no_face_cycles = 0
        self.state.detect_every_n = DETECT_EVERY_N_BASE
        
        # Primary face
        primary = results[0]
        self._process_primary_face(primary, frame)
        
        # Secondary faces
        secondary_faces = results[1:] if len(results) > 1 else []
        self._process_secondary_faces(secondary_faces)
    
    def _process_primary_face(self, result, frame):
        """Process primary detected face."""
        bbox, identity, dist_raw, is_auth_raw, elapsed = result
        
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
        
        # Determine authorization
        is_authorized = self._determine_authorization(identity, is_auth_raw, dist_raw)
        
        # Update state
        if is_authorized and identity != "Unknown":
            if identity == self.state.current_identity:
                self.state.auth_streak += 1
            else:
                self.state.current_identity = identity
                self.state.auth_streak = 1
        else:
            self.state.current_identity = None
            self.state.auth_streak = 0
        
        # Display properties
        color, label = self._get_display_properties(is_authorized, identity)
        
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
        
        # Process login/logout
        self.login_logout.process_recognition(
            identity,
            is_authorized,
            self.state.smoothed_distance
        )
        
        # Update tracker
        self.tracker.init(frame, (x1, y1, x2, y2))
    
    def _determine_authorization(self, identity, is_auth_raw, dist_raw) -> bool:
        """Determine if face is authorized."""
        raw_ok = bool(is_auth_raw) and identity is not None and identity != "Unknown"
        
        if self.state.current_identity == identity and self.state.current_identity is not None:
            # Current identity: use revoke threshold
            return (
                raw_ok
                and self.state.smoothed_distance is not None
                and self.state.smoothed_distance <= REVOKE_DIST_THRESH
            )
        else:
            # New identity: use accept threshold
            return (
                raw_ok
                and self.state.smoothed_distance is not None
                and self.state.smoothed_distance <= ACCEPT_DIST_THRESH
            )
    
    def _get_display_properties(self, is_authorized, identity) -> tuple:
        """Get display color and label."""
        if is_authorized:
            color = (0, 200, 0)  # Green
            label = identity
        else:
            color = (0, 0, 255)  # Red
            if self.state.smoothed_distance is not None:
                label = f"NOT AUTHORIZED d={self.state.smoothed_distance:.2f}"
            else:
                label = "NOT AUTHORIZED"
        
        return color, label
    
    def _process_secondary_faces(self, secondary_faces):
        """Process secondary detected faces."""
        self.state.last_secondary_overlays = []
        seen_ids = set()
        
        for (bbox, identity, distance, is_auth, _) in secondary_faces:
            if identity is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            is_authorized = (
                is_auth
                and identity != "Unknown"
                and distance is not None
                and distance <= ACCEPT_DIST_THRESH
            )
            
            color = (0, 200, 0) if is_authorized else (0, 0, 255)
            label = identity if is_authorized else f"NOT AUTH d={distance:.2f}"
            
            self.state.last_secondary_overlays.append((x1, y1, x2, y2, label, color))
            
            # Process login/logout for secondary face
            self.login_logout.process_recognition(identity, is_authorized, distance)
            seen_ids.add(identity)
        
        # Reset streaks for unseen identities
        self.state.reset_streak_for_unseen_ids(seen_ids)
    
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
            
            # Draw secondary overlays
            self.display.draw_secondary_faces(frame)
            
            self.state.hold_counter = max(0, self.state.hold_counter - 1)
        else:
            self.tracker.reset()
            self.state.last_secondary_overlays = []
    
    def _draw_overlays(self, frame, fps):
        """Draw UI overlays."""
        # Stats
        y = 24
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22
        cv2.putText(frame, f"Recognition: {self.state.last_detect_recog_seconds:.3f}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(
            frame,
            "Stand in front of camera to login/logout",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # UI overlays (banners, etc.)
        self.display.draw_ui_overlays(frame)
    
    def _cleanup(self):
        """Cleanup resources."""
        print("[EMPLOYEE] Shutting down...")
        
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
    
    module = EmployeeModule()
    module.run()


if __name__ == "__main__":
    main()