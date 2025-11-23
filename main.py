# Main Application
# Orchestrates all components and runs the main loop

import cv2
import time
from datetime import datetime

import config
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
    CAPTURE_ONLY_WHEN_UNAUTHORIZED,
    CAPTURE_MIN_DIST_FOR_NEW,
    CAPTURE_SUPPRESS_AFTER_AUTH_SEC,
    BBOX_SMOOTH_ALPHA,
    DIST_SMOOTH_ALPHA,
    COLOR_HOLD_FRAMES
)
from models import FaceDetector, FaceRecognizer, RecognitionWorker, FaceTracker, ema, ema_bbox
from database.db_manager import DatabaseManager
from services.tts_service import TTSService
from services.attendance_service import AttendanceService
from services.enrollment_service import EnrollmentService
from ui.display import DisplayManager
from utils.state import get_state


class FaceAttendanceSystem:
    """Main application class."""
    
    def __init__(self):
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
        
        print("[INIT] Initializing services...")
        self.attendance = AttendanceService(self.db_manager, self.tts, self.state)
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
        print("[INIT] Loading templates from database...")
        self.state.set_db_templates(self.db_manager.get_all_templates())
        print(f"[INIT] Loaded {len(self.state.get_db_templates())} identities")
        
        # Camera
        print("[INIT] Opening camera...")
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        print("[INIT] System ready!")
        self.tts.speak("النظام يعمل الآن. يمكنك الضغط على حرف جي من لوحة المفاتيح لتسجيل مستخدم جديد.")
    
    def run(self):
        """Main application loop."""
        loop_start = time.perf_counter()
        frame_idx = 0
        
        try:
            while True:
                # Read frame
                ok, frame = self.cap.read()
                if not ok:
                    print("[ERROR] Camera read failed")
                    break
                
                # Calculate FPS
                now = time.perf_counter()
                dt = now - loop_start
                loop_start = now
                fps = (1.0 / dt) if dt > 0 else 0.0
                
                # Downscale frame for processing
                frame_disp = self._prepare_frame(frame)
                frame_idx += 1
                
                # Store frame for enrollment
                self.state.set_latest_frame(frame)
                
                # Process frame
                self._process_frame(frame_disp, frame_idx)
                
                # Draw overlays
                self._draw_overlays(frame_disp, fps)
                
                # Display
                cv2.imshow("Face Attendance System (press 'g' to enroll, 'q' to quit)", frame_disp)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame_idx):
                    break
                
                # Process pending enrollments
                self.enrollment.process_pending_enrollments()
                
                # Reload database if requested
                if self.state.check_and_clear_db_reload():
                    print("[DB] Reloading templates...")
                    self.state.set_db_templates(self.db_manager.get_all_templates())
        
        finally:
            self._cleanup()
    
    def _prepare_frame(self, frame):
        """Prepare frame for display (downscale and optionally mirror)."""
        if FRAME_DOWNSCALE != 1.0:
            frame = cv2.resize(frame, None, fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)
        
        if MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)
        
        return frame
    
    def _process_frame(self, frame, frame_idx):
        """Process a frame (detection and recognition)."""
        # Determine if we should run heavy recognition
        run_due_to_cadence = (frame_idx % self.state.detect_every_n) == 0
        run_due_to_period = (
            time.time() - self.state.last_heavy_submit_timestamp
        ) >= HEAVY_MIN_PERIOD_SEC
        
        if run_due_to_cadence and run_due_to_period:
            self._run_recognition(frame, frame_idx)
        else:
            self._run_tracker(frame)
    
    def _run_recognition(self, frame, frame_idx):
        """Run heavy detection and recognition."""
        print(f"[DEBUG] Running recognition on frame {frame_idx}")
        self.state.last_heavy_submit_timestamp = time.time()
        
        # Submit frame for recognition
        submitted = self.recognition_worker.submit(frame)
        print(f"[DEBUG] Frame submitted: {submitted}")
        results = self.recognition_worker.get_result()
        print(f"[DEBUG] Recognition results: {results}")
        
        if not results:
            # No faces detected
            self._handle_no_faces_detected(frame, frame_idx)
        else:
            # Faces detected
            self._handle_faces_detected(results, frame, frame_idx)
    
    def _handle_no_faces_detected(self, frame, frame_idx):
        """Handle case when no faces are detected."""
        print(f"[DEBUG] No faces detected in recognition worker")
        self.state.hold_counter = max(0, self.state.hold_counter - 1)
        self.state.no_face_cycles = min(self.state.no_face_cycles + 1, 50)
        
        # Backoff detection frequency
        self.state.detect_every_n = min(
            DETECT_EVERY_N_BASE + self.state.no_face_cycles * NO_FACE_BACKOFF_STEP,
            NO_FACE_BACKOFF_MAX_N
        )
        
        self.state.last_secondary_overlays = []
        self.state.current_identity = None
        self.state.auth_streak = 0
        
        # Draw debug boxes for raw detections
        debug_boxes = self.detector.detect_multiple_faces(frame, max_faces=5)
        print(f"[DEBUG] Debug boxes (fallback): {len(debug_boxes)}")
        for (x1, y1, x2, y2) in debug_boxes:
            label = "NOT AUTHORIZED"
            color = (0, 0, 255)
            self.display.draw_primary_face_box(frame, (x1, y1, x2, y2), label, color)
    
    def _handle_faces_detected(self, results, frame, frame_idx):
        """Handle case when faces are detected and recognized."""
        self.state.no_face_cycles = 0
        self.state.detect_every_n = DETECT_EVERY_N_BASE
        
        # Select primary face (prefer current identity if present)
        primary_idx = self._select_primary_face(results)
        primary = results[primary_idx]
        secondary_faces = [results[i] for i in range(len(results)) if i != primary_idx]
        
        # Process primary face
        self._process_primary_face(primary, frame)
        
        # Process secondary faces
        self._process_secondary_faces(secondary_faces)
    
    def _select_primary_face(self, results) -> int:
        """Select which face should be the primary focus."""
        if self.state.current_identity is not None:
            for i, (_, identity, _, _, _) in enumerate(results):
                if identity == self.state.current_identity:
                    return i
        return 0  # Default to first face
    
    def _process_primary_face(self, result, frame):
        """Process the primary detected face."""
        bbox, identity, dist_raw, is_auth_raw, elapsed = result
        
        self.state.last_detect_recog_seconds = elapsed
        
        # Smooth distance
        self.state.smoothed_distance = ema(
            self.state.smoothed_distance,
            float(dist_raw),
            DIST_SMOOTH_ALPHA
        )
        
        # Smooth bbox
        x1, y1, x2, y2 = bbox
        self.state.smoothed_bbox = ema_bbox(
            self.state.smoothed_bbox,
            (x1, y1, x2, y2),
            BBOX_SMOOTH_ALPHA
        )
        x1, y1, x2, y2 = map(int, map(round, self.state.smoothed_bbox))
        
        # Determine authorization
        is_authorized = self._determine_authorization(identity, is_auth_raw, dist_raw)
        
        # Update identity and streak
        if is_authorized and identity != "Unknown":
            if identity == self.state.current_identity:
                self.state.auth_streak += 1
            else:
                self.state.current_identity = identity
                self.state.auth_streak = 1
        else:
            self.state.current_identity = None
            self.state.auth_streak = 0
        
        # Determine display properties
        color, label = self._get_display_properties(is_authorized, identity)
        
        # Hold logic (smooth color/label transitions)
        if color != self.state.last_draw_color or label != self.state.last_draw_label:
            if self.state.hold_counter <= 0:
                self.state.last_draw_color = color
                self.state.last_draw_label = label
                self.state.hold_counter = config.COLOR_HOLD_FRAMES
        else:
            self.state.hold_counter = max(0, self.state.hold_counter - 1)
        
        # Draw primary box
        self.display.draw_primary_face_box(
            frame,
            (x1, y1, x2, y2),
            self.state.last_draw_label,
            self.state.last_draw_color
        )
        
        # Process attendance
        self.attendance.process_recognition_result(
            identity,
            is_authorized,
            self.state.smoothed_distance
        )
        
        # Re-initialize tracker
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
        """Get display color and label based on authorization status."""
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
        """Process and display secondary detected faces."""
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
            
            # Process attendance for secondary face
            self.attendance.process_recognition_result(identity, is_authorized, distance)
            seen_ids.add(identity)
        
        # Reset streaks for unseen identities
        self.state.reset_streak_for_unseen_ids(seen_ids)
    
    def _run_tracker(self, frame):
        """Run tracker to follow face between recognition runs."""
        ok, bbox = self.tracker.update(frame)
        
        if ok and bbox is not None:
            # Smooth bbox
            self.state.smoothed_bbox = ema_bbox(
                self.state.smoothed_bbox,
                bbox,
                BBOX_SMOOTH_ALPHA
            )
            sx1, sy1, sx2, sy2 = map(int, map(round, self.state.smoothed_bbox))
            
            # Draw tracked box
            self.display.draw_primary_face_box(
                frame,
                (sx1, sy1, sx2, sy2),
                self.state.last_draw_label,
                self.state.last_draw_color
            )
            
            # Draw secondary overlays from last recognition
            self.display.draw_secondary_faces(frame)
            
            self.state.hold_counter = max(0, self.state.hold_counter - 1)
        else:
            # Tracker lost - reset
            self.tracker.reset()
            self.state.last_secondary_overlays = []
    
    def _draw_overlays(self, frame, fps):
        """Draw all UI overlays."""
        self.display.draw_stats(frame, fps)
        self.display.draw_ui_overlays(frame)
    
    def _handle_key(self, key, frame_idx) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            False to quit, True to continue
        """
        if key == ord('q'):
            return False
        
        if key == ord('g'):
            self._handle_enrollment_trigger(frame_idx)
        
        return True
    
    def _handle_enrollment_trigger(self, frame_idx):
        """Handle enrollment trigger ('g' key press)."""
        if self.state.is_capture_in_progress():
            print("[ENROLLMENT] Already in progress")
            self.tts.speak("عملية التسجيل قيد التنفيذ بالفعل.")
            return
        
        # Check if enrollment is allowed
        allowed, reason = self._check_enrollment_allowed()
        
        if allowed:
            if self.enrollment.start_capture():
                self.state.last_capture_trigger_timestamp = time.time()
                self.state.last_capture_frame_index = frame_idx
                print("[ENROLLMENT] Started by 'g' key")
                self.tts.speak("تم الضغط على جي. سيتم الآن تسجيل مستخدم جديد.")
            else:
                print("[ENROLLMENT] Worker busy")
                self.tts.speak("لا يمكن بدء التسجيل الآن. عملية سابقة ما زالت قيد التنفيذ.")
        else:
            print(f"[ENROLLMENT] Blocked: {reason}")
            self._speak_enrollment_blocked_reason(reason)
    
    def _check_enrollment_allowed(self) -> tuple:
        """
        Check if enrollment is currently allowed.
        
        Returns:
            (is_allowed, reason_string)
        """
        now = time.time()
        
        if self.state.is_capture_in_progress():
            return False, "capture already in progress"
        
        if (now - self.state.last_capture_trigger_timestamp) < CAPTURE_TRIGGER_COOLDOWN_SEC:
            return False, "global capture cooldown"
        
        if self.state.current_identity is not None:
            return False, "face already exists in DB (named match)"
        
        if (self.state.smoothed_distance is not None and 
            self.state.smoothed_distance <= ACCEPT_DIST_THRESH):
            return False, f"face already exists in DB (d={self.state.smoothed_distance:.3f})"
        
        if CAPTURE_ONLY_WHEN_UNAUTHORIZED:
            if (now - self.state.last_auth_seen_timestamp) < CAPTURE_SUPPRESS_AFTER_AUTH_SEC:
                return False, "recent authorization suppression"
            
            if (self.state.smoothed_distance is not None and 
                self.state.smoothed_distance < CAPTURE_MIN_DIST_FOR_NEW):
                return False, "face too similar to existing identities"
        
        return True, "ok"
    
    def _speak_enrollment_blocked_reason(self, reason):
        """Speak appropriate message for enrollment blocking reason."""
        if "face too similar" in reason:
            self.tts.speak("لا يمكن إنشاء مستخدم جديد الآن. الوجه مشابه جداً لِوجه موجود.")
        elif "recent authorization" in reason:
            self.tts.speak("تم التعرف عليك منذ قليل. انتظر قليلاً قبل محاولة التسجيل مرة أخرى.")
        elif "currently authenticated" in reason or "already exists" in reason:
            self.tts.speak("هذا الوجه مسجل بالفعل كمستخدم معتمد.")
        elif "cooldown" in reason:
            self.tts.speak("من فضلك انتظر قليلاً قبل محاولة التسجيل مرة أخرى.")
        else:
            self.tts.speak("لا يمكن بدء التسجيل في الوقت الحالي.")
    
    def _cleanup(self):
        """Cleanup resources."""
        print("[CLEANUP] Shutting down...")
        
        try:
            self.recognition_worker.shutdown()
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
    
    app = FaceAttendanceSystem()
    app.run()


if __name__ == "__main__":
    main()