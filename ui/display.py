# UI Display Module
# Handles all visualization and overlay rendering

import cv2
import time
from typing import Optional, Tuple

from config import FONT, DRAW_THICKNESS, OVERLAY_TEXT


FONT = FONT


class DisplayManager:
    """Manages UI rendering and overlays."""
    
    def __init__(self, state):
        self.state = state
    
    def draw_primary_face_box(
        self,
        frame,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int]
    ):
        """Draw bounding box and label for primary face."""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, DRAW_THICKNESS)
        
        # Draw label background and text
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y2 + 4),
            (x1 + tw + 6, y2 + th + 10),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, y2 + th + 3),
            FONT,
            0.6,
            (255, 255, 255),
            2
        )
    
    def draw_secondary_faces(self, frame):
        """Draw overlays for secondary detected faces."""
        for (x1, y1, x2, y2, label, color) in self.state.last_secondary_overlays:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, DRAW_THICKNESS)
            
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            cv2.rectangle(
                frame,
                (x1, y2 + 4),
                (x1 + tw + 6, y2 + th + 10),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 3, y2 + th + 3),
                FONT,
                0.6,
                (255, 255, 255),
                2
            )
    
    def draw_debug_boxes(self, frame, boxes):
        """Draw thin red rectangles for debug face detections."""
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    def draw_stats(self, frame, fps: float):
        """Draw performance statistics."""
        y = 24
        
        def put_text(text):
            nonlocal y
            # Black outline
            cv2.putText(frame, text, (10, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            # White text
            cv2.putText(frame, text, (10, y), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        
        put_text(f"Detect+Recognize: {self.state.last_detect_recog_seconds:.3f} s")
        put_text(f"Auth latency: {self.state.last_auth_latency_seconds:.3f} s")
        put_text(f"FPS: {fps:.1f}")
        put_text(f"Heavy cadence: ~{self.state.detect_every_n} frames")
        
        # Enrollment hint
        cv2.putText(
            frame,
            "Press 'g' to enroll",
            (10, frame.shape[0] - 12),
            FONT,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    def draw_ui_overlays(self, frame):
        """Draw all UI overlays (countdown, banners, etc.)."""
        ui_state = self.state.get_ui_state()
        now = time.time()
        
        # Countdown overlay
        if ui_state["countdown_active"]:
            secs_left = int(max(0, round(ui_state["countdown_end"] - now)))
            
            if secs_left > 0:
                # Semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (overlay.shape[1], overlay.shape[0]),
                    (0, 0, 0),
                    -1
                )
                frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                
                # Countdown label
                label = ui_state["countdown_label"]
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
                cx = frame.shape[1] // 2 - tw // 2
                cy = frame.shape[0] // 2 - 80
                cv2.putText(
                    frame,
                    label,
                    (cx, cy),
                    FONT,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Countdown number
                num = str(secs_left)
                (tw2, th2), _ = cv2.getTextSize(num, FONT, 3.0, 8)
                cx2 = frame.shape[1] // 2 - tw2 // 2
                cy2 = frame.shape[0] // 2 + th2 // 2
                cv2.putText(
                    frame,
                    num,
                    (cx2, cy2),
                    FONT,
                    3.0,
                    (0, 255, 0),
                    8,
                    cv2.LINE_AA
                )
        
        # Flash capture overlay
        if ui_state["flash_capture_until"] > now:
            cv2.rectangle(
                frame,
                (4, 4),
                (frame.shape[1] - 4, frame.shape[0] - 4),
                (0, 255, 0),
                6
            )
            
            text = "Capturing..."
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 3)
            x = frame.shape[1] // 2 - tw // 2
            cv2.rectangle(
                frame,
                (x - 10, 45 - th - 10),
                (x + tw + 10, 45 + 10),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame,
                text,
                (x, 45),
                FONT,
                0.9,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )
        
        # Name prompt banner
        if ui_state["name_prompt_until"] > now:
            msg = "اكتب اسمك في نافذة الأوامر ثم اضغط Enter..."
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
            cv2.rectangle(
                frame,
                (10, 10),
                (10 + tw + 14, 10 + th + 20),
                (60, 60, 220),
                -1
            )
            cv2.putText(
                frame,
                msg,
                (17, 10 + th + 10),
                FONT,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        
        # General banner
        if ui_state["banner_until"] > now:
            msg = ui_state["banner_text"]
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
            x = frame.shape[1] // 2 - tw // 2
            y = 40
            cv2.rectangle(
                frame,
                (x - 10, y - th - 10),
                (x + tw + 10, y + 10),
                (0, 200, 0),
                -1
            )
            cv2.putText(
                frame,
                msg,
                (x, y),
                FONT,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
