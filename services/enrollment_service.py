# Enrollment Service Module
# Handles multi-stage face capture for new user enrollment

import cv2
import time
import threading
import shutil
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import (
    AUTHORIZED_DIR,
    DIRECT_ENROLL_TO_AUTH,
    NEW_USER_PREFIX,
    STAGE_LIST,
    PRE_CAPTURE_COOLDOWN_SEC,
    STAGE_COOLDOWN_SEC,
    STAGE_TIMEOUT_PER_STAGE,
    CAPTURE_IMAGE_INTERVAL,
    NAME_PROMPT_TIMEOUT_SEC,
    STORE_IMAGES_IN_DB
)
from database.db_manager import DatabaseManager
from services.tts_service import TTSService
from utils.state import ApplicationState


class EnrollmentService:
    """
    Service for enrolling new users with multi-stage face capture.
    """
    
    def __init__(
        self,
        detector,
        recognizer,
        db_manager: DatabaseManager,
        tts_service: TTSService,
        state: ApplicationState
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.db = db_manager
        self.tts = tts_service
        self.state = state
        self._worker_thread = None
    
    def start_capture(self) -> bool:
        """
        Start the capture process in a background thread.
        
        Returns:
            True if capture started, False if already in progress
        """
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return False
        
        self._worker_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True
        )
        self._worker_thread.start()
        return True
    
    def _capture_worker(self):
        """Background worker for multi-stage capture."""
        try:
            # Set flag
            self.state.set_capture_in_progress(True)
            
            # Determine output directory
            base_dir = AUTHORIZED_DIR if DIRECT_ENROLL_TO_AUTH else Path("staging_enroll")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique folder name
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_name = ("_pending_" if DIRECT_ENROLL_TO_AUTH else "") + NEW_USER_PREFIX + ts
            target = base_dir / ts_name
            
            counter = 0
            while target.exists():
                counter += 1
                target = base_dir / f"{ts_name}_{counter}"
            
            target.mkdir(parents=True, exist_ok=True)
            print(f"[ENROLLMENT] Saving to: {target}")
            
            # Preparation phase
            self.tts.speak("جارٍ بدء التقاط صور المستخدم الجديد، من فضلك انتظر.")
            time.sleep(PRE_CAPTURE_COOLDOWN_SEC)
            
            total_saved = 0
            
            # Multi-stage capture
            for idx, (stage_name, stage_count, stage_prompt) in enumerate(STAGE_LIST, start=1):
                # Announce stage
                self.tts.speak(stage_prompt)
                self.state.start_countdown(f"{stage_name.replace('_', ' ').title()} in", 3)
                time.sleep(3.2)
                self.state.stop_countdown()
                
                saved = 0
                last_save_time = 0.0
                deadline = time.time() + STAGE_TIMEOUT_PER_STAGE
                
                # Capture images for this stage
                while saved < stage_count and time.time() < deadline:
                    frame = self.state.get_latest_frame()
                    
                    if frame is None or frame.size == 0:
                        time.sleep(0.05)
                        continue
                    
                    # Detect face for this stage
                    bbox = self.detector.detect_for_stage(frame, stage_name)
                    if bbox is None:
                        time.sleep(0.04)
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    h, w = frame.shape[:2]
                    
                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        time.sleep(0.04)
                        continue
                    
                    # Rate limiting
                    now = time.time()
                    if (now - last_save_time) < CAPTURE_IMAGE_INTERVAL:
                        time.sleep(0.03)
                        continue
                    
                    # Expand crop with padding
                    pad = 0.12
                    dx = int((x2 - x1) * pad)
                    dy = int((y2 - y1) * pad)
                    x1p = max(0, x1 - dx)
                    y1p = max(0, y1 - dy)
                    x2p = min(w - 1, x2 + dx)
                    y2p = min(h - 1, y2 + dy)
                    
                    crop = frame[y1p:y2p, x1p:x2p]
                    
                    if crop is None or crop.size == 0:
                        time.sleep(0.03)
                        continue
                    
                    # Save image
                    output_path = target / f"{target.name}_{stage_name}_{saved + 1:02d}.jpg"
                    self.state.flash_capture(0.6)
                    
                    ok = cv2.imwrite(str(output_path), crop)
                    print(f"[ENROLLMENT] [{stage_name}] Saved {output_path.name} ok={ok}")
                    
                    if ok:
                        saved += 1
                        total_saved += 1
                        last_save_time = now
                
                # Cooldown between stages
                if idx < len(STAGE_LIST):
                    time.sleep(STAGE_COOLDOWN_SEC)
            
            # Check if any images were saved
            if total_saved == 0:
                try:
                    shutil.rmtree(target)
                except Exception:
                    pass
                print("[ENROLLMENT] No images saved.")
                self.tts.speak("لم يتم حفظ أي صور. سيتم إلغاء التسجيل.")
                return
            
            # Add to staging queue
            self.state.enqueue_staging(target)
            print(f"[ENROLLMENT] Capture complete: {target} (saved={total_saved})")
            self.tts.speak("تم الانتهاء من التقاط الصور. من فضلك أدخل اسمك في نافذة الأوامر.")
            
        except Exception as e:
            print(f"[ENROLLMENT] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.state.set_capture_in_progress(False)
    
    def finalize_enrollment(self, folder_path: Path):
        """
        Finalize an enrollment by extracting embeddings and adding to database.
        
        Args:
            folder_path: Path to the folder containing captured images
        """
        if folder_path is None or not folder_path.exists():
            return
        
        print(f"[ENROLLMENT] Finalizing: {folder_path}")
        
        # Extract embedding from folder
        embedding = self._extract_embedding_from_folder(folder_path)
        
        if embedding is None:
            print("[ENROLLMENT] No usable embedding; removing folder")
            try:
                shutil.rmtree(folder_path)
            except Exception:
                pass
            self.tts.speak("لم يتم العثور على وجه مناسب في الصور. سيتم إلغاء التسجيل.")
            return
        
        # Check for duplicates
        duplicate_result = self._check_duplicate(embedding)
        if duplicate_result is not None:
            name, distance = duplicate_result
            print(f"[ENROLLMENT] Duplicate detected: '{name}' (distance={distance:.3f})")
            
            from config import ACCEPT_DIST_THRESH
            if distance <= ACCEPT_DIST_THRESH:
                try:
                    shutil.rmtree(folder_path)
                except Exception:
                    pass
                self.tts.speak("هذا الوجه مسجل بالفعل، لن يتم إنشاء مستخدم جديد.")
                return
        
        # Prompt for name
        default_name = folder_path.name.replace("_pending_", "")
        final_name = self._prompt_for_name(default_name)
        
        # Add to database
        try:
            person_id = self.db.add_person(final_name)
            embeddings_added = 0
            
            # Process all images
                        # Move folder to final location FIRST
            final_folder = AUTHORIZED_DIR / final_name
            if final_folder.exists():
                print(f"[ENROLLMENT] Removing existing folder: {final_folder}")
                shutil.rmtree(final_folder)

            # Rename pending folder to person's name
            print(f"[ENROLLMENT] Moving {folder_path} -> {final_folder}")
            folder_path.rename(final_folder)

            # Process all images from the new location
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                for img_path in sorted(final_folder.glob(ext)):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    bbox = self.detector.detect_largest_face(img)
                    if bbox is None:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    crop = img[y1:y2, x1:x2]
                    
                    # Relaxed check for enrollment - only size
                    h, w = crop.shape[:2]
                    if (w * h) < 2500:
                        continue
                    
                    emb = self.recognizer.extract_embedding(crop)
                    if emb is None:
                        continue
                    
                    self.db.add_template(person_id, emb)
                    embeddings_added += 1
                    
                    # Store image path reference in database
                    match = re.search(
                        r"_(front|right_side|left_side)_\d+\.(jpg|jpeg|png|webp|bmp)$",
                        img_path.name,
                        re.I
                    )
                    stage = match.group(1) if match else "unknown"
                    
                    # Save path reference to database
                    self.db.add_image(person_id, stage, str(img_path))
                    
                    from config import MAX_TEMPLATES_PER_ID
                    if embeddings_added >= MAX_TEMPLATES_PER_ID:
                        break
                
                if embeddings_added >= MAX_TEMPLATES_PER_ID:
                    break
            
            
            print(f"[ENROLLMENT] Finalized: '{final_name}' (templates: {embeddings_added})")
            
            # Log new user
            self.db.log_new_user(final_name, person_id)
            
        except Exception as e:
            print(f"[ENROLLMENT] Database error: {e}")
        
        # Clean up folder
        # Folder already moved to final location - no cleanup needed
        print(f"[ENROLLMENT] Images saved to: {final_folder}")
        
        # Request DB reload
        self.state.request_db_reload()
    
    def process_pending_enrollments(self):
        """Process all pending enrollments in the queue."""
        while self.state.has_pending_staging():
            folder = self.state.dequeue_staging()
            if folder is not None:
                self.finalize_enrollment(folder)
    
    # ========== PRIVATE METHODS ==========
    
    def _extract_embedding_from_folder(self, folder: Path) -> Optional:
        """Extract the first usable embedding from a folder of images."""
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            for img_path in sorted(folder.glob(ext)):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                bbox = self.detector.detect_largest_face(img)
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = bbox
                crop = img[y1:y2, x1:x2]
                
                # RELAXED quality check for enrollment
                h, w = crop.shape[:2]
                if (w * h) < 2500:  # Only check size, ignore blur
                    continue
                
                e = self.recognizer.extract_embedding(crop)
                if e is not None:
                    return e
        
        return None
    
    def _check_duplicate(self, embedding) -> Optional[tuple]:
        """Check if embedding matches an existing identity."""
        best_name = None
        best_dist = 1e9
        
        for name, templates in self.state.get_db_templates().items():
            if not templates:
                continue
            
            dist = min(
                self.recognizer.cosine_distance(embedding, template)
                for template in templates
            )
            
            if dist < best_dist:
                best_dist = dist
                best_name = name
        
        return (best_name, best_dist) if best_name is not None else None
    
    def _prompt_for_name(self, default_name: str) -> str:
        """Prompt user for a name with timeout."""
        self.state.show_name_prompt_banner(NAME_PROMPT_TIMEOUT_SEC)
        self.tts.speak("اكتب اسمك في نافذة الأوامر ثم اضغط إنتر.")
        
        # Timed input
        entered = self._timed_input(
            f"New user name (default '{default_name}', {NAME_PROMPT_TIMEOUT_SEC:.0f}s): ",
            NAME_PROMPT_TIMEOUT_SEC,
            default_name
        )
        
        # Sanitize name
        entered = re.sub(r"\s+", "_", entered)
        entered = re.sub(r"[^A-Za-z0-9_\-]", "", entered)[:64]
        
        return entered or default_name
    
    @staticmethod
    def _timed_input(prompt: str, timeout: float, default: str) -> str:
        """Get input with timeout."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        
        result = {"text": default}
        
        def reader():
            try:
                txt = input().strip()
                if txt:
                    result["text"] = txt
            except Exception:
                pass
        
        if not sys.stdin or not sys.stdin.isatty():
            print("")
            return default
        
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            print("")
        
        return result["text"]
