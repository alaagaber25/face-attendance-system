# Attendance Service Module
# Handles attendance logging logic and business rules

from datetime import datetime
from typing import Optional

from database.db_manager import DatabaseManager
from services.tts_service import TTSService
from utils.state import ApplicationState


class AttendanceService:
    """
    Service for managing attendance logging with business rules.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        tts_service: TTSService,
        state: ApplicationState
    ):
        self.db = db_manager
        self.tts = tts_service
        self.state = state
    
    def log_event(
        self, 
        name: str, 
        status: str, 
        distance: Optional[float]
    ) -> bool:
        """
        Log an attendance event (Arrival or Departure).
        
        Args:
            name: Person's name
            status: "Arrival" or "Departure"
            distance: Recognition distance (for tracking accuracy)
            
        Returns:
            True if logged successfully, False if already logged today
        """
        # Try to log in database
        success = self.db.log_attendance(name, status, distance)
        
        if success:
            self.state.show_banner(f"Logged: {name} → {status}")
            print(f"[ATTENDANCE] Logged: {name} → {status}")
            
            # TTS feedback
            if status == "Arrival":
                self.tts.speak(f"تم تسجيل الحضور لِـ {name}")
                self.tts.speak(f"مرحباً يا {name}")
            else:
                self.tts.speak(f"تم تسجيل الانصراف لِـ {name}")
                self.tts.speak(f"إلى اللقاء يا {name}")
        else:
            self.state.show_banner(f"{name} already has {status} logged today")
            print(f"[ATTENDANCE] Skip: {name} already has {status} for today")
        
        return success
    
    def determine_status(self) -> str:
        """
        Determine if the current event should be Arrival or Departure.
        Simple heuristic: before 6 PM is Arrival, after is Departure.
        
        Returns:
            "Arrival" or "Departure"
        """
        hour = datetime.now().hour
        return "Arrival" if hour < 18 else "Departure"
    
    def process_recognition_result(
        self,
        identity: str,
        is_authorized: bool,
        distance: Optional[float]
    ):
        """
        Process a recognition result and potentially log attendance.
        
        This uses the state manager's streak tracking to ensure stable
        recognition before logging.
        """
        # Update streak and check if we should log
        should_log = self.state.update_streak(identity, is_authorized, distance)
        
        if should_log:
            status = self.determine_status()
            self.log_event(identity, status, distance)
