"""
Login/Logout Service Module
Handles user login and logout with attendance status tracking.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

from database.db_manager import DatabaseManager
from services.tts_service import TTSService
from utils.state import ApplicationState


class LoginLogoutService:
    """
    Service for managing user login/logout with time-based attendance status.
    
    Business Rules:
    - First login of the day: Mark as "Attended" (Arrival)
    - Logout after 4-8 hours: Mark as "Half Day" (Departure)
    - Logout after 8+ hours: Mark as "Full Day" (Departure)
    """
    
    # Time thresholds in hours
    HALF_DAY_HOURS = 4.0
    FULL_DAY_HOURS = 8.0
    
    # Cooldown between login/logout events for same person (seconds)
    EVENT_COOLDOWN_SEC = 45.0
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        tts_service: TTSService,
        state: ApplicationState
    ):
        self.db = db_manager
        self.tts = tts_service
        self.state = state
    
    def process_recognition(
        self,
        identity: str,
        is_authorized: bool,
        distance: Optional[float]
    ):
        """
        Process a recognition result and handle login/logout.
        
        This uses the state manager's streak tracking to ensure stable
        recognition before logging.
        """
        # Update streak and check if we should process
        should_process = self.state.update_streak(identity, is_authorized, distance)
        
        if should_process:
            self._handle_login_logout(identity, distance)
    
    def _handle_login_logout(self, name: str, distance: Optional[float]):
        """
        Handle login or logout for a user.
        
        Args:
            name: Person's name
            distance: Recognition distance (for tracking accuracy)
        """
        # Get today's attendance for this user
        login_record = self._get_today_login(name)
        
        if login_record is None:
            # First time seeing this user today → LOGIN
            self._handle_login(name, distance)
        else:
            # User already logged in today → LOGOUT
            self._handle_logout(name, login_record, distance)
    
    def _handle_login(self, name: str, distance: Optional[float]):
        """Handle user login (first appearance today)."""
        success = self.db.log_attendance(name, "Attended", distance)
        
        if success:
            self.state.show_banner(f"{name} - Attended (Login)")
            print(f"[LOGIN] {name} logged in - Status: Attended")
            
            # TTS feedback
            self.tts.speak(f"تم تسجيل حضور {name}")
            self.tts.speak(f"مرحباً يا {name}")
        else:
            self.state.show_banner(f"{name} already logged in today")
            print(f"[LOGIN] {name} already has login record for today")
    
    def _handle_logout(
        self,
        name: str,
        login_record: dict,
        distance: Optional[float]
    ):
        """
        Handle user logout (subsequent appearance after login).
        
        Args:
            name: Person's name
            login_record: Dict with 'ts_unix' and 'status' from login
            distance: Recognition distance
        """
        # Calculate time elapsed since login
        login_time = login_record['ts_unix']
        now = datetime.now().timestamp()
        hours_elapsed = (now - login_time) / 3600.0
        
        # Determine status based on hours worked
        if hours_elapsed < self.HALF_DAY_HOURS:
            # Too early for logout
            print(f"[LOGOUT] {name} - Only {hours_elapsed:.1f}h elapsed, minimum is {self.HALF_DAY_HOURS}h")
            self.state.show_banner(f"{name} - Too early to logout ({hours_elapsed:.1f}h)")
            self.tts.speak(f"وقت مبكر للانصراف يا {name}")
            return
        
        elif hours_elapsed < self.FULL_DAY_HOURS:
            status = "Half Day"
            status_ar = "نصف يوم"
        else:
            status = "Full Day"
            status_ar = "يوم كامل"
        
        # Update the attendance record
        success = self._update_attendance_status(name, status, distance)
        
        if success:
            self.state.show_banner(f"{name} - {status} (Logout)")
            print(f"[LOGOUT] {name} logged out - Status: {status} ({hours_elapsed:.1f}h)")
            
            # TTS feedback
            self.tts.speak(f"تم تسجيل انصراف {name}")
            self.tts.speak(f"حالة الحضور: {status_ar}")
            self.tts.speak(f"إلى اللقاء يا {name}")
        else:
            self.state.show_banner(f"{name} - Logout failed")
    
    def _get_today_login(self, name: str) -> Optional[dict]:
        """
        Get today's login record for a user.
        
        Returns:
            Dict with 'ts_unix' and 'status', or None if no login today
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        try:
            with self.db._connect() as con:
                row = con.execute("""
                    SELECT ts_unix, status FROM attendance
                    WHERE name = ?
                      AND substr(ts_iso, 1, 10) = ?
                    ORDER BY ts_unix ASC
                    LIMIT 1
                """, (name, today_str)).fetchone()
                
                if row:
                    return {'ts_unix': row[0], 'status': row[1]}
                return None
                
        except Exception as e:
            print(f"[DB] Error getting today's login: {e}")
            return None
    
    def _update_attendance_status(
        self,
        name: str,
        new_status: str,
        distance: Optional[float]
    ) -> bool:
        """
        Update the attendance status for today's record.
        
        Args:
            name: Person's name
            new_status: New status ("Half Day" or "Full Day")
            distance: Recognition distance
            
        Returns:
            True if updated successfully
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        try:
            with self.db._connect() as con:
                # Update the status and distance
                con.execute("""
                    UPDATE attendance
                    SET status = ?,
                        distance = ?
                    WHERE name = ?
                      AND substr(ts_iso, 1, 10) = ?
                """, (new_status, distance, name, today_str))
                
                rows_affected = con.total_changes
                return rows_affected > 0
                
        except Exception as e:
            print(f"[DB] Error updating attendance status: {e}")
            return False