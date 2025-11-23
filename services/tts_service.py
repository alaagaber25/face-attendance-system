import hashlib
import time
import threading
from pathlib import Path
from queue import Queue
from gtts import gTTS
from playsound import playsound

from config import ENABLE_TTS, TTS_LANG, TTS_DEDUPE_SEC, TTS_CACHE_DIR


class TTSService:
    """Text-to-Speech service with caching and de-duplication."""
    
    def __init__(
        self,
        enabled: bool = ENABLE_TTS,
        language: str = TTS_LANG,
        cache_dir: Path = TTS_CACHE_DIR,
        dedupe_seconds: float = TTS_DEDUPE_SEC
    ):
        self.enabled = enabled
        self.language = language
        self.cache_dir = cache_dir
        self.dedupe_seconds = dedupe_seconds
        
        self._last_spoken = {"text": "", "timestamp": 0.0}
        self._queue = Queue()
        self._stop_worker = False
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Start worker thread
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
    
    def speak(self, text: str):
        """Queue text for TTS playback (non-blocking)."""
        if not self.enabled:
            return
        
        # De-duplication check
        now = time.time()
        if (text == self._last_spoken["text"] and 
            (now - self._last_spoken["timestamp"]) < self.dedupe_seconds):
            return
        
        self._last_spoken["text"] = text
        self._last_spoken["timestamp"] = now
        
        print(f"[TTS] {text}")
        
        # Add to queue
        self._queue.put(text)
    
    def _worker(self):
        """Worker thread that plays TTS sequentially."""
        while not self._stop_worker:
            try:
                # Wait for next text (blocks until available)
                text = self._queue.get(timeout=1.0)
                
                # Generate cache key
                cache_key = hashlib.md5(
                    (self.language + "||" + text).encode("utf-8")
                ).hexdigest()[:16]
                
                mp3_path = self.cache_dir / f"{cache_key}.mp3"
                
                # Generate if not cached
                if not mp3_path.exists():
                    tts = gTTS(text=text, lang=self.language)
                    tts.save(str(mp3_path))
                
                # Fix path for Windows MCI
                safe_path = str(mp3_path.resolve()).replace("\\", "/")
                
                # Play audio (blocks until complete)
                try:
                    playsound(safe_path, block=True)  # Add block=True
                except Exception as play_error:
                    print(f"[TTS] Playback error: {play_error}")
                
                # Mark task as done
                self._queue.task_done()
                
            except Exception as e:
                error_msg = str(e)
                # Ignore queue.Empty timeout
                if "Empty" not in error_msg and error_msg.strip():
                    print(f"[TTS] Error: {e}")
                    
    def shutdown(self):
        """Stop the worker thread."""
        self._stop_worker = True
        if hasattr(self, '_worker_thread'):
            self._worker_thread.join(timeout=1.0)