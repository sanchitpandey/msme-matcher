import os
import tempfile
import logging
from faster_whisper import WhisperModel
from app.core.config import settings

logger = logging.getLogger(__name__)

# Lazy loading global model
_model = None

def load_asr_model():
    global _model
    if _model is not None:
        return _model
        
    logger.info(f"Loading Whisper Model ({settings.WHISPER_MODEL_SIZE})...")
    try:
        _model = WhisperModel(settings.WHISPER_MODEL_SIZE, compute_type=settings.WHISPER_COMPUTE_TYPE)
        logger.info("Whisper Model loaded.")
    except Exception as e:
        logger.critical(f"Failed to load Whisper model: {e}")
        raise

    return _model

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Saves bytes to a temp file, transcribes, and ensures file deletion.
    """
    model = load_asr_model()
    temp_path = None
    
    try:
        # Create temp file (delete=False is required for Windows compatibility)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        segments, _ = model.transcribe(temp_path)
        
        # Combine segments
        text = " ".join([seg.text for seg in segments])
        return text.strip()

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""
        
    finally:
        # Critical: Ensure cleanup happens even if transcription fails
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Could not remove temp file {temp_path}: {e}")