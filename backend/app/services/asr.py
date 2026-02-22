import os
import tempfile
import logging
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None

INITIAL_PROMPT = "Delhi Mumbai Surat Ahmedabad. Hindi: मुझे कपड़े की फैक्ट्री चाहिए। Tamil: துணி ஆலை வேண்டும். Marathi: कापड कारखाना."

def load_asr_model():
    """Loads the Faster Whisper model into memory."""
    global _model
    if _model is not None:
        return _model
        
    logger.info(f"Loading Modular ASR Pipeline (Whisper {settings.WHISPER_MODEL_SIZE})...")
    try:
        _model = WhisperModel(settings.WHISPER_MODEL_SIZE, compute_type=settings.WHISPER_COMPUTE_TYPE)
        logger.info("ASR Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load Whisper model: {e}")
        raise

    return _model

def safe_translate(text: str, source_lang: str = 'auto') -> str:
    """
    Safely translates text. Fallbacks to the original text if the external API fails,
    ensuring the application never crashes during a live demonstration.
    """
    try:
        translator = GoogleTranslator(source=source_lang, target='en')
        return translator.translate(text)
    except Exception as e:
        logger.warning(f"MT Pipeline Warning - Translation failed: {e}. Using original text fallback.")
        return text

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Decoupled ASR + MT Pipeline.
    1. Transcribes audio in its native language.
    2. Routes to MT service only if the language is non-English with high confidence.
    """
    model = load_asr_model()
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        segments, info = model.transcribe(
            temp_path, 
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300), 
            condition_on_previous_text=False,
            initial_prompt=INITIAL_PROMPT
        )
        
        logger.info(f"ASR Stage - Detected Language: '{info.language}' (Confidence: {info.language_probability:.2f})")

        raw_text = " ".join([seg.text for seg in segments]).strip()
        logger.info(f"ASR Stage - Raw Transcript: {raw_text}")
        
        if info.language != "en" and info.language_probability > 0.60 and raw_text:
            logger.info("MT Stage - Routing non-English text to Translation Service...")
            final_text = safe_translate(raw_text, source_lang=info.language)
            logger.info(f"MT Stage - Final English Query: {final_text}")
            return final_text

        return raw_text

    except Exception as e:
        logger.error(f"ASR Pipeline Error: {e}")
        return ""
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temporary audio file: {e}")