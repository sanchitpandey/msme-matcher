from faster_whisper import WhisperModel
import tempfile
import os

# load model once
model = WhisperModel("base", compute_type="int8")  

def transcribe_audio(audio_bytes: bytes):
    # save temp audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    segments, info = model.transcribe(temp_path)

    text = ""
    for seg in segments:
        text += seg.text + " "

    os.remove(temp_path)
    return text.strip()
