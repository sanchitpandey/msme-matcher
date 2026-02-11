from fastapi import FastAPI, UploadFile, File
import wave
import json
from app.services.asr import transcribe_audio

app = FastAPI(title="IndiaAI MSME Matching System")

@app.get("/")
def root():
    return {"message": "IndiaAI MSME AI system running"}

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    return {"transcript": text}
