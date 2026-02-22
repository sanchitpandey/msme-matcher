# IndiaAI MSME Intelligent Matching System

Submission for IndiaAI Innovation Challenge 2026 (MSME – TEAMS)

## Overview

This project is a multimodal AI-powered MSME onboarding and supplier discovery system.

It enables:

* Voice-based MSME registration (multilingual ASR)
* OCR-based document onboarding (GST, invoices, certificates)
* Automatic category classification
* Attribute extraction from unstructured text
* Hybrid semantic + keyword search
* Learning-to-Rank based supplier ranking
* Explainable AI decision factors
* ONDC protocol-compatible search endpoint
* Feedback loop for continuous learning
* DPDP-compliant data retention mechanism

---

## Architecture

User Input (Voice / Document / Text)
↓
ASR (Faster-Whisper)
OCR (Tesseract)
↓
SBERT Embeddings
Hybrid Search (BM25 + FAISS)
↓
LightGBM LambdaRank Model
↓
Explainable AI + ONDC Adapter

---

## Tech Stack

**Backend:**

* FastAPI
* SQLAlchemy
* SQLite

**AI Models:**

* Faster-Whisper (ASR)
* Tesseract OCR
* SentenceTransformers (all-MiniLM-L6-v2)
* LightGBM LambdaRank
* FAISS
* BM25

**Frontend:**

* Next.js
* TypeScript
* TailwindCSS

---

## Setup Instructions

### 1. Backend Setup

Create virtual environment:
```
python -m venv venv
venv\Scripts\activate (Windows)
```
Install dependencies:
```
pip install -r requirements.txt
```
Run backend:
```
cd backend
uvicorn app.main:app --reload
```
Backend runs at:
http://localhost:8000

---

### 2. Frontend Setup

Navigate to frontend folder:
```
cd frontend
npm install
npm run dev
```
Frontend runs at:
http://localhost:3000

---

## Demo Flow

1. Speak MSME requirement in Hindi/Tamil/English
2. Upload GST or invoice document
3. System auto-fills registration form
4. Intelligent supplier ranking
5. View Explainable AI factors
6. Toggle ONDC Network mode

---

## Compliance

* Search logs stored for audit
* Feedback stored for model improvement
* Auto purge script included (30-day retention)
* Designed to be DPDP compliant

---

## Submission Notes

This system demonstrates a deployable national-scale MSME matching infrastructure aligned with IndiaAI and ONDC digital public infrastructure vision.
