import os
import io
import logging
import torch
import numpy as np
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pydub import AudioSegment

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI App
app = FastAPI(
    title="Toxicity Multi-Input Classifier API",
    description="Identify toxic comments from text, audio files, or live microphone.",
    version="3.0",
)

# Load Pretrained Model and Tokenizer
@app.on_event("startup")
async def load_model():
    global tokenizer, model, toxicity_classifier
    MODEL_NAME = "unitary/unbiased-toxic-roberta"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    toxicity_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    logger.info("✅ Model loaded successfully!")

# Define Input Data Model for API Request
class CommentRequest(BaseModel):
    comment: str

# Toxicity Thresholds
thresholds = {
    'toxicity': 0.5,
    'severe_toxic': 0.5,
    'obscene': 0.5,
    'threat': 0.5,
    'insult': 0.3,
    'identity_hate': 0.5,
}

# Classification Logic
def classify_text(comment: str):
    if not comment.strip():
        return {"error": "Comment is empty."}

    results = toxicity_classifier(comment)
    toxic_categories = {}
    is_toxic = False

    for category_scores in results:
        for score_info in category_scores:
            label = score_info["label"].lower()
            score = score_info["score"]
            if label in thresholds and score >= thresholds[label]:
                toxic_categories[label] = round(score, 2)
                is_toxic = True

    response = {
        "comment": comment,
        "is_toxic": is_toxic,
        "toxic_categories": toxic_categories if is_toxic else {},
    }
    return response

# --------- API Endpoints ---------

# Health Check
@app.get("/")
def health_check():
    return {"message": "🎯 Toxicity Multi-Input Classifier API is running!"}

# Text Classification
@app.post("/classify/")
def classify_comment(request: CommentRequest):
    try:
        return classify_text(request.comment)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Audio File Classification
@app.post("/classify-audio/")
async def classify_audio(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .wav, .mp3, or .ogg are supported.")

        # Read file bytes
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)

        # Convert to WAV if necessary
        if not file.filename.endswith('.wav'):
            audio = AudioSegment.from_file(audio_file)
            audio_wav_io = io.BytesIO()
            audio.export(audio_wav_io, format="wav")
            audio_wav_io.seek(0)
            audio_file = audio_wav_io
        else:
            audio_file.seek(0)

        # Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                raise HTTPException(status_code=400, detail="Could not understand the audio.")
            except sr.RequestError:
                raise HTTPException(status_code=503, detail="Speech recognition service unavailable.")

        # Classify the transcribed text
        return classify_text(text)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Live Microphone Listening
@app.get("/live-mic/")
def live_mic_listen():
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        print("\n🎤 Speak into the microphone! (Listening for one sentence...)")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"\n🔎 Recognized Text: {text}")

            # Classify the transcribed text
            result = classify_text(text)

            return {
                "recognized_text": text,
                "toxicity_result": result
            }

        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Could not understand the audio.")
        except sr.RequestError:
            raise HTTPException(status_code=503, detail="Speech recognition service unavailable.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --------- Main Runner ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
