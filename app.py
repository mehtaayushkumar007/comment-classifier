import io
import torch
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize the FastAPI App
app = FastAPI(
    title="Toxicity Multi-Input Classifier API",
    description="Identify toxic comments from text, audio files, or live microphone input.",
    version="3.0",
)

# Load Pretrained Model and Tokenizer
MODEL_NAME = "unitary/unbiased-toxic-roberta"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    toxicity_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {e}")

# Define Request Model
class CommentRequest(BaseModel):
    comment: str

# Thresholds
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
    comment = comment.strip()
    if not comment:
        raise HTTPException(status_code=400, detail="Comment text is empty.")

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

    return {
        "comment": comment,
        "is_toxic": is_toxic,
        "toxic_categories": toxic_categories if is_toxic else {},
    }

# --------- API Endpoints ---------

@app.get("/")
def root():
    return {"message": "🎯 Toxicity Classifier API is running!"}

@app.post("/classify/")
def classify_comment(request: CommentRequest):
    return classify_text(request.comment)

@app.post("/classify-audio/")
async def classify_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav, .mp3, or .ogg supported.")

    try:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return classify_text(text)

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand the audio.")
    except sr.RequestError:
        raise HTTPException(status_code=503, detail="Speech recognition service unavailable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/live-mic/")
def live_mic_listen():
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)

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
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --------- Local Runner ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
