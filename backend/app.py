from contextlib import asynccontextmanager
from pathlib import Path
import pickle
import re
import string

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

for resource in ["wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mood_bilstm.keras"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"

MAX_LENGTH = 30
CLASSES = ["Angry", "Happy", "Neutral", "Sad"]
EMOJI_MAP = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐",
}

PROFANITY_PATTERNS = [
    r"\b(fuck|fucking|fucker|motherfucker|cunt|bitch|asshole|bastard|dickhead|faggot|nigger|nigga)\b",
    r"\b(madarchod|bhenchod|behenchod|bhosdike|bhosadi|chutiya|chut|lund|gaand|gandu|harami|haraami|raand|bhadwa|bhadwe)\b",
    r"\b(zhavadya|zhavli|zavnya|zavnar|aai zhavadya|aai zavli|lavdya|lavada|chavat|bhadkhau|randa)\b",
]

ACTIONABLE_HARM_PATTERNS = [
    r"\b(how to (make|build|create|assemble|cook)|plant a|make a|build a|detonate (a )?)\s*bomb\b",
    r"\b(join|support|fund|recruit for)\s*(a\s+)?(terrorist|isis|taliban|terror group|terrorism)\b",
    r"\b(how to hack|hack into)\s*(a\s+)?(bank|atm|credit card|crypto wallet|bank account)\b",
    r"\b(sell|buy|manufacture|distribute)\s*(illegal drugs|cocaine|heroin|meth|fentanyl)\b",
    r"\b(how to make|how to cook)\s*(meth|fentanyl|heroin)\b",
    r"\b(hire|find)\s*(a\s+)?(hitman|contract killer)\b",
    r"\b(commit|do)\s*(mass shooting|school shooting)\b",
]

COMPILED_SAFETY_REGEXES = [
    re.compile(pattern, re.IGNORECASE) for pattern in PROFANITY_PATTERNS + ACTIONABLE_HARM_PATTERNS
]


def evaluate_content_safety(text: str) -> bool:
    cleaned_input = text.strip()
    return any(regex.search(cleaned_input) for regex in COMPILED_SAFETY_REGEXES)


lemmatizer = WordNetLemmatizer()
model = None
tokenizer = None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(tokens)


def load_artifacts():
    global model, tokenizer, CLASSES
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
    if TOKENIZER_PATH.exists():
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    if LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
            CLASSES = list(label_encoder.classes_)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Mood Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",
        "*",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model or tokenizer is not loaded. Train the model first using train.py.",
        )

    if evaluate_content_safety(request.text):
        return {
            "mood": "Flagged",
            "confidence": 100.0,
            "emoji": "⚠️",
            "flagged": True,
            "warning": "Inappropriate or harmful intent detected. Please enter respectful text.",
            "probabilities": {},
        }

    cleaned = clean_text(request.text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post", truncating="post")

    predictions = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))

    mood = CLASSES[pred_idx]
    confidence = round(float(predictions[pred_idx]) * 100, 2)
    emoji = EMOJI_MAP.get(mood, "")
    probabilities = {cls: round(float(prob), 4) for cls, prob in zip(CLASSES, predictions)}

    return {
        "mood": mood,
        "confidence": confidence,
        "emoji": emoji,
        "flagged": False,
        "probabilities": probabilities,
    }
