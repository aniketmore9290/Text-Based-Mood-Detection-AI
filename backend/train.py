from pathlib import Path
import pickle
import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

for resource in ["wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mood_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "mood_bilstm.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

VOCAB_SIZE = 5000
MAX_LENGTH = 30
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 64
NUM_CLASSES = 4

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(tokens)


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_text)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["mood"])
    y = to_categorical(y_encoded, num_classes=NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    print(f"Classes: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
