from pathlib import Path
from typing import List

import time

import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =====================
# Configuração / Carga
# =====================

BASE_DIR = Path(__file__).resolve().parents[2]  # .../fase_4
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "lstm_nflx.keras"
SCALER_PATH = MODELS_DIR / "scaler_close.pkl"

# CSV real já existente no seu projeto (pelo print)
DATA_PATH = BASE_DIR / "notebooks" / "data" / "nflx_2018-01-01_2024-07-20.csv"

app = FastAPI(title="Tech Challenge Fase 4 - LSTM NFLX", version="1.0.0")

model = None
scaler = None

LOOKBACK = 60  # precisa bater com o que você usou no treino


class PredictRequest(BaseModel):
    closes: List[float] = Field(
        ...,
        min_length=LOOKBACK,
        max_length=LOOKBACK,
        description="Lista com os últimos 60 preços de fechamento (escala original)."
    )


class PredictResponse(BaseModel):
    predicted_close: float
    lookback: int


@app.on_event("startup")
def load_artifacts():
    global model, scaler

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler não encontrado em: {SCALER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get(
    "/predict/example",
    summary="Exemplo de payload para predição",
    description=(
        "Retorna um payload válido com os últimos 60 valores reais de fechamento "
        "da NFLX, pronto para ser usado diretamente no endpoint POST /predict."
    )
)
def predict_example():
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Arquivo de dados não encontrado em: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Close" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Coluna 'Close' não encontrada no CSV."
        )

    closes = df["Close"].tail(LOOKBACK).astype(float).tolist()

    # Retorna exatamente o formato esperado pelo /predict
    return {
        "closes": closes
    }



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo/scaler não carregados.")

    start_time = time.time()  # início

    closes = np.array(req.closes, dtype=np.float32).reshape(-1, 1)

    # Normalizar com o scaler treinado
    closes_scaled = scaler.transform(closes)

    # Formato LSTM: (1, timesteps=60, features=1)
    X_input = closes_scaled.reshape(1, LOOKBACK, 1)

    # Previsão (ainda normalizada)
    pred_scaled = model.predict(X_input, verbose=0)

    # Voltar para escala original
    pred_real = scaler.inverse_transform(pred_scaled)[0, 0]

    elapsed_ms = (time.time() - start_time) * 1000  # fim

    print(f"[MONITOR] /predict latency: {elapsed_ms:.2f} ms")

    return PredictResponse(
        predicted_close=float(pred_real),
        lookback=LOOKBACK
    )

