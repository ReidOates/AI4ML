
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model


# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "best_lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

HISTORY_PATH = DATA_DIR / "earthquake_daily_counts.csv"
EVALUATION_PATH = REPORT_DIR / "evaluation_results.csv"


# ==========================================
# 2. MEMUAT MODEL DAN DATA
# ==========================================

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

history_df = pd.read_csv(HISTORY_PATH, parse_dates=["tanggal"])
history_df = history_df.sort_values("tanggal").reset_index(drop=True)

evaluation_df = pd.read_csv(EVALUATION_PATH)

TIMESTEPS = int(metadata["timesteps"])

HORIZON_OPTIONS = [1, 7, 14, 30]


# ==========================================
# 3. FUNGSI FORECASTING
# ==========================================

def forecast_future(horizon):

    historical_values = history_df["jumlah_gempa"].values

    # Normalisasi 90 hari terakhir
    last_values = historical_values[-TIMESTEPS:]

    current_sequence = scaler.transform(
        last_values.reshape(-1, 1)
    ).flatten()

    predictions = []

    for _ in range(horizon):

        input_data = current_sequence.reshape(
            1, TIMESTEPS, 1
        )

        prediction_scaled = model.predict(
            input_data,
            verbose=0
        )[0][0]

        prediction = scaler.inverse_transform(
            np.array([[prediction_scaled]])
        )[0][0]

        prediction = max(0, float(prediction))

        predictions.append(prediction)

        next_scaled = scaler.transform(
            np.array([[prediction]])
        )[0][0]

        current_sequence = np.append(
            current_sequence[1:],
            next_scaled
        )

    return predictions


# ==========================================
# 4. HALAMAN UTAMA
# ==========================================

@app.route("/")
def index():

    # Mengambil horizon dari URL
    try:
        horizon = int(request.args.get("horizon", 7))
    except ValueError:
        horizon = 7

    if horizon not in HORIZON_OPTIONS:
        horizon = 7

    # Mengambil data historis
    recent_history = history_df.tail(180).copy()

    history_dates = recent_history["tanggal"].dt.strftime(
        "%Y-%m-%d"
    ).tolist()

    history_values = recent_history[
        "jumlah_gempa"
    ].astype(float).tolist()

    # Forecast
    predictions = forecast_future(horizon)

    last_date = history_df["tanggal"].iloc[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    future_dates = future_dates.strftime(
        "%Y-%m-%d"
    ).tolist()

    forecast_data = [
        {
            "tanggal": date,
            "prediksi": round(value, 3)
        }
        for date, value in zip(future_dates, predictions)
    ]

    # Mengambil hasil evaluasi
    evaluation_records = evaluation_df.to_dict(
        orient="records"
    )

    # Statistik data historis
    total_records = int(history_df["jumlah_gempa"].sum())

    average_daily = float(
        history_df["jumlah_gempa"].mean()
    )

    zero_days = int(
        (history_df["jumlah_gempa"] == 0).sum()
    )

    return render_template(
        "index.html",
        horizon=horizon,
        horizon_options=HORIZON_OPTIONS,
        history_dates=history_dates,
        history_values=history_values,
        future_dates=future_dates,
        future_values=predictions,
        forecast_data=forecast_data,
        evaluation_records=evaluation_records,
        metadata=metadata,
        total_records=total_records,
        average_daily=average_daily,
        zero_days=zero_days
    )


# ==========================================
# 5. MENJALANKAN SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
