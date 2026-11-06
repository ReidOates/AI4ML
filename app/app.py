from pathlib import Path
import json

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

HISTORY_PATH = DATA_DIR / "earthquake_daily_counts.csv"
FORECAST_PATH = DATA_DIR / "future_forecast.csv"
EVALUATION_PATH = REPORT_DIR / "evaluation_results.csv"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"

HORIZON_OPTIONS = [1, 7, 14, 30]


def load_data():
    # Data historis
    history = pd.read_csv(HISTORY_PATH)
    history["tanggal"] = pd.to_datetime(history["tanggal"])
    history = history.sort_values("tanggal").reset_index(drop=True)

    # Hasil prediksi yang sudah dihitung sebelumnya
    forecast = pd.read_csv(FORECAST_PATH)

    # Menyesuaikan nama kolom agar lebih mudah dikenali
    forecast.columns = [
        str(column).strip().lower()
        for column in forecast.columns
    ]

    date_candidates = ["tanggal", "date", "ds"]
    prediction_candidates = [
        "prediksi",
        "prediksi_jumlah_gempa",
        "jumlah_gempa_prediksi",
        "prediction",
        "forecast",
        "yhat"
    ]

    date_column = next(
        (col for col in date_candidates if col in forecast.columns),
        None
    )

    prediction_column = next(
        (col for col in prediction_candidates if col in forecast.columns),
        None
    )

    if date_column is None or prediction_column is None:
        raise ValueError(
            "Kolom future_forecast.csv tidak dikenali. "
            f"Kolom yang tersedia: {list(forecast.columns)}"
        )

    forecast = forecast.rename(columns={
        date_column: "tanggal",
        prediction_column: "prediksi"
    })

    forecast["tanggal"] = pd.to_datetime(forecast["tanggal"])
    forecast["prediksi"] = pd.to_numeric(
        forecast["prediksi"], errors="coerce"
    )

    forecast = forecast.dropna(
        subset=["tanggal", "prediksi"]
    ).sort_values("tanggal").reset_index(drop=True)

    # Hasil evaluasi
    if EVALUATION_PATH.exists():
        evaluation = pd.read_csv(EVALUATION_PATH)
    else:
        evaluation = pd.DataFrame()

    # Metadata bersifat opsional
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        metadata = {}

    return history, forecast, evaluation, metadata


history_df, forecast_df, evaluation_df, metadata = load_data()


@app.route("/")
def index():
    try:
        horizon = int(request.args.get("horizon", 7))
    except (ValueError, TypeError):
        horizon = 7

    if horizon not in HORIZON_OPTIONS:
        horizon = 7

    # Data historis 180 hari terakhir
    recent_history = history_df.tail(180).copy()

    history_dates = recent_history["tanggal"].dt.strftime(
        "%Y-%m-%d"
    ).tolist()

    history_values = recent_history[
        "jumlah_gempa"
    ].astype(float).tolist()

    # Ambil hasil prediksi sesuai horizon
    selected_forecast = forecast_df.head(horizon)

    future_dates = selected_forecast["tanggal"].dt.strftime(
        "%Y-%m-%d"
    ).tolist()

    predictions = selected_forecast[
        "prediksi"
    ].astype(float).tolist()

    forecast_data = [
        {
            "tanggal": date,
            "prediksi": round(value, 3)
        }
        for date, value in zip(future_dates, predictions)
    ]

    evaluation_records = evaluation_df.to_dict(
        orient="records"
    )

    total_records = int(history_df["jumlah_gempa"].sum())
    average_daily = float(history_df["jumlah_gempa"].mean())
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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )