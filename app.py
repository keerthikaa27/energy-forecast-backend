from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("lstm_energy_model.h5", compile=False)

app = Flask(__name__)
CORS(app)  # allow frontend to call backend

# Dummy historical data (replace with real DB or CSV)
historical_data = pd.Series(
    np.random.randint(20, 80, size=100)  # last 100 hours
)

# Utility: prepare input for LSTM (same shape as training)
def prepare_input(values, lookback=24):
    arr = np.array(values).reshape(1, lookback, 1)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        values = data.get("values")  # last N time steps of power_kW

        if not values or len(values) < 24:
            return jsonify({"error": "Send at least 24 values"}), 400

        input_data = prepare_input(values[-24:])
        prediction = model.predict(input_data)
        return jsonify({"prediction": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint: Weekly trend (average per day for last 7 days)
@app.route("/weekly-trend", methods=["GET"])
def weekly_trend():
    try:
        # Assume 24 hours per day
        last_7_days = historical_data[-24*7:]
        week_avg = [
            int(last_7_days[i*24:(i+1)*24].mean())
            for i in range(7)
        ]
        return jsonify({"week": week_avg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint: Yesterday usage (last 24 hours)
@app.route("/yesterday-usage", methods=["GET"])
def yesterday_usage():
    try:
        yesterday = int(historical_data[-48:-24].mean())  # 24h before last day
        return jsonify({"yesterday": yesterday})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Energy Forecasting API is running!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
