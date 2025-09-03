from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib

# --- Initialize Flask ---
app = Flask(__name__)
CORS(app, supports_credentials=True)

# --- Load model & scaler ---
MODEL_PATH = "lstm_energy_model_multi.h5"
SCALER_PATH = "scaler_multi.pkl"
SEQ_LEN = 24

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# --- Helper: Prepare input for prediction ---
def prepare_input(values):
    values = np.array(values[-SEQ_LEN:]).reshape(-1, 1)
    scaled = scaler.transform(values)
    return scaled.reshape(1, SEQ_LEN, 1)

# --- Helper: Compute trend metadata ---
def get_trend_metadata(forecast):
    trend = np.mean(np.diff(forecast))
    arrow_direction = "up" if trend > 0 else "down"
    curvature = "accelerating" if np.mean(np.diff(np.diff(forecast))) > 0 else "steady"
    return arrow_direction, curvature

# --- Endpoint: Forecast next <horizon> hours ---
@app.route("/predict/<int:horizon>", methods=["POST"])
def predict_horizon(horizon):
    try:
        data = request.get_json()
        values = data.get("values")

        print("Incoming values:", values)

        if not values or len(values) < SEQ_LEN:
            return jsonify({"error": f"Need at least {SEQ_LEN} values"}), 400

        input_data = prepare_input(values)
        print("Prepared input shape:", input_data.shape)

        prediction_scaled = model.predict(input_data)
        print("Raw prediction (scaled):", prediction_scaled)

        try:
            prediction_rescaled = scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).flatten()
        except Exception as e:
            print("Inverse scaling failed:", e)
            return jsonify({"error": f"Scaling error: {str(e)}"}), 500

        forecast = prediction_rescaled[:horizon].tolist()
        print("Rescaled forecast:", forecast)

        # --- Compute trend metadata ---
        arrow, curvature = get_trend_metadata(forecast)

        return jsonify({
            "forecast": forecast,
            "arrow": arrow,
            "curvature": curvature
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# --- Health check ---
@app.route("/", methods=["GET"])
def home():
    return "*Energy Forecasting API is running!*"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
