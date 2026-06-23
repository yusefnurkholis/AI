import csv
import io
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "rice_leaf_disease_model.keras"
CONFIG_PATH = MODEL_DIR / "model_config.json"
HISTORY_PATH = DATA_DIR / "training_history.json"
REPORT_PATH = DATA_DIR / "classification_report.csv"
METRICS_PATH = DATA_DIR / "metrics.json"
CONFUSION_PATH = DATA_DIR / "confusion_matrix.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

_model = None
_config = None


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_config():
    global _config
    if _config is None:
        _config = read_json(CONFIG_PATH, {})
    return _config


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model belum tersedia. Salin rice_leaf_disease_model.keras "
                "ke folder model/."
            )
        from tensorflow import keras
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def read_classification_report():
    if not REPORT_PATH.exists():
        return []
    rows = []
    with REPORT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key is None:
                    continue
                name = key.strip() or "kelas"
                if value is None:
                    cleaned[name] = ""
                    continue
                value = value.strip()
                try:
                    cleaned[name] = float(value)
                except ValueError:
                    cleaned[name] = value
            rows.append(cleaned)
    return rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    config = load_config()
    return jsonify({
        "model_ready": MODEL_PATH.exists(),
        "config_ready": CONFIG_PATH.exists(),
        "history_ready": HISTORY_PATH.exists(),
        "report_ready": REPORT_PATH.exists(),
        "metrics_ready": METRICS_PATH.exists(),
        "confusion_ready": CONFUSION_PATH.exists(),
        "class_names": config.get("class_names", []),
        "architecture": config.get("architecture", "EfficientNetB0 Transfer Learning"),
        "image_size": config.get("image_size", 224),
    })


@app.route("/api/results")
def results():
    config = load_config()
    metrics = read_json(METRICS_PATH, {})
    history = read_json(HISTORY_PATH, {})
    confusion = read_json(CONFUSION_PATH, {})
    report = read_classification_report()

    combined_metrics = {
        "test_accuracy": config.get("test_accuracy", metrics.get("test_accuracy")),
        "test_loss": config.get("test_loss", metrics.get("test_loss")),
        "weighted_precision": config.get(
            "weighted_precision", metrics.get("weighted_precision")
        ),
        "weighted_recall": config.get(
            "weighted_recall", metrics.get("weighted_recall")
        ),
        "weighted_f1_score": config.get(
            "weighted_f1_score", metrics.get("weighted_f1_score")
        ),
    }

    return jsonify({
        "metrics": combined_metrics,
        "history": history,
        "classification_report": report,
        "confusion_matrix": confusion,
        "class_names": config.get("class_names", []),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar yang dikirim."}), 400

    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"error": "Nama file gambar kosong."}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "File yang dikirim bukan gambar valid."}), 400

    try:
        config = load_config()
        class_names = config.get("class_names", [])
        image_size = int(config.get("image_size", 224))

        if not class_names:
            return jsonify({
                "error": "model_config.json belum tersedia atau class_names kosong."
            }), 503

        model = load_model()
        resized = image.resize((image_size, image_size))
        array = np.asarray(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        probabilities = model.predict(array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        predictions = [
            {"class": name, "probability": float(probabilities[index])}
            for index, name in enumerate(class_names)
        ]
        predictions.sort(key=lambda item: item["probability"], reverse=True)

        return jsonify({
            "predicted_class": class_names[predicted_index],
            "confidence": confidence,
            "uncertain": confidence < 0.60,
            "probabilities": predictions,
        })
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        return jsonify({"error": f"Prediksi gagal: {error}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Ukuran file maksimal 10 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
