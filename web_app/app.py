"""Flask entry point for the phishing detection web application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services import DetectionService, ServiceError


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
service = DetectionService()


@app.after_request
def add_extension_headers(response):
    """Allow the local Chrome extension to call the analysis API."""

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    status = service.model_status()
    return jsonify({"status": "ready", "model_loaded": service.has_trained_model, "model": status})


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_url():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "")
    try:
        result = service.analyze_url(url, source=payload.get("source", "web"))
        return jsonify({"ok": True, "result": result})
    except ServiceError as exc:
        service.log_event("ERROR", {"operation": "url_analysis", "reason": str(exc)})
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - final API guard
        LOGGER.exception("URL analysis failed: %s", exc)
        service.log_event("ERROR", {"operation": "url_analysis", "reason": str(exc)})
        return jsonify({"ok": False, "error": "URL analysis failed"}), 500


@app.route("/api/train", methods=["POST", "OPTIONS"])
def train_from_csv():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    uploaded_file = request.files.get("dataset")
    try:
        result = service.train_from_upload(uploaded_file)
        return jsonify({"ok": True, "training": result})
    except ServiceError as exc:
        service.log_event("ERROR", {"operation": "model_training", "reason": str(exc)})
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - final API guard
        LOGGER.exception("Model training failed: %s", exc)
        service.log_event("ERROR", {"operation": "model_training", "reason": str(exc)})
        return jsonify({"ok": False, "error": "Model training failed"}), 500


@app.route("/api/batch-analyze", methods=["POST", "OPTIONS"])
def batch_analyze():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    uploaded_file = request.files.get("dataset")
    try:
        result = service.batch_analyze(uploaded_file)
        return jsonify({"ok": True, "batch": result})
    except ServiceError as exc:
        service.log_event("ERROR", {"operation": "batch_analysis", "reason": str(exc)})
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - final API guard
        LOGGER.exception("Batch analysis failed: %s", exc)
        service.log_event("ERROR", {"operation": "batch_analysis", "reason": str(exc)})
        return jsonify({"ok": False, "error": "Batch analysis failed"}), 500


@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    if request.method == "DELETE":
        service.clear_history()
        return jsonify({"ok": True, "history": []})
    return jsonify({"ok": True, "history": service.get_history()})


@app.route("/api/logs", methods=["GET", "DELETE"])
def logs():
    if request.method == "DELETE":
        service.clear_logs()
        return jsonify({"ok": True, "logs": []})
    return jsonify({"ok": True, "logs": service.get_logs()})


@app.route("/api/training-status", methods=["GET"])
def training_status():
    return jsonify({"ok": True, "training_status": service.get_training_status()})


@app.route("/api/model-status", methods=["GET"])
def model_status():
    return jsonify({"ok": True, "model": service.model_status()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
