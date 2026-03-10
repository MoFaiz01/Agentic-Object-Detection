from pathlib import Path
import os

from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np

from src.utils.config_loader import load_config
from src.models.yolo_model import YOLODetector
from src.models.lightweight_detector import LightweightDetector

FRONTEND_DIR = Path(__file__).parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR / "static"), static_url_path="/static")
cfg = load_config("config.yaml")


def _build_detector(config):
    model_cfg = config.get("model", {})
    web_weights = os.getenv("AODS_WEB_WEIGHTS", "yolov8n.pt")
    web_conf = float(os.getenv("AODS_WEB_CONF", "0.25"))
    web_iou = float(os.getenv("AODS_WEB_IOU", str(model_cfg.get("iou_threshold", 0.5))))
    try:
        detector = YOLODetector(
            weights=web_weights,
            conf=web_conf,
            iou=web_iou,
        )
        return detector, "yolo", web_weights
    except Exception as ex:
        print(f"[deployment.app] YOLO unavailable, using lightweight detector: {ex}")
        detector = LightweightDetector(model_cfg.get("conf_threshold", 0.35))
        return detector, "lightweight", "lightweight"


detector, detector_backend, detector_weights = _build_detector(cfg)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backend": detector_backend, "weights": detector_weights})

@app.route("/", methods=["GET"])
def index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return "Frontend not found. Expected deployment/frontend/index.html", 404
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400

    file = request.files["file"]
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    preds = detector.predict(img)
    return jsonify({"detections": preds, "backend": detector_backend, "weights": detector_weights})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
