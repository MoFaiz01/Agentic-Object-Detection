from pathlib import Path
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

from src.utils.config_loader import load_config
from src.models.yolo_model import YOLODetector
from src.models.lightweight_detector import LightweightDetector

app = FastAPI()
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
        print(f"[deployment.api] YOLO unavailable, using lightweight detector: {ex}")
        detector = LightweightDetector(model_cfg.get("conf_threshold", 0.35))
        return detector, "lightweight", "lightweight"


detector, detector_backend, detector_weights = _build_detector(cfg)
FRONTEND_DIR = Path(__file__).parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    if FRONTEND_DIR.exists():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    return HTMLResponse("<h3>Frontend not found. Expected deployment/frontend/index.html</h3>", status_code=404)

@app.get("/health")
def health():
    return {"status": "ok", "backend": detector_backend, "weights": detector_weights}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    preds = detector.predict(img)
    return {"detections": preds, "backend": detector_backend, "weights": detector_weights}
