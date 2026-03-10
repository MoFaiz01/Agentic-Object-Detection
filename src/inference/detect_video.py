import argparse
import cv2
import os

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.agents.detection_agent import DetectionAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/sample_outputs/out.mp4")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("detect_video", cfg["logging"]["level"], cfg["logging"]["log_dir"])
    agent = DetectionAgent(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prev = agent.detection_history[-1] if agent.detection_history else []
        obs = agent.observe({"image": frame, "detections": prev})
        agent.act(obs)
        det_final = agent.detect(frame)
        vis = agent.visualizer.draw_detections(frame, det_final)
        writer.write(vis)

    cap.release()
    writer.release()
    logger.info(f"Saved video output: {args.out}")

if __name__ == "__main__":
    main()
