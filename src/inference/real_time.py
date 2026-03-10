import argparse
import cv2

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.agents.detection_agent import DetectionAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("real_time", cfg["logging"]["level"], cfg["logging"]["log_dir"])
    agent = DetectionAgent(cfg)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prev = agent.detection_history[-1] if agent.detection_history else []
        obs = agent.observe({"image": frame, "detections": prev})
        action = agent.act(obs)
        det_final = agent.detect(frame)
        vis = agent.visualizer.draw_detections(frame, det_final)
        cv2.putText(
            vis,
            f"Model: {action['model']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imshow("Agentic Object Detection", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
