import argparse
import cv2
import os

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.agents.detection_agent import DetectionAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--out", default="results/sample_outputs/out.jpg", help="Output image path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("detect_image", cfg["logging"]["level"], cfg["logging"]["log_dir"])
    agent = DetectionAgent(cfg)

    image = agent.load_image(args.image)
    prev = agent.detection_history[-1] if agent.detection_history else []
    obs = agent.observe({"image": image, "detections": prev})
    action = agent.act(obs)
    det_final = agent.detect(image)
    vis = agent.visualizer.draw_detections(image.copy(), det_final)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)
    logger.info(
        f"Saved output to: {args.out} | model={action['model']} detections={len(det_final)}"
    )

    if args.show:
        cv2.imshow("Detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
