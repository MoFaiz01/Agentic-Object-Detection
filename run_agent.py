#!/usr/bin/env python3
"""
Main entry point for the Agentic Object Detection System.
Run detection with the new agent architecture.
Fully connected: agents, models, data, tools, and memory.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.detection_agent import DetectionAgent, create_detection_agent
from src.memory.episodic_memory import EpisodicMemory, SemanticMemory
from src.tools.vision_tools import DetectionVisualizer, MetricsCalculator
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Agentic Object Detection System"
    )
    
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to input image"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        help="Path to input video"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--policy", 
        type=str, 
        default="entropy",
        choices=["entropy", "confidence", "random", "adaptive", "cascade"],
        help="Policy for model selection"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/agent_output",
        help="Output directory"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Display output"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--batch", 
        type=str,
        help="Process all images in directory"
    )
    parser.add_argument(
        "--model", 
        type=str,
        default=None,
        help="Force specific model (lightweight, yolo, faster_rcnn)"
    )
    
    return parser.parse_args()


def run_image_detection(agent: DetectionAgent, image_path: str, 
                       output_dir: str, show: bool = False):
    """Run detection on a single image."""
    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print(f"{'='*50}")
    
    # Use agent's data integration
    image = agent.load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Create observation state
    previous_detections = agent.detection_history[-1] if agent.detection_history else []
    state = {"image": image, "detections": previous_detections}
    observation = agent.observe(state)
    print(f"Observation: {observation}")
    
    # Get action
    action = agent.act(observation)
    print(f"Selected action: {action}")
    
    # Perform detection using agent (which uses underlying models)
    detections = agent.detect(image)
    print(f"Detections: {len(detections)} objects found")
    
    for i, det in enumerate(detections):
        print(f"  [{i}] {det.get('class_name', 'unknown')}: "
              f"score={det.get('score', 0):.3f}, "
              f"bbox={det.get('bbox', [])}")
    
    # Visualize using tools
    vis_image = agent.visualizer.draw_detections(image, detections)
    
    # Save output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = Path(image_path).stem
    out_file = output_path / f"{filename}_result.jpg"
    cv2.imwrite(str(out_file), vis_image)
    print(f"\nOutput saved to: {out_file}")
    
    if show:
        cv2.imshow("Detection Result", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


def run_batch_detection(agent: DetectionAgent, directory: str,
                        output_dir: str, show: bool = False):
    """Run detection on all images in a directory."""
    print(f"\n{'='*50}")
    print(f"Batch Processing: {directory}")
    print(f"{'='*50}")
    
    # Use agent's directory detection
    results = agent.detect_directory(directory)
    
    print(f"\nProcessed {len(results)} images:")
    total_detections = 0
    
    for result in results:
        img_path = result["image_path"]
        detections = result["detections"]
        model = result["model_used"]
        num_dets = len(detections)
        total_detections += num_dets
        
        print(f"  {Path(img_path).name}: {num_dets} detections using {model}")
        
        # Save visualization
        if result["visualization"] is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            filename = Path(img_path).stem
            out_file = output_path / f"{filename}_result.jpg"
            cv2.imwrite(str(out_file), result["visualization"])
    
    print(f"\nTotal detections: {total_detections}")
    print(f"Output saved to: {output_dir}")
    
    return results


def run_video_detection(agent: DetectionAgent, video_path: str,
                      output_dir: str, show: bool = False):
    """Run detection on video."""
    print(f"\n{'='*50}")
    print(f"Processing video: {video_path}")
    print(f"{'='*50}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = Path(video_path).stem
    out_file = output_path / f"{filename}_result.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame using agent
        previous_detections = agent.detection_history[-1] if agent.detection_history else []
        state = {"image": frame, "detections": previous_detections}
        observation = agent.observe(state)
        action = agent.act(observation)
        detections = agent.detect(frame)
        
        # Visualize
        vis_frame = agent.visualizer.draw_detections(frame, detections)
        
        # Add frame number
        cv2.putText(
            vis_frame, 
            f"Frame: {frame_idx}", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Add model info
        cv2.putText(
            vis_frame,
            f"Model: {agent.current_model}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Write frame
        writer.write(vis_frame)
        
        if show:
            cv2.imshow("Video Detection", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    writer.release()
    
    if show:
        cv2.destroyAllWindows()
    
    print(f"\nOutput saved to: {out_file}")


def run_camera_detection(agent: DetectionAgent, camera_idx: int,
                        output_dir: str, show: bool = False):
    """Run real-time detection from camera."""
    print(f"\n{'='*50}")
    print(f"Starting camera: {camera_idx}")
    print(f"{'='*50}")
    
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open camera {camera_idx}")
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame using agent
        previous_detections = agent.detection_history[-1] if agent.detection_history else []
        state = {"image": frame, "detections": previous_detections}
        observation = agent.observe(state)
        action = agent.act(observation)
        detections = agent.detect(frame)
        
        # Visualize using tools
        vis_frame = agent.visualizer.draw_detections(frame, detections)
        
        # Add info
        cv2.putText(
            vis_frame,
            f"Model: {agent.current_model}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            vis_frame,
            f"Detections: {len(detections)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Camera Detection", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.policy:
        if "agent" not in config:
            config["agent"] = {}
        config["agent"]["policy"] = args.policy
    
    if args.model:
        if "model" not in config:
            config["model"] = {}
        config["model"]["primary"] = args.model
    
    print("="*60)
    print("Agentic Object Detection System")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Policy: {args.policy}")
    if args.model:
        print(f"Model: {args.model}")
    
    # Initialize agent (which connects models, tools, and data)
    agent = DetectionAgent(config)
    
    print(f"\nAgent initialized: {agent}")
    print(f"Available models: {list(agent.models.keys())}")
    print(f"Current model: {agent.current_model}")
    print(f"Data directory: {agent.data_dir}")
    print(f"Results directory: {agent.results_dir}")
    
    try:
        # Run based on input type
        if args.batch:
            # Batch directory processing
            run_batch_detection(agent, args.batch, args.output, args.show)
        
        elif args.image:
            # Single image
            run_image_detection(agent, args.image, args.output, args.show)
        
        elif args.video:
            # Video
            run_video_detection(agent, args.video, args.output, args.show)
        
        else:
            # Default: camera
            run_camera_detection(agent, args.camera, args.output, args.show)
        
        # Print final stats
        print(f"\n{'='*50}")
        print("Final Statistics:")
        print(f"{'='*50}")
        print(f"Agent: {agent}")
        print(f"\nDetection Stats:")
        stats = agent.get_detection_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        print(f"\nAll components connected successfully!")
        print(f"  - Models: {list(agent.models.keys())}")
        print(f"  - Tools: ImageProcessor, DetectionVisualizer, MetricsCalculator")
        print(f"  - Data: {agent.data_dir}")
        print(f"  - Results: {agent.results_dir}")
    
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
