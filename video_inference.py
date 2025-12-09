#!/usr/bin/env python3
"""
Video Inference Script for Feature-Level De-raining RT-DETR

This script performs object detection on video files using either the
Feature De-rain RT-DETR model or vanilla RT-DETR.

Usage:
    python video_inference.py --video path/to/video.mp4
    python video_inference.py --video path/to/video.mp4 --model vanilla
    python video_inference.py --video path/to/video.mp4 --threshold 0.5
    python video_inference.py --video path/to/video.mp4 --silent
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import FeatureDerainRTDETR


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "PekingU/rtdetr_r18vd"
DERAIN_TYPE = "multiscale"
CHECKPOINT_PATH = "./outputs_feature_derain/coco-rain_0.3/feature_derain_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Colors for visualization (one per class, cycling)
np.random.seed(42)
COLORS = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
          for _ in range(len(COCO_CLASSES))]


# =============================================================================
# Model Loading
# =============================================================================

def load_feature_derain_model(checkpoint_path: str, device: str = "cuda"):
    """Load the trained Feature De-rain RT-DETR model."""
    print(f"Loading Feature De-rain RT-DETR from: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Create model
    rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = FeatureDerainRTDETR(
        rtdetr_model=rtdetr,
        derain_type=DERAIN_TYPE,
        freeze_backbone=False
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'derain_module' in checkpoint:
            model.derain_module.load_state_dict(checkpoint['derain_module'])
            print("✓ Loaded de-rain module weights")
        if 'rtdetr' in checkpoint:
            model.rtdetr.load_state_dict(checkpoint['rtdetr'])
            print("✓ Loaded RT-DETR weights")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using pretrained weights only")
    
    model = model.to(device)
    model.eval()
    
    return model


def load_vanilla_model(device: str = "cuda"):
    """Load vanilla RT-DETR model."""
    print(f"Loading vanilla RT-DETR...")
    print(f"Device: {device}")
    model = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def process_frame(
    model, 
    processor, 
    frame, 
    device: str = "cuda",
    confidence_threshold: float = 0.4
):
    """
    Run inference on a single frame.
    
    Args:
        model: The detection model
        processor: RT-DETR image processor
        frame: Input frame (numpy array in BGR format from OpenCV)
        device: Device to run inference on
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        dict with 'boxes', 'scores', 'labels', 'inference_time'
    """
    # Convert BGR to RGB and create PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    original_size = image.size  # (width, height)
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Run inference with timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    outputs = model(pixel_values=pixel_values)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Post-process results
    target_sizes = torch.tensor([original_size[::-1]]).to(device)  # (height, width)
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes,
        threshold=confidence_threshold
    )[0]
    
    return {
        'boxes': results['boxes'].cpu().numpy(),
        'scores': results['scores'].cpu().numpy(),
        'labels': results['labels'].cpu().numpy(),
        'inference_time': inference_time
    }


# =============================================================================
# Visualization
# =============================================================================

def draw_detections_on_frame(frame, results, fps: float = 0.0, frame_number: int = 0):
    """
    Draw detection boxes, labels, and FPS on a frame.
    
    Args:
        frame: OpenCV frame (numpy array in BGR format)
        results: Detection results dict with 'boxes', 'scores', 'labels'
        fps: FPS value to display
        frame_number: Current frame number
        
    Returns:
        Frame with detections drawn (numpy array in BGR format)
    """
    frame = frame.copy()
    height, width = frame.shape[:2]
    
    boxes = results['boxes']
    scores = results['scores']
    labels = results['labels']
    
    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        # Get class name and color
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        color = COLORS[label % len(COLORS)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background and text
        text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw filled rectangle for text background
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # Draw text
        cv2.putText(frame, text, 
                   (x1, y1 - baseline - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    # Draw FPS in top-right corner
    fps_text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
    
    # Position in top-right corner with padding
    padding = 10
    x_pos = width - text_width - padding
    y_pos = text_height + padding
    
    # Draw semi-transparent background for FPS
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (x_pos - 5, padding - 5), 
                 (width - padding + 5, y_pos + baseline + 5), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw FPS text
    cv2.putText(frame, fps_text, 
               (x_pos, y_pos), 
               font, font_scale, (0, 255, 0), thickness)
    
    # Draw frame number in top-left corner (optional)
    frame_text = f"Frame: {frame_number}"
    cv2.putText(frame, frame_text, 
               (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


# =============================================================================
# Video Processing
# =============================================================================

def process_video(
    video_path: str,
    model,
    processor,
    output_path: str,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    show_progress: bool = True,
    start_time: float = 0.0,
    end_time: float = None
):
    """
    Process a video file and save the result with detections.
    
    Args:
        video_path: Path to input video
        model: Detection model
        processor: RT-DETR image processor
        output_path: Path to save output video
        device: Device to run inference on
        confidence_threshold: Minimum confidence for detections
        show_progress: Whether to show real-time progress
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: None = end of video)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = min(int(end_time * fps), total_frames)
    else:
        end_frame = total_frames
    
    # Validate frame range
    if start_frame >= total_frames:
        raise ValueError(f"Start time {start_time}s is beyond video duration")
    if start_frame >= end_frame:
        raise ValueError(f"Start time must be before end time")
    
    frames_to_process = end_frame - start_frame
    
    print(f"\nVideo Properties:")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"\nProcessing Range:")
    print(f"  Start: {start_time:.2f}s (frame {start_frame})")
    print(f"  End: {end_time if end_time else total_frames/fps:.2f}s (frame {end_frame})")
    print(f"  Frames to process: {frames_to_process}")
    print(f"  Output: {output_path}\n")
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Seeking to frame {start_frame}...")

    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process frames
    frame_count = 0
    fps_history = []
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_frame_number = start_frame + frame_count
            
            # Stop if we've reached the end frame
            if current_frame_number > end_frame:
                break
            
            # Run detection
            frame_start = time.time()
            results = process_frame(
                model, 
                processor, 
                frame, 
                device, 
                confidence_threshold
            )
            frame_time = time.time() - frame_start
            
            # Calculate FPS (moving average over last 30 frames)
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Draw detections on frame
            output_frame = draw_detections_on_frame(
                frame, 
                results, 
                fps=avg_fps,
                frame_number=current_frame_number
            )
            
            # Write frame
            out.write(output_frame)
            
            # Show progress
            if show_progress:
                elapsed = time.time() - start_time
                progress = (frame_count / frames_to_process) * 100
                eta = (elapsed / frame_count) * (frames_to_process - frame_count) if frame_count > 0 else 0
                
                print(f"\rProcessing: {frame_count}/{frames_to_process} ({progress:.1f}%) | "
                      f"FPS: {avg_fps:.1f} | ETA: {eta:.1f}s", end='', flush=True)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        
        if show_progress:
            print(f"\n\nProcessing complete!")
            print(f"  Frames processed: {frame_count}/{frames_to_process}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average FPS: {frame_count/total_time:.2f}")
            print(f"  Output saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Video inference with Feature De-rain RT-DETR',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output video file (default: <video_name>_output.mp4 in current directory)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['feature-derain', 'vanilla'],
        default='feature-derain',
        help='Model to use: feature-derain (default) or vanilla RT-DETR'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=CHECKPOINT_PATH,
        help=f'Path to Feature De-rain checkpoint (default: {CHECKPOINT_PATH})'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )
    
    parser.add_argument(
        '--silent', '-s',
        action='store_true',
        help='Run silently without showing progress'
    )
    
    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Start time in seconds (default: 0.0)'
    )
    
    parser.add_argument(
        '--end',
        type=float,
        default=None,
        help='End time in seconds (default: None = end of video)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=DEVICE,
        choices=['cuda', 'cpu'],
        help=f'Device to run inference on (default: {DEVICE})'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input video
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Set output path
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"{video_name}_output.mp4"
    
    # Ensure output has .mp4 extension
    if not args.output.endswith('.mp4'):
        args.output += '.mp4'
    
    print("=" * 70)
    print("Feature-Level De-raining RT-DETR - Video Inference")
    print("=" * 70)
    print(f"Input video: {args.video}")
    print(f"Output video: {args.output}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print(f"Progress display: {'OFF' if args.silent else 'ON'}")
    if args.start > 0 or args.end is not None:
        print(f"Time range: {args.start:.2f}s to {args.end if args.end else 'end'}s")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    if args.model == 'feature-derain':
        model = load_feature_derain_model(args.checkpoint, device=args.device)
    else:
        model = load_vanilla_model(device=args.device)
    
    # Load processor
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    
    print("✓ Model loaded successfully\n")
    
    # Process video
    try:
        process_video(
            video_path=args.video,
            model=model,
            processor=processor,
            output_path=args.output,
            device=args.device,
            confidence_threshold=args.threshold,
            show_progress=not args.silent,
            start_time=args.start,
            end_time=args.end
        )
    except Exception as e:
        print(f"\nError during video processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("Video processing complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
