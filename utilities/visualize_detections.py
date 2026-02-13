"""
Utility for visualizing bird detections on video frames.

This script extracts frames from videos, runs YOLO or MegaDetector detection,
and displays or saves frames with bounding boxes where birds/animals are detected.
"""

from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import urllib.request


class BirdDetectionVisualizer:
    """Visualizes bird/animal detections on video frames using YOLO or MegaDetector."""

    def __init__(self, model_path: str = "yolo11n.pt", model_type: str = "yolo"):
        """
        Initialize the visualizer with a YOLO or MegaDetector model.

        Args:
            model_path: Path to the model weights file
            model_type: Type of model - "yolo" or "megadetector"
        """
        self.model_type = model_type.lower()

        if self.model_type == "yolo":
            self.model = YOLO(model_path)
        elif self.model_type == "megadetector":
            # Download MegaDetector if path is URL or doesn't exist
            if model_path.startswith("http"):
                local_path = Path("md_v5a.0.0.pt")
                if not local_path.exists():
                    print(f"Downloading MegaDetector model...")
                    urllib.request.urlretrieve(model_path, local_path)
                model_path = str(local_path)
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'yolo' or 'megadetector'")

    def extract_frames(self, video_path: Path, frame_interval: int = 30) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from a video at regular intervals.

        Args:
            video_path: Path to the video file
            frame_interval: Extract every nth frame (default: 30, ~1 frame per second for 30fps video)

        Returns:
            List of tuples containing (frame_number, frame_image)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append((frame_count, frame))

            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames

    def detect_and_draw(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> Tuple[np.ndarray, int]:
        """
        Run detection on a frame and draw bounding boxes.

        Args:
            frame: Input frame (BGR format from OpenCV)
            confidence_threshold: Minimum confidence for detections

        Returns:
            Tuple of (annotated_frame, number_of_detections)
        """
        annotated_frame = frame.copy()
        detection_count = 0

        if self.model_type == "yolo":
            # YOLO inference
            results = self.model(frame, verbose=False)[0]
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    if confidence >= confidence_threshold:
                        detection_count += 1
                        class_name = results.names[class_id] if hasattr(results, 'names') else f"Class {class_id}"
                        self._draw_box(annotated_frame, int(x1), int(y1), int(x2), int(y2), class_name, confidence)

        elif self.model_type == "megadetector":
            # MegaDetector inference
            results = self.model(frame)
            detections = results.pandas().xyxy[0]

            # Filter for animals (class 0 in YOLOv5 0-indexed)
            animals = detections[detections['confidence'] >= confidence_threshold]

            for _, detection in animals.iterrows():
                detection_count += 1
                x1, y1 = int(detection['xmin']), int(detection['ymin'])
                x2, y2 = int(detection['xmax']), int(detection['ymax'])
                confidence = detection['confidence']
                class_name = detection['name'] if 'name' in detection else "Animal"
                self._draw_box(annotated_frame, x1, y1, x2, y2, class_name, confidence)

        return annotated_frame, detection_count

    def _draw_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, class_name: str, confidence: float):
        """Helper method to draw bounding box and label on frame."""
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

    def visualize_video(
        self,
        video_path: Path,
        frame_interval: int = 30,
        confidence_threshold: float = 0.25,
        max_frames: Optional[int] = None,
        save_output: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Process video and visualize detections on multiple frames.

        Args:
            video_path: Path to the video file
            frame_interval: Extract every nth frame
            confidence_threshold: Minimum confidence for detections
            max_frames: Maximum number of frames to display (None for all)
            save_output: Whether to save annotated frames to disk
            output_dir: Directory to save frames (if save_output=True)
        """
        # Extract frames
        frames = self.extract_frames(video_path, frame_interval)

        if max_frames:
            frames = frames[:max_frames]

        # Process frames and collect those with detections
        frames_with_detections = []

        print(f"\nProcessing {len(frames)} frames...")
        for frame_num, frame in frames:
            annotated_frame, detection_count = self.detect_and_draw(frame, confidence_threshold)

            if detection_count > 0:
                frames_with_detections.append((frame_num, annotated_frame, detection_count))
                print(f"Frame {frame_num}: {detection_count} detection(s)")

        print(f"\nFound detections in {len(frames_with_detections)} out of {len(frames)} frames")

        # Save frames if requested
        if save_output and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for frame_num, annotated_frame, detection_count in frames_with_detections:
                output_path = output_dir / f"frame_{frame_num:06d}_detections_{detection_count}.jpg"
                cv2.imwrite(str(output_path), annotated_frame)
            print(f"Saved {len(frames_with_detections)} annotated frames to {output_dir}")

        # Display frames with detections
        if frames_with_detections:
            self._display_frames(frames_with_detections, video_path.name)
        else:
            print("No detections found in any frames.")

    def _display_frames(self, frames_with_detections: List[Tuple[int, np.ndarray, int]], video_name: str):
        """
        Display frames with detections in a grid layout.

        Args:
            frames_with_detections: List of (frame_number, annotated_frame, detection_count)
            video_name: Name of the video for the plot title
        """
        num_frames = len(frames_with_detections)

        # Determine grid layout
        cols = min(3, num_frames)
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        model_name = "MegaDetector" if self.model_type == "megadetector" else "YOLO"
        fig.suptitle(f"{model_name} Detections - {video_name}", fontsize=16)

        # Handle single frame case
        if num_frames == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_frames > 1 else axes

        for idx, (frame_num, annotated_frame, detection_count) in enumerate(frames_with_detections):
            # Convert BGR to RGB for matplotlib
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            axes[idx].imshow(rgb_frame)
            axes[idx].set_title(f"Frame {frame_num} ({detection_count} detection{'s' if detection_count > 1 else ''})")
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_frames, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    """Example usage of the BirdDetectionVisualizer."""
    import sys

    # Get the directory containing this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Find test video
    test_video = project_root / "data" / "test" / "IMG_2473.MP4"

    if not test_video.exists():
        print(f"Error: Test video not found at {test_video}")
        return

    # Check if user wants MegaDetector or YOLO
    use_megadetector = len(sys.argv) > 1 and sys.argv[1].lower() == "megadetector"

    if use_megadetector:
        # Use MegaDetector
        print("Using MegaDetector model...")
        model_url = "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
        visualizer = BirdDetectionVisualizer(model_url, model_type="megadetector")
        output_dir = project_root / "outputs" / "megadetector_visualizations"
    else:
        # Use YOLO
        model_path = project_root / "yolo11n.pt"
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            print("Please ensure yolo11n.pt is in the project root directory")
            print("\nOr run with 'megadetector' argument to use MegaDetector:")
            print(f"  python {Path(__file__).name} megadetector")
            return

        print(f"Using YOLO model from {model_path}")
        visualizer = BirdDetectionVisualizer(str(model_path), model_type="yolo")
        output_dir = project_root / "outputs" / "detections"

    # Process video
    print(f"\nProcessing video: {test_video.name}")

    visualizer.visualize_video(
        video_path=test_video,
        frame_interval=30,  # Extract every 30th frame (~1 fps for 30fps video)
        confidence_threshold=0.25,  # Minimum confidence for detections
        max_frames=20,  # Limit to first 20 extracted frames
        save_output=True,  # Save annotated frames
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
