import torch
from pathlib import Path
import urllib.request
import cv2
import tempfile


def extract_frames(video_path, fps=1):
    """Extract frames from video at specified rate (frames per second)."""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_numbers = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)
            frame_numbers.append(frame_count)

        frame_count += 1

    cap.release()
    return frames, frame_numbers, video_fps


def main():
    script_dir = Path(__file__).parent
    test_data = script_dir / ".." / ".." / "data" / "test" / "IMG_2473.MP4"
    test_data = test_data.resolve()

    # MegaDetector v5a model URL
    model_url = "https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
    model_path = script_dir / "md_v5a.0.0.pt"

    # Download model if not present
    if not model_path.exists():
        print(f"Downloading MegaDetector model to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")

    # Load MegaDetector model
    print("Loading MegaDetector model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Extract frames from video (1 frame per second is typical for camera traps)
    print(f"Extracting frames from {test_data.name}...")
    frames, frame_numbers, video_fps = extract_frames(test_data, fps=1)
    print(f"Extracted {len(frames)} frames (1 per second, video fps: {video_fps:.1f})")

    # Create output directory for annotated frames
    output_dir = script_dir / ".." / ".." / "outputs" / "megadetector"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference on each frame
    print("Running MegaDetector inference...")
    video_name = test_data.name
    detections_found = False
    saved_frames = 0

    for idx, (frame, frame_num) in enumerate(zip(frames, frame_numbers)):
        results = model(frame)

        # MegaDetector classes: 1=animal, 2=person, 3=vehicle
        detections = results.pandas().xyxy[0]
        animals = detections[detections['class'] == 0]  # YOLOv5 uses 0-indexed classes

        if len(animals) > 0:
            detections_found = True
            timestamp = frame_num / video_fps
            print(f"\n[DETECTION] Video: {video_name} | Frame: {frame_num} | Time: {timestamp:.2f}s")
            print(f"  Found {len(animals)} animal(s) with confidence: {animals['confidence'].values}")

            # Draw bounding boxes on frame
            annotated_frame = frame.copy()
            for _, animal in animals.iterrows():
                x1, y1, x2, y2 = int(animal['xmin']), int(animal['ymin']), int(animal['xmax']), int(animal['ymax'])
                confidence = animal['confidence']

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with confidence
                label = f"Animal: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

            # Save annotated frame
            output_path = output_dir / f"{video_name.stem}_frame_{frame_num:06d}_detections_{len(animals)}.jpg"
            cv2.imwrite(str(output_path), annotated_frame)
            saved_frames += 1

    if not detections_found:
        print(f"\nNo animals detected in {video_name}")
    else:
        print(f"\nSaved {saved_frames} annotated frames to {output_dir}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
