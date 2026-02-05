# Create a base model (YOLO)
# Need to install the hugingface package first
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Train YOLO model')
parser.add_argument('--data', type=str, default='coco8.yaml', help='Path to dataset yaml')
parser.add_argument('--output', type=str, default='runs/detect', help='Path to output directory')
args = parser.parse_args()

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO11 example dataset for 100 epochs
results = model.train(data=args.data, epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg"
results = model(args.output)

