# Create a base model (YOLO v11)
'''
This code trains a YOLO v11 model on a custom dataset (--train_data).
Then predict on a test image (--test_data).
Output is saved to (--output).

Example usage:
python yolo.py --train_data data/custom.yaml --test_data data/custom.yaml --output runs/detect

'''
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--train_data', type=str, default='coco8.yaml',
                        help='Path to dataset yaml')
    parser.add_argument('--test_data', type=str, default='coco8.yaml',
                        help='Path to dataset yaml')
    parser.add_argument('--output', type=str, default='runs/detect',
                        help='Path to output directory')
    args = parser.parse_args()

    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO11 example dataset for 100 epochs
    results = model.train(data=args.train_data, epochs=100, imgsz=640, save_dir=args.output)

    # Run inference with the YOLO11n model on the 'bus.jpg' image
    # results = model("path/to/bus.jpg"
    results = model(args.test_data, save_dir=args.output)

if __name__ =="__main__":
    main()