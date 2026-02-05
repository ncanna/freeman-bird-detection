# Create a base model (YOLO v11)
'''
This code trains a YOLO v11 model on a custom dataset (--train_data).
Then predict on a test image (--test_data).
Output is saved to (--output).

Example usage:
module load anaconda3/2023.03
conda activate bird_behavior

python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/models/yolo.py \
--train_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml \
--test_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/test/IMG_0163.MP4 \
--output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/output \
--epochs 5

# Prediction only
python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/models/yolo.py \
--test_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/test/IMG_0163.MP4 \
--output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/output/predict_only \
--predict_only

'''
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--train_data', type=str,
                        default='/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml',
                        help='Path to dataset yaml')
    parser.add_argument('--test_data', type=str,
                        default='/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/test/IMG_0163.MP4',
                        help='Path to test image/video/folder (data/test/ or clip.mp4)')
    parser.add_argument('--output_dir', type=str,
                        default='runs/detect',
                        help='Path to the directory where YOLO saves all outputs for this run')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640,
                        help='Image size is resized to imgsz x imgsz')
    parser.add_argument("--conf", type=float, default=0.25,
                        help='Confidence threshold for inference. Discard predictions with confidence < 0.25 by default')
    parser.add_argument('--yolo_model', type=str,
                        default='yolo11n.pt', help='Version of YOLO model to use. Defalt to v11'
                        help='Path to test image/video/folder (data/test/ or clip.mp4)')
    parser.add_argument("--predict_only", action='store_true',
                        help='Predict only, no training. Using default YOLO model without tuning')
    args = parser.parse_args()

    # ##### Load a COCO-pretrained YOLO11n model #####
    model = YOLO(args.yolo_model)

    # Only predict on the test data, no training
    if args.predict_only:
        print('# - Predict only, no training')
        results = model.predict(args.test_data, project=args.output_dir, 
                                name='detect', conf=args.conf, save=True)
        # Or this one?
        # results = model(args.test_data)
        return

    # ##### Train/fine tune the model #####
    # name='train': The subfolder name for this experiment inside project
    # save=True: write the output files
    train_results = model.train(data=args.train_data,
                                epochs=args.epochs,
                                imgsz=args.imgsz,
                                project=args.output_dir,
                                name='train')
    
    # ##### Run inference with the YOLO11n model #####
    best_pt = os.path.join(train_results.save_dir, 'weights', 'best.pt') # Get the best model
    # Check best.pt exists (avoid silent issues)
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"best.pt not found at {best_pt}")
    print('# - Load best YOLO model')
    model = YOLO(str(best_pt))
    results = model.predict(args.test_data, project=args.output_dir, 
                            name='detect', conf=args.conf, save=True)

    # ##### Save outputs including bounding box #####
    detect_dir = os.path.join(args.output_dir, 'detect')
    bbox_file = os.path.join(detect_dir, "pred_boxes.csv")
    with open(bbox_file, 'w') as f:
        f.write(f'frame,cls,conf,x1,y1,x2,y2\n')
        for frame_idx, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                f.write(f'{frame_idx},{int(b.cls[0])},{float(b.conf[0])},{int(x1)},{int(y1)},{int(x2)},{int(y2)}\n')

    print('# Done')

if __name__ =="__main__":
    main()