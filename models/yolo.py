# Create a base model (YOLO v11)
'''
This code trains a YOLO v11 model on a custom dataset (--train_data).
Then predict on a test image (--test_data).
Output is saved to (--output_dir).

Example usage:
module load anaconda3/2023.03
conda activate bird_behavior

python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/models/yolo.py \
--train_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml \
--test_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/test/IMG_0165.MP4 \
--output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/output \
--epochs 5

# Prediction only, using yolo26n.pt model
python /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/models/yolo.py \
--test_data /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/test/IMG_0165.MP4 \
--output_dir /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/output/predict_only \
--yolo_model yolo26n.pt \
--predict_only

'''
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import argparse
import os
import cv2

'''
# Already implememnted in other scripts
def extract_frames():
    pass

def recreate_video(): # In video_utils.py
    pass
'''

def draw_bounding_box(img, result):
    '''
    Draw bounding box on the image
    Refer to this post: https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
    Params:
    - img: image frame to draw bounding box on
    - result: a result object
    
    '''
    if result.boxes is None or len(result.boxes) == 0: # If there are no bounding boxes, continue
        return img
    
    # Else draw bounding box
    annotator = Annotator(img)
    boxes = result.boxes
    for box in boxes:
        b = box.xyxy[0].tolist()   # get box coordinates in (left, top, right, bottom) format. Ensure to use plain list
        annotator.box_label(box=b, label='') # Might want to add label to the box in the future
    return annotator.result() # Get the image with bounding box


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--train_data', type=str,
                        default='/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/train.yaml',
                        help='Path to dataset yaml')
    parser.add_argument("--val_data", type=str,
                        default='/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/val.yaml',
                        help='Path to validation data yaml')
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
    # yolo26n.pt is the latest version of YOLO model (2026/02)
    parser.add_argument('--yolo_model', type=str,
                        default='yolo11n.pt',
                        help='Version of YOLO model to use. Default to v11n in the Ultralytics repo, or provide a customized YOLO model such as yolo_custom.yaml, last.pt, etc.')
    parser.add_argument("--validation_only", action='store_true',
                        help='Validation only, no training. Using selected YOLO model without tuning. Only --val_data (also need labels) is used if this flag is set.')
    parser.add_argument("--predict_only", action='store_true',
                        help='Predict only, no training. Using selected YOLO model without tuning. Only --test_data is used if this flag is set.')
    parser.add_argument("--draw_bounding_box", action='store_true',
                        help='Draw bounding box on the test image/video/folder')
    parser.add_argument("--recreate_video", action='store_true',
                        help='Recreate video with bounding box labeled')
    args = parser.parse_args()

    # ##### Load a COCO-pretrained YOLO11n model #####
    model = YOLO(args.yolo_model)

    # Only predict on the test data, no training
    if args.predict_only:
        print('# Prediction only, no training')
        results = model.predict(args.test_data,
                                project=args.output_dir, 
                                name='detect',
                                conf=args.conf,
                                save=True)
        # Or this one?
        # results = model(args.test_data)
        return

    # Only run validation on --val_data using the selected YOLO model without training
    if args.validation_only:
        print('# Validation only, no training')
        results = model.val(data=args.val_data,
                            project=args.output_dir, 
                            name='val',
                            conf=args.conf)
        return

    # ##### Train/fine tune the model #####
    print('# Training')
    # name='train': The subfolder name for this experiment inside project
    # save=True: write the output files
    # Details of params of train() are here: https://docs.ultralytics.com/modes/train/#train-settings
    # By default train() uses cpu or single gpu when possible
    # TODO: Add **kwargs to train() to allow more params
    train_results = model.train(data=args.train_data,
                                epochs=args.epochs,
                                imgsz=args.imgsz,
                                project=args.output_dir, # Name of the project directory where training outputs are saved
                                name='train',
                                seed=42  # Ensure reproducibility
                                )
    
    # ##### Run inference with the best YOLO model #####
    best_pt = os.path.join(train_results.save_dir, 'weights', 'best.pt') # Get the best model
    # Check best.pt exists (avoid silent issues)
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"best.pt not found at {best_pt}")
    print('# - Load the best YOLO model from training')
    model = YOLO(str(best_pt))
    # Details of params of predict() are here: https://docs.ultralytics.com/modes/predict/#inference-arguments
    # Returned results is a list of Result objects (if one image then a list of length 1)
    # Details of the result object are here: https://docs.ultralytics.com/modes/predict/#working-with-results
    results = model.predict(args.test_data, # Data can be an image path, video file, URL
                            project=args.output_dir, 
                            name='detect',
                            conf=args.conf, # the minimum confidence threshold for detection
                            save=True,
                            imgsz=args.imgsz, # the image size for inference
                            batch=1, # batch size for inference (only works when the source is a directory, video file, or .txt file)
                            visualize=False
                            )

    # ##### Save outputs including bounding box #####
    # Determine save_dir (attached to each Result); grab first one safely
    # Should be the same are args.output_dir/detect, but just in case
    first = None
    for r in results:
        first = r
        break
    if first is None:
        print('# - No results returned (empty input?) Exiting...')
        return

    save_dir = str(first.save_dir)
    bbox_file = os.path.join(save_dir, "pred_boxes.csv")
    with open(bbox_file, 'w') as f:
        f.write(f'frame,cls,conf,x1,y1,x2,y2\n')
        for frame_idx, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                f.write(f'{frame_idx},{int(b.cls[0])},{float(b.conf[0])},{int(x1)},{int(y1)},{int(x2)},{int(y2)}\n')

    # Draw bounding box on the test image/video/folder
    if args.draw_bounding_box:
        for frame_idx, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0: # If there are no bounding boxes, continue
                continue
            img = draw_bounding_box(r.orig_img, r)
            cv2.imwrite(f"{save_dir}/frame_{frame_idx:05d}.jpg", img)

    print('# Done')

if __name__ =="__main__":
    main()