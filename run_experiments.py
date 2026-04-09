from pprint import pprint

from pathlib import Path

from hlwdetector.runner import ExperimentRunner
from utilities.video_dataset_prep_tools import extract_frames_by_split


video_dir = Path("data/h23/Videos")  # directory containing annotated video dataset
split_json = Path("data/h23/split.json")  # COCO annotations JSON
out_dir = Path("data/h23/images")  # Path to extract split images

#extract_frames_by_split(split_json, video_dir, out_dir)

yolo11_config_yaml = "configs/yolo11_h23_resume_yolo11_h23_20260402_004059.yaml"
runner = ExperimentRunner(yolo11_config_yaml)
runner.run()
