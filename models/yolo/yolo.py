from ultralytics import YOLO
from ultralytics.data.split import autosplit
import os
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    test_data = script_dir / ".." / ".." / "data" / "test" / "IMG_2473.MP4"
    test_data = test_data.resolve()
    model = YOLO("yolo11n.pt")
    results = model(test_data)    
    

if __name__ =="__main__":
    main()