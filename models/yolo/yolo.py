from ultralytics import YOLO
from ultralytics.data.split import autosplit
import os


def main():
    model = YOLO("yolo11n.pt")
    model.train(data="./datasets/yolo_data_split/data.yaml", epochs=10)

    # path = "../datasets/yolo_data/data/images/test/"
    # pred_folder = os.listdir("../datasets/yolo_data/data/images/test")
    

if __name__ =="__main__":
    main()