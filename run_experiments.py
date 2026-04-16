from hlwdetector.runner import ExperimentRunner


for config in ["configs/yolo11_h23.yaml", "configs/yolo26_h23.yaml", "configs/rtdetr_h23.yaml"]:
    ExperimentRunner(config).run_pipeline()