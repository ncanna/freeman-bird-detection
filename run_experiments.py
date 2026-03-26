from pprint import pprint

from hlwdetector.runner import ExperimentRunner
runner = ExperimentRunner("configs/yolo11_h23.yaml")


runner.run()
