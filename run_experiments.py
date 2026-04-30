import time

from hlwdetector.runner import ExperimentRunner


start_total = time.time()

#for config in ["configs/yolo11_h23.yaml", "configs/yolo26_h23.yaml", "configs/rtdetr_h23.yaml"]:
for config in ["configs/rtdetr_h23.yaml"]:
    start = time.time()
    ExperimentRunner(config).run_pipeline()
    elapsed = (time.time() - start) / 60
    print(f"[{config}] finished in {elapsed:.2f} min")

total_elapsed = (time.time() - start_total) / 60
print(f"\nTotal time elapsed: {total_elapsed:.2f} min")