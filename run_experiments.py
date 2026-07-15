import time

from hlwdetector.runner import ExperimentRunner


start_total = time.time()

#for config in ["configs/yolo11_h23_full.yaml", "configs/yolo26_h23_full.yaml", "configs/rtdetr_h23_full.yaml"]:
for config in ["configs/experiment/yolo26_h23_subset.yaml"]:
    start = time.time()
    runner = ExperimentRunner(config)
    #runner.run_pipeline()
    runner.train()
    elapsed = (time.time() - start) / 60
    print(f"[{config}] finished in {elapsed:.2f} min")

total_elapsed = (time.time() - start_total) / 60
print(f"\nTotal time elapsed: {total_elapsed:.2f} min")