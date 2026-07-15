from pathlib import Path
from hlwdetector.hp_optimizer import HPOptimizer

hpo_config_path = Path("configs/hpo/yolo26_hpo.yaml")
optimizer = HPOptimizer(hpo_config_path)
optimizer.run_study()

#TODO
# Test run