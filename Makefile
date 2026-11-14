SHELL := /bin/bash
PYTHON ?= python
MODEL ?= yolo
HPO_CONFIG ?=
COCO_JSON ?=
SPLIT_JSON ?=
IMAGES_DIR ?=
PACE_ACCOUNT ?=
PACE_QOS ?=
PACE_ENV ?=
PACE_MODULES ?=
SLURM_TIME ?= 12:00:00
SLURM_CPUS ?= 8
SLURM_MEMORY ?= 32G
SLURM_GPUS ?= 1

.PHONY: help hyperparam validate-hyperparam hyperparam-yolo hyperparam-rtdetr hyperparam-swin hyperparam-detr submit-hyperparam

help:
	@printf '%s\n' \
	  'Hyperparameter optimization (run hyperparam inside a Slurm allocation):' \
	  '  make hyperparam MODEL=swin' \
	  '  make hyperparam MODEL=detr COCO_JSON=/path/data.json SPLIT_JSON=/path/split.json IMAGES_DIR=/path/frames' \
	  '  make validate-hyperparam MODEL=swin COCO_JSON=... SPLIT_JSON=... IMAGES_DIR=...' \
	  '  make submit-hyperparam MODEL=swin PACE_ACCOUNT=<your-account> COCO_JSON=... SPLIT_JSON=... IMAGES_DIR=...' \
	  'Models: yolo, rtdetr, swin, detr. Override HPO_CONFIG to use a custom study YAML.'

hyperparam:
	HPO_MODEL="$(MODEL)" HPO_CONFIG="$(HPO_CONFIG)" COCO_JSON="$(COCO_JSON)" SPLIT_JSON="$(SPLIT_JSON)" IMAGES_DIR="$(IMAGES_DIR)" $(PYTHON) run_hpo.py

validate-hyperparam:
	HPO_MODEL="$(MODEL)" HPO_CONFIG="$(HPO_CONFIG)" COCO_JSON="$(COCO_JSON)" SPLIT_JSON="$(SPLIT_JSON)" IMAGES_DIR="$(IMAGES_DIR)" $(PYTHON) run_hpo.py --validate-only

hyperparam-yolo:
	$(MAKE) hyperparam MODEL=yolo

hyperparam-rtdetr:
	$(MAKE) hyperparam MODEL=rtdetr

hyperparam-swin:
	$(MAKE) hyperparam MODEL=swin

hyperparam-detr:
	$(MAKE) hyperparam MODEL=detr

submit-hyperparam:
	@test -n "$(PACE_ACCOUNT)" || { echo 'Set PACE_ACCOUNT to your PACE project/account.' >&2; exit 2; }
	@mkdir -p logs
	sbatch --account="$(PACE_ACCOUNT)" $(if $(PACE_QOS),--qos="$(PACE_QOS)",) --nodes=1 --ntasks=1 --cpus-per-task="$(SLURM_CPUS)" --mem="$(SLURM_MEMORY)" --gres="gpu:$(SLURM_GPUS)" --time="$(SLURM_TIME)" --output="logs/hpo-%j.out" --export="ALL,HPO_MODEL=$(MODEL),HPO_CONFIG=$(HPO_CONFIG),COCO_JSON=$(COCO_JSON),SPLIT_JSON=$(SPLIT_JSON),IMAGES_DIR=$(IMAGES_DIR),PACE_ENV=$(PACE_ENV),PACE_MODULES=$(PACE_MODULES)" sbatch_files/hyperparam.sbatch
