# Running hyperparameter studies on PACE

This repository already has Optuna study configs for YOLO, RT-DETR, Swin, and DETR under `configs/hpo/`. A study trains several trials with different sampled hyperparameters and ranks them by the configured validation metric (`map50_95` by default). It does not extract frames or create annotation/split files.

## 1. Prepare the dataset

The selected HPO config needs three inputs:

- COCO annotations JSON for the annotated frames.
- Split JSON with `train`, `val`, and `test` arrays containing video filename stems.
- A flat directory containing the extracted frame images referenced by the COCO JSON.

Keep these files in PACE-accessible project/scratch storage. The default HPO configs refer to the existing h23 subset; pass all three paths to use the video set you are annotating. The COCO file, split, and extracted frames must refer to the same dataset.

## 2. Prepare the PACE environment

Log in to the correct PACE cluster and prepare a Python environment where this repository's dependencies are installed. Ensure `make`, Python, PyTorch, and the dependencies from `requirements.txt` are available. Transfer/clone the repo and dataset to storage visible to compute nodes. Do not train on a login node. For batch jobs, pass `PACE_ENV=/path/to/venv` to activate that virtualenv in the job, or set `PACE_MODULES` to space-separated module names to load before activation. The batch shell does not rely on your interactive `.bashrc`.

## 3. Validate inputs before requesting a long job

From the repository root, run the validation target on a node/environment with the Python dependencies installed:

```bash
make validate-hyperparam MODEL=swin \
  COCO_JSON=/path/to/annotations.json \
  SPLIT_JSON=/path/to/split.json \
  IMAGES_DIR=/path/to/frames
```

This checks the selected model's HPO config and the existence of all dataset paths without starting training. The existing configs use `device: "0"` for the GPU-oriented studies.

## 4. Submit the study

Recommended one-GPU batch submission (study trials run sequentially on that GPU):

```bash
make submit-hyperparam MODEL=swin PACE_ACCOUNT=<your-PACE-account> \
  COCO_JSON=/path/to/annotations.json \
  SPLIT_JSON=/path/to/split.json \
  IMAGES_DIR=/path/to/frames
```

Optional scheduler settings are `PACE_QOS`, `SLURM_TIME`, `SLURM_CPUS`, `SLURM_MEMORY`, and `SLURM_GPUS`. QoS/partition conventions and account names vary by PACE cluster/account; supply a QoS only when your PACE instructions require one. The job log is written to `logs/hpo-<job-id>.out`.

Use `MODEL=yolo`, `rtdetr`, `swin`, or `detr`. The matching defaults are `configs/hpo/yolo26_hpo.yaml`, `rtdetr_hpo.yaml`, `swin_hpo.yaml`, and `detr_hpo.yaml`. Set `HPO_CONFIG=path/to/custom.yaml` to adjust trial count, epochs, model weights, or search ranges. Use `make hyperparam ...` only from an interactive compute allocation; `make submit-hyperparam ...` submits a batch job.

Check queue status with `squeue -u "$USER"`, follow the log with `tail -f logs/hpo-<job-id>.out`, and cancel with `scancel <job-id>`. Results are written under `outputs/hpo/`.

## Notes

- A full study can take a long time: the checked-in configs request 8 trials for Swin, DETR, and RT-DETR, with 20–50 epochs per trial. Review/adjust `optimize_args.n_trials` and `hyperparameters.static.epochs` in your custom config before submitting.
- Vanilla DETR is explicitly noted in its config as slow to converge; short trials may not be informative.
- HPO is a hyperparameter search, not a single definitive training run. After choosing settings, use a normal experiment config for a final training run on the desired full dataset.
- This workflow submits jobs but does not run them automatically in CI or on your laptop.
