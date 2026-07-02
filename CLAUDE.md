# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solar Flare Prediction — a deep learning research project that predicts X-class solar flares from NASA AIA (Atmospheric Imaging Assembly) EUV imagery. Uses 7-channel (94, 131, 171, 193, 211, 304, 335 Å), 512×512 pixel AARP (Active Area Reduced Polarimetry) sequences with Vision Transformer as the primary model, plus AlexNet and XGBoost as secondary models. Explainability via Integrated Gradients (Captum) and SHAP values.

## Environment Setup

```bash
conda install -c conda-forge mamba
conda create -n torch-tf-312
mamba env update --file torch-tf-312.environment.yml
conda activate torch-tf-312
```

Configure DVC remote (Google Drive):
```bash
gdown "1257ELI7CIC3_wobMXoOVw7r2x71obomq" -O .dvc/config.local
dvc pull
```

## Common Commands

All scripts run as `python -m module.path` from repo root.

**Data pipeline:**
```bash
python -m src.data_single --download --process --select --extract --json --stats
dvc repro              # Run full DVC pipeline
dvc repro <stage>      # Reproduce a specific stage
```

**Training:**
```bash
python -m src.torch.vit.train         # Vision Transformer (primary)
python -m src.torch.train_alexnet     # AlexNet (PyTorch)
python -m src.torch.xgb_train        # XGBoost
python -m src.tensorflow.train_alexnet --json-path solar_dataset.json --stats-file stats.pkl
```

**Evaluation & analysis:**
```bash
python -m src.torch.vit.test --subset test
python -m src.torch.vit.kshap && python -m src.torch.vit.analyze_shap
python -m src.torch.vit.class_wise_distribution
python -m src.torch.vit.attributions_analyze
```

**Tests:**
```bash
python -m unittest tests.vit.test_utils
python -m unittest tests.vit.test_ig.TestSingleAARP.read_data
```

**JSON utilities:**
```bash
python -m src.check_json    # Validate dataset JSON paths
python -m src.clean_json    # Remove missing paths from JSON
```

**Documentation (local preview):**
```bash
mkdocs serve --livereload --dev-addr localhost:8001
```

## Architecture

```
aarp_ml/          # Core reusable library
  config.py       # DatasetPaths and TF/NumPy config
  dataset.py      # aarp_dataset (JSON-backed data loader)
  torch/
    base.py       # BaseModel class
    model.py      # DeepFlare_ViT, SaveBestModel
    dataset.py    # aia_euv dataset, AIALogTransform
    trained_model.py  # Model loading utilities
  integrated_gradients/  # IG implementation

src/              # Pipeline scripts (run via python -m)
  data_single.py  # Main data download → process → split → stats pipeline
  torch/
    vit/          # All ViT-related training, analysis, explainability
      train.py    # Main training entry point
      config.py   # TrainingConfig dataclass
      utils.py    # Shared loading/preprocessing utilities
      ig.py       # Integrated Gradients

tests/vit/        # Validation tests for IG and image utilities
docs/stages/      # Per-stage documentation (11 DVC stages)
docs/code/        # Auto-generated code documentation
```

**Key data flow:** Raw FITS files → `data_single.py` → `solar_dataset.json` + `stats.pkl` → `aia_euv` dataset (with `AIALogTransform`) → model training → `outputs/<wandb-run-id>/trained_model.pth`

**Dataset config:** Paths and split sizes are set in `DatasetPaths` in `aarp_ml/config.py`. Raw data lives under `/data/linn/E8/{compressed,extracted}/{pos,neg}/`.

## DVC Pipeline Stages

11 stages defined in `dvc.yaml`: `data-single`, `create-distribution`, `confusion-matrix-val`, `prediction-plots`, `plot-class-dist`, `plot_contour_image_grid`, `attribution_analysis`, `create_movie`, `xgb-feat-importance`, `kshap-stats`, `kshap-analyze`.

Trained model checkpoints referenced in stages: ViT at `outputs/glad-shape-197/trained_model.pth`, XGBoost at `outputs/deep-spaceship-10/best_xgboost_model.json`.

## Custom Dependencies

Two private packages installed from GitHub via the environment YAML:
- `astro-utils` — solar physics utilities
- `ml-scripts` — ML training utilities

## Active Development

Current branch `feat/per-aarp-analysis` is building per-AARP sequence analysis (Phase 1). Planning docs are in `docs/planning/` (excluded from the public mkdocs site).
