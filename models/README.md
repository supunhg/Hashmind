# Models Directory

This directory will contain:

## Pre-trained Models (Phase 2+)

- `hashmind_v1.onnx` - Primary XGBoost model (ONNX format)
- `hashmind_ensemble_v1.pkl` - Ensemble model
- `feature_scaler.pkl` - Feature normalization parameters

## Model Metadata

- `model_info.json` - Model version, training date, accuracy metrics
- `feature_names.json` - Feature list and importance scores

## Training Checkpoints

- `checkpoints/` - Intermediate training checkpoints
- `experiments/` - Experiment tracking and results

## Usage

Models will be automatically downloaded on first use when installing with `pip install hashmind[ml]`.

For custom model training, see the training scripts in `/scripts/train_model.py`.

---

**Note**: ML features are planned for Phase 2. The current version (0.1.0) uses heuristic detection only.
