# retinal-ai

A lightweight retinal imaging project comparing 200-degree ultra-widefield views vs 45-degree crops for multi-class disease classification.

This repo is organized so you can quickly find:
- data prep code
- model training and analysis scripts
- extracted features
- saved model weights
- experiment outputs

## Project map (where everything is)

### Top level
- `requirements.txt`: Python dependencies
- `uwf.csv`: main dataset metadata/input table

### Experiment outputs and extracted features
- `balanced-700/`: outputs for the balanced 700-image setup
	- `analysis_outputs/`: reports and interpretation writeups
	- `balanced_fundus_features/`: handcrafted features for 45-degree/fundus-style images
	- `balanced_optus_features/`: handcrafted features for 200-degree/optus-style images
- `imbalanced-2031/`: outputs for the larger imbalanced setup
	- `analysis_outputs/`: majority-class and performance summaries
	- `imbalanced_fundus_features/`: extracted features (fundus-style)
	- `imbalanced_optus_features/`: extracted features (optus-style)

### Source code
- `src/data_prep/`: image preprocessing and synthetic cropping pipeline
	- `preprocess.py`, `crop.py`, `synthetic_crop.py`
- `src/ai/`: deep learning models and explainability tools
	- `cnn.py`: CNN/EfficientNet workflows
	- `gradcam.py`: Grad-CAM visual explanations
	- `occlusion.py`: occlusion sensitivity analysis
	- `best_model*.pth`: trained model checkpoints
	- `enhanced_overlays/`, `occlusion_results/`: visualization outputs
- `src/scripts/`: classical ML and class-level analysis
	- `logistic.py`: Logistic Regression pipeline
	- `randomforest.py`: Random Forest pipeline
	- `class_analysis.py`: per-class analysis utilities
- `src/data/`, `src/models/`, `src/interpret/`: supporting folders for data, models, and interpretation workflows

## Quick start

1. Install dependencies from `requirements.txt`.
2. Start with preprocessing in `src/data_prep/`.
3. Run model pipelines from `src/scripts/` (classical ML) or `src/ai/` (deep learning).
4. Review outputs in `balanced-700/analysis_outputs/` or `imbalanced-2031/analysis_outputs/`.

## Workflow

Use this order:
1. `src/data_prep/preprocess.py`
2. `src/data_prep/synthetic_crop.py`
3. `src/scripts/logistic.py` and `src/scripts/randomforest.py`
4. `src/ai/cnn.py`
5. `src/ai/gradcam.py` and `src/ai/occlusion.py`

That flow takes you from cleaned images to baseline models to deep learning and explainability.
