# Project Context: ECG Disease Classification Pipeline

## 1. Main Project Objective

This project aims to build a clean, modular, reproducible ECG disease classification pipeline whose experimental outputs are suitable for IEEE-style research reporting.

The pipeline must classify ECG data into at least four target classes:

1. `myocardial_infarction`
2. `abnormal_heartbeats`
3. `history_of_myocardial_infarction`
4. `normal`

The current default model is:

```text
MobileNetV2
```

However, the codebase must be designed flexibly so that additional models can be added later, such as ResNet, EfficientNet, DenseNet, Vision Transformer, custom CNN architectures, ECG-specific pretrained models, or hybrid architectures.

The final pipeline must support:

1. Training and evaluation.
2. Training only.
3. Evaluation only.
4. Inference only.
5. Multiple ECG input schemes.
6. IEEE-style metric reporting.
7. Reproducible experiment logging.
8. Running long experiments using `nohup`.

---

## 2. Critical Instructions for the AI Agent

Before creating or modifying any code, the AI agent must inspect and understand the current project.

The agent must:

1. Inspect the root directory.
2. Read all existing `README.md` files if available.
3. Find and read all existing notebooks with `.ipynb` extension.
4. Understand the current data folder structure.
5. Understand how the existing notebook loads data, preprocesses ECG images or signals, maps labels, trains models, and evaluates performance.
6. Reuse and refactor relevant notebook logic into modular Python scripts.
7. Avoid deleting, renaming, moving, or restructuring existing data files.
8. Avoid forcing the current dataset into a new format.
9. Preserve the current `data/` directory structure.
10. Write assumptions clearly when something cannot be inferred automatically.
11. Save new experiment artifacts under `runs/`, not inside `data/`.

The agent must not treat this document as a replacement for inspecting the project. This document is a specification. The actual implementation must be adapted to the current repository, current notebooks, and current dataset structure.

---

## 3. Existing Data Folder Policy

The project already has a `data/` folder.

The `data/` folder must be treated as read-only by default.

The agent must not:

```text
rename files inside data/
move files inside data/
delete files inside data/
overwrite metadata inside data/
force data/ into a new train/val/test structure
create new class folders inside data/
convert the whole dataset into a new layout inside data/
```

If the pipeline needs generated metadata, train/validation/test split files, preprocessing cache, or label mapping outputs, those files must be saved inside the current run directory under `runs/`.

For example:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/splits/
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/cache/
```

The pipeline must adapt to the existing dataset structure instead of modifying the dataset structure.

---

## 4. First Required Steps for the Agent

The first implementation step must be project discovery.

The agent must run or perform the equivalent of:

```bash
pwd
ls -lah
find . -maxdepth 3 -type f | sort
find . -name "*.ipynb"
find . -name "README.md" -o -name "readme.md"
```

Then the agent must inspect the existing data folder:

```bash
find data -maxdepth 4 -type d | sort
find data -maxdepth 4 -type f | head -100
find data -type f \( -name "*.csv" -o -name "*.xlsx" -o -name "*.json" -o -name "*.txt" \)
```

The agent must read existing notebooks to identify:

1. Dataset paths.
2. Data format.
3. ECG image or signal format.
4. Label format.
5. Class names.
6. Existing preprocessing steps.
7. Existing split strategy.
8. Existing model architecture.
9. Existing training loop.
10. Existing evaluation logic.
11. Existing errors, limitations, or assumptions.

Only after this discovery step should the agent create or refactor code.

---

## 5. Target Classes

The target classes for this project are:

```text
myocardial_infarction
abnormal_heartbeats
history_of_myocardial_infarction
normal
```

However, the actual dataset may use different label names.

Possible examples:

```text
MI
HMI
PMI
AMI
IMI
LMI
NORM
NORMAL
ABNORMAL
ARRHYTHMIA
abnormal_heartbeat
```

The agent must inspect the actual dataset and create an explicit label mapping.

Example:

```json
{
  "MI": "myocardial_infarction",
  "HMI": "history_of_myocardial_infarction",
  "ABNORMAL": "abnormal_heartbeats",
  "NORM": "normal"
}
```

The final label mapping used in a run must be saved to:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/label_mapping.json
```

If the mapping cannot be inferred with high confidence, the agent must document the assumed mapping in:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/run_summary.md
```

---

## 6. ECG Input Schemes

The pipeline must support three major input schemes:

1. Single input.
2. Multi-branch input.
3. Stacked input.

Each scheme must be implemented in a flexible way so that it can adapt to the current dataset format.

---

## 7. Single Input Schemes

Single input means the model receives one image or one tensor as the main input.

The required single-input variants are:

### 7.1 `single_raw_image`

Input is a raw ECG image.

This may include the full ECG layout before heavy preprocessing.

### 7.2 `single_clean_image`

Input is a cleaned or preprocessed ECG image.

Possible preprocessing may include:

```text
cropping
resizing
denoising
thresholding
background cleaning
normalization
contrast enhancement
```

The agent must not create cleaned data inside `data/` unless explicitly instructed. If preprocessing cache is required, save it under the run directory.

### 7.3 `single_long_lead_ii`

Input uses only the long lead II image or tensor.

This experiment is used to evaluate whether long lead II is sufficient for classification.

### 7.4 `single_12_lead`

Input uses the 12-lead ECG representation.

This may be:

1. A single ECG image containing all 12 leads.
2. A generated layout from separate lead images.
3. A tensor created from 12 lead inputs.

The implementation must follow the actual data format discovered from the current project.

---

## 8. Multi-Branch Input Schemes

Multi-branch input means different ECG lead groups are passed through separate model branches, then their extracted features are fused before classification.

The required multi-branch variants are:

### 8.1 `multibranch_12lead_longlead`

Input branches:

```text
Branch 1: 12-lead ECG
Branch 2: 1 long lead
```

### 8.2 `multibranch_6lead_6lead_longlead`

Input branches:

```text
Branch 1: first 6 leads
Branch 2: second 6 leads
Branch 3: 1 long lead
```

### 8.3 `multibranch_13lead_individual`

Input branches:

```text
Branch 1: lead 1
Branch 2: lead 2
Branch 3: lead 3
...
Branch 12: lead 12
Branch 13: long lead
```

The implementation may use:

1. Shared backbone weights across lead branches.
2. Independent lightweight branches.
3. A shared feature extractor followed by fusion layers.

The default approach should be simple and stable first.

---

## 9. Stacked Input Schemes

Stacked input means multiple ECG leads are combined into one tensor before being passed to the model.

The required stacked variants are:

### 9.1 `stacked_12lead_longlead`

Input:

```text
12 leads + 1 long lead
```

These are combined as one tensor or one structured image input.

### 9.2 `stacked_6lead_6lead_longlead`

Input:

```text
6 leads + 6 leads + 1 long lead
```

The grouped leads are stacked into one input representation.

### 9.3 `stacked_13lead_individual`

Input:

```text
13 individual lead inputs combined into one stacked representation
```

The stacked representation may use:

1. Channel stacking.
2. Grid layout.
3. Sequence-like stacking.
4. Tensor stacking.

The exact implementation must be based on the actual dataset format.

---

## 10. Default Model Requirement

The default model must be:

```text
MobileNetV2
```

MobileNetV2 must support:

1. Single-input classification.
2. Multi-branch feature extraction.
3. Stacked-input classification.

The default model configuration should include:

```text
model_name: mobilenet_v2
pretrained: true
freeze_backbone: false
num_classes: 4
```

If the current notebook uses TensorFlow/Keras, the implementation should preferably use:

```text
tf.keras.applications.MobileNetV2
```

If the current notebook uses PyTorch, the implementation should preferably use:

```text
torchvision.models.mobilenet_v2
```

The agent must choose one framework based on the current project and notebook. Do not mix TensorFlow and PyTorch unless the current project already does so or there is a strong technical reason.

---

## 11. Project Structure Policy

The agent must adapt to the current project structure.

Do not blindly replace the existing project structure. Do not move or restructure the existing `data/` folder.

The following structure is a recommended target for modular code only. It is not an instruction to rewrite the entire repository or alter the current dataset layout.

Recommended structure:

```text
project_root/
├── context.md
├── README.md
├── requirements.txt or pyproject.toml
├── notebooks/
│   └── existing_notebook.ipynb
├── configs/
│   ├── default.yaml
│   ├── mobilenet_v2_single_raw.yaml
│   ├── mobilenet_v2_single_clean.yaml
│   ├── mobilenet_v2_single_long_lead_ii.yaml
│   ├── mobilenet_v2_single_12lead.yaml
│   ├── mobilenet_v2_multibranch.yaml
│   └── mobilenet_v2_stacked.yaml
├── scripts/
│   ├── run_nohup.sh
│   ├── train_eval.sh
│   ├── eval_only.sh
│   └── inference_only.sh
├── src/
│   ├── runner.py
│   ├── config.py
│   ├── logger.py
│   ├── seed.py
│   ├── data/
│   │   ├── data_discovery.py
│   │   ├── datasets.py
│   │   ├── dataloaders.py
│   │   ├── transforms.py
│   │   └── lead_parser.py
│   ├── models/
│   │   ├── build_model.py
│   │   ├── mobilenet_v2.py
│   │   ├── single_input.py
│   │   ├── multibranch.py
│   │   └── stacked.py
│   ├── engine/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── metrics/
│   │   ├── classification_metrics.py
│   │   └── confusion_matrix.py
│   └── utils/
│       ├── io.py
│       ├── plots.py
│       └── environment.py
└── runs/
```

Important note about `data/`:

The `data/` folder already exists in the current project. It must not be moved, renamed, regenerated, or forced into the example structure above.

The agent must read the existing `data/` folder and create compatible dataloaders based on the discovered format.

If the current repository already has a different but valid structure, the agent may keep it and only add missing modules where appropriate.

---

## 12. Data Discovery and Dataset Handling

The pipeline must include a data discovery module.

Suggested file:

```text
src/data/data_discovery.py
```

This module must inspect the existing dataset and generate a report.

It must detect:

1. Whether the dataset uses folder-per-class format.
2. Whether the dataset already has train/validation/test splits.
3. Whether the dataset uses metadata files.
4. Whether metadata files contain image paths.
5. Whether metadata files contain signal paths.
6. Whether metadata files contain label columns.
7. Whether metadata files contain ECG lead columns.
8. Whether long lead II is available.
9. Whether 12-lead or 13-lead inputs are available.
10. Whether the existing notebook uses a specific data loading strategy.

The discovery report must be saved to:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/data_discovery.json
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/data_discovery.md
```

The `data_discovery.md` file should include:

```text
# Data Discovery Report

## Root Data Directory
- data_dir:

## Detected Structure
- structure_type:
- has_train_val_test:
- has_metadata:
- number_of_files:
- number_of_images:
- detected_classes:

## Metadata Files
- file:
- detected_columns:
- possible_label_column:
- possible_path_columns:
- possible_lead_columns:

## Recommended Loader Strategy
- loader_type:
- input_scheme_compatibility:

## Assumptions
- assumption_1:
- assumption_2:
```

---

## 13. Dataset Formats That Must Be Supported

The pipeline must support multiple possible dataset formats because the current `data/` folder may already use its own structure.

### 13.1 Folder-per-class Dataset

Example:

```text
data/
├── myocardial_infarction/
├── abnormal_heartbeats/
├── history_of_myocardial_infarction/
└── normal/
```

If this format is detected, the dataloader may infer labels from folder names.

### 13.2 Existing Split Folder Dataset

Example:

```text
data/
├── train/
├── val/
└── test/
```

If this format is detected, the dataloader must use the existing split.

The agent must not regenerate the split unless explicitly required.

### 13.3 Metadata CSV Dataset

Example:

```csv
image_path,label
path/to/image_001.png,myocardial_infarction
path/to/image_002.png,normal
```

If this format is detected, the dataloader must read image paths and labels from the metadata.

### 13.4 Multi-Input ECG Metadata Dataset

Example:

```csv
lead_1_path,lead_2_path,lead_3_path,lead_4_path,lead_5_path,lead_6_path,lead_7_path,lead_8_path,lead_9_path,lead_10_path,lead_11_path,lead_12_path,long_lead_path,label
```

If this format is detected, the dataloader must use the lead columns according to the selected `input_scheme`.

### 13.5 Notebook-Specific Dataset Format

If the existing notebook already implements a custom dataset format, the agent must prioritize that logic.

The notebook logic should be refactored into modular code instead of forcing the dataset into a generic format.

---

## 14. Dataset Splitting Policy

If the dataset already has train/validation/test splits, use the existing splits.

If the dataset does not have splits and auto-splitting is enabled, create split files under the run directory only.

Example:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/artifacts/splits/
├── train_split.csv
├── val_split.csv
└── test_split.csv
```

Do not create new split folders inside `data/`.

Do not move files into new split folders.

Default split ratios:

```text
train: 0.70
validation: 0.15
test: 0.15
```

Use stratified splitting whenever possible.

---

## 15. Data Configuration

The data config must be flexible and must not assume a fixed dataset structure.

Example:

```yaml
data:
  data_dir: data

  # Optional. Use only if the dataset already has split folders.
  train_dir: null
  val_dir: null
  test_dir: null

  # Optional. Use only if metadata files exist.
  metadata_path: null
  train_metadata_path: null
  val_metadata_path: null
  test_metadata_path: null

  # Optional. Allow auto-detection when null.
  image_path_column: null
  label_column: null

  # Optional for multi-input ECG.
  lead_columns: null
  long_lead_column: null

  # Split handling.
  auto_split_if_missing: true
  split_output_dir: artifacts/splits
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratified_split: true

  # Target classes.
  num_classes: 4
  class_names:
    - myocardial_infarction
    - abnormal_heartbeats
    - history_of_myocardial_infarction
    - normal

  # Image/input settings.
  image_size: 224
  input_scheme: single_raw_image
```

Configuration priority must be:

```text
CLI flags > config.yaml > default config
```

---

## 16. Pipeline Modes

The pipeline must support four modes:

### 16.1 `train_eval`

Default mode.

This mode trains the model and evaluates it on the test set in one run.

### 16.2 `train_only`

This mode only trains the model.

It must still save:

```text
training history
best checkpoint
last checkpoint
training logs
training curves
```

### 16.3 `eval_only`

This mode only evaluates a trained checkpoint.

It must require a checkpoint path.

Example:

```bash
python -m src.runner \
  --mode eval_only \
  --checkpoint_path runs/previous_run/checkpoints/best.pt
```

### 16.4 `inference_only`

This mode only performs inference.

It must support:

1. A single ECG image.
2. A folder of ECG images.
3. A metadata CSV for multi-input ECG inference.

Default mode:

```text
train_eval
```

---

## 17. Required CLI Flags

The main runner must be executable from the terminal.

Example:

```bash
python -m src.runner \
  --mode train_eval \
  --model_name mobilenet_v2 \
  --input_scheme single_raw_image \
  --data_dir data \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --image_size 224 \
  --run_name mobilenet_v2_single_raw
```

Required CLI flags:

```text
--mode
--config
--run_name
--model_name
--input_scheme
--data_dir
--train_dir
--val_dir
--test_dir
--metadata_path
--train_metadata_path
--val_metadata_path
--test_metadata_path
--inference_dir
--inference_file
--checkpoint_path
--num_classes
--class_names
--image_size
--batch_size
--epochs
--learning_rate
--weight_decay
--optimizer
--scheduler
--pretrained
--freeze_backbone
--num_workers
--seed
--device
--output_dir
--save_plots
--save_predictions
--early_stopping
--patience
```

The runner must support both YAML config and CLI override.

---

## 18. Default Configuration

Create a safe default configuration.

Example:

```yaml
project:
  name: ecg_disease_classification
  output_root: runs
  seed: 42

run:
  mode: train_eval
  run_name: mobilenet_v2_default

data:
  data_dir: data
  train_dir: null
  val_dir: null
  test_dir: null
  metadata_path: null
  train_metadata_path: null
  val_metadata_path: null
  test_metadata_path: null
  image_path_column: null
  label_column: null
  lead_columns: null
  long_lead_column: null
  auto_split_if_missing: true
  split_output_dir: artifacts/splits
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratified_split: true
  num_classes: 4
  class_names:
    - myocardial_infarction
    - abnormal_heartbeats
    - history_of_myocardial_infarction
    - normal
  image_size: 224
  input_scheme: single_raw_image

model:
  model_name: mobilenet_v2
  pretrained: true
  freeze_backbone: false
  dropout: 0.3

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.00001
  optimizer: adamw
  scheduler: cosine
  early_stopping: true
  patience: 10
  monitor_metric: val_macro_f1

runtime:
  device: auto
  num_workers: 4
  mixed_precision: true

logging:
  log_level: INFO
  save_stdout: true
  save_stderr: true

outputs:
  save_plots: true
  save_predictions: true
  save_checkpoints: true
```

---

## 19. Run Directory and Output Standard

All experiment outputs must be saved under:

```text
runs/
```

Each run must create a new folder with this format:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/
```

Example:

```text
runs/20260620_143015_mobilenet_v2_single_raw/
runs/20260620_151230_mobilenet_v2_multibranch_12lead_longlead/
```

Each run folder must contain:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/
├── config.yaml
├── config_resolved.yaml
├── run_summary.md
├── logs/
│   ├── train.log
│   ├── eval.log
│   ├── inference.log
│   └── runner.log
├── checkpoints/
│   ├── best.pt or best.keras
│   └── last.pt or last.keras
├── metrics/
│   ├── train_history.csv
│   ├── train_history.json
│   ├── test_metrics.csv
│   ├── test_metrics.json
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   └── predictions.csv
├── plots/
│   ├── confusion_matrix.png
│   ├── training_curve_loss.png
│   ├── training_curve_accuracy.png
│   ├── training_curve_f1.png
│   └── roc_pr_curve.png
└── artifacts/
    ├── data_discovery.json
    ├── data_discovery.md
    ├── label_mapping.json
    ├── model_summary.txt
    ├── splits/
    ├── cache/
    └── inference_examples/
```

Not every file is required for every mode. For example, `train_history.csv` is not required in `inference_only` mode.

---

## 20. Logging Requirements

The pipeline must use a proper logger instead of relying on `print`.

The logger must:

1. Save logs to files inside the run directory.
2. Save runner logs.
3. Save training logs.
4. Save evaluation logs.
5. Save inference logs.
6. Capture exceptions and full tracebacks.
7. Save resolved configuration.
8. Save start time and end time.
9. Save Python version.
10. Save important library versions.
11. Save CUDA/GPU information if available.
12. Save the exact command used to run the experiment.

The runner must not flood the terminal.

When running with `nohup`, all terminal output must be redirected to:

```text
runs/YYYYMMDD_HHMMSS_[run_name]/logs/runner.log
```

---

## 21. Nohup Runner Script

Create a shell script for running experiments using `nohup`.

Suggested file:

```text
scripts/run_nohup.sh
```

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-train_eval}"
MODEL_NAME="${MODEL_NAME:-mobilenet_v2}"
INPUT_SCHEME="${INPUT_SCHEME:-single_raw_image}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_${INPUT_SCHEME}}"
CONFIG="${CONFIG:-configs/default.yaml}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/${TIMESTAMP}_${RUN_NAME}"

mkdir -p "${RUN_DIR}/logs"

nohup python -m src.runner \
  --config "${CONFIG}" \
  --mode "${MODE}" \
  --model_name "${MODEL_NAME}" \
  --input_scheme "${INPUT_SCHEME}" \
  --run_name "${RUN_NAME}" \
  --output_dir "${RUN_DIR}" \
  > "${RUN_DIR}/logs/runner.log" 2>&1 &

echo "Started run: ${RUN_DIR}"
echo "Log file: ${RUN_DIR}/logs/runner.log"
```

The Python runner must also write internal logs to the same run directory.

---

## 22. IEEE-Style Metrics

Because ECG datasets are often imbalanced, the main metrics must use macro averaging.

Required metrics:

1. Overall Accuracy.
2. Balanced Accuracy.
3. Macro Precision.
4. Macro Recall.
5. Macro F1-Score.
6. Per-class Precision.
7. Per-class Recall.
8. Per-class F1-Score.
9. Support per class.
10. Confusion Matrix.

The primary model selection metric must be:

```text
validation macro F1-score
```

Do not select the best checkpoint using accuracy only.

Summary metrics table format:

```text
Model | Input Scheme | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score
```

Per-class metrics table format:

```text
Class | Precision | Recall | F1-Score | Support
```

All metrics must be saved in both CSV and JSON format when applicable.

---

## 23. Training Requirements

The training loop must save:

1. Training loss per epoch.
2. Validation loss per epoch.
3. Training accuracy per epoch.
4. Validation accuracy per epoch.
5. Validation macro precision per epoch.
6. Validation macro recall per epoch.
7. Validation macro F1-score per epoch.
8. Learning rate per epoch if scheduler is used.
9. Best checkpoint based on validation macro F1-score.
10. Last checkpoint at the final epoch.

The training history must be saved to:

```text
metrics/train_history.csv
metrics/train_history.json
```

The best checkpoint must be saved to:

```text
checkpoints/best.pt
```

or:

```text
checkpoints/best.keras
```

depending on the framework.

The last checkpoint must be saved to:

```text
checkpoints/last.pt
```

or:

```text
checkpoints/last.keras
```

---

## 24. Evaluation Requirements

Evaluation on the test set must:

1. Load the best checkpoint.
2. Compute all required metrics.
3. Compute per-class metrics.
4. Save confusion matrix as CSV.
5. Save confusion matrix as PNG.
6. Save prediction details.
7. Save a summary table suitable for IEEE reporting.

The `predictions.csv` file must contain at least:

```text
sample_id
input_path
true_label
predicted_label
confidence
correct
```

If possible, include class probabilities:

```text
prob_myocardial_infarction
prob_abnormal_heartbeats
prob_history_of_myocardial_infarction
prob_normal
```

For multi-input ECG, include relevant input paths or sample identifiers.

---

## 25. Inference Requirements

Inference mode must support:

1. One ECG image.
2. A folder of ECG images.
3. A metadata CSV for multi-input ECG.
4. A checkpoint path.

Example:

```bash
python -m src.runner \
  --mode inference_only \
  --model_name mobilenet_v2 \
  --input_scheme single_raw_image \
  --checkpoint_path runs/previous_run/checkpoints/best.pt \
  --inference_dir data/inference \
  --run_name inference_mobilenet_v2
```

Inference output must include:

```text
metrics/predictions.csv
logs/inference.log
```

Optional output:

```text
artifacts/inference_examples/
```

The pipeline must not modify the original inference files.

---

## 26. Plot Requirements

The pipeline must generate publication-friendly plots.

Required plots:

1. Confusion matrix.
2. Training loss curve.
3. Validation loss curve.
4. Training accuracy curve.
5. Validation accuracy curve.
6. Validation macro F1-score curve.

Optional plots:

1. ROC curve.
2. Precision-recall curve.
3. Per-class F1-score bar chart.
4. Class distribution plot.

All plots must be saved in:

```text
plots/
```

Use clear titles, readable labels, and high-resolution image output suitable for research reporting.

---

## 27. Reproducibility Requirements

The pipeline must support reproducible experiments.

It must save:

1. Random seed.
2. Resolved configuration.
3. Command-line arguments.
4. Library versions.
5. Python version.
6. CUDA/GPU information if available.
7. Git commit hash if the project is a Git repository.
8. Dataset discovery report.
9. Label mapping.
10. Split files if generated automatically.

The seed must control:

1. Python random.
2. NumPy random.
3. Framework random state.
4. Data loader shuffling when possible.

---

## 28. Run Summary

Each run must create:

```text
run_summary.md
```

Suggested content:

```text
# Run Summary

## Basic Information
- Run name:
- Timestamp:
- Mode:
- Model:
- Input scheme:
- Number of classes:
- Class names:

## Data
- Data directory:
- Detected data structure:
- Metadata used:
- Split strategy:
- Label mapping:
- Data assumptions:

## Training Configuration
- Epochs:
- Batch size:
- Learning rate:
- Optimizer:
- Scheduler:
- Image size:
- Pretrained:
- Freeze backbone:
- Early stopping:
- Monitor metric:

## Best Validation Result
- Best epoch:
- Validation macro F1-score:
- Validation macro precision:
- Validation macro recall:
- Validation accuracy:

## Test Result
- Test accuracy:
- Test balanced accuracy:
- Test macro precision:
- Test macro recall:
- Test macro F1-score:

## Output Files
- Best checkpoint:
- Last checkpoint:
- Metrics:
- Plots:
- Logs:
```

---

## 29. Experiment Comparison Targets

The pipeline must make it easy to compare MobileNetV2 across different ECG input schemes.

Required experiment combinations:

### 29.1 Single Input

```text
mobilenet_v2 + single_raw_image
mobilenet_v2 + single_clean_image
mobilenet_v2 + single_long_lead_ii
mobilenet_v2 + single_12_lead
```

### 29.2 Multi-Branch Input

```text
mobilenet_v2 + multibranch_12lead_longlead
mobilenet_v2 + multibranch_6lead_6lead_longlead
mobilenet_v2 + multibranch_13lead_individual
```

### 29.3 Stacked Input

```text
mobilenet_v2 + stacked_12lead_longlead
mobilenet_v2 + stacked_6lead_6lead_longlead
mobilenet_v2 + stacked_13lead_individual
```

Each experiment must create its own run directory and save its own metrics.

---

## 30. Implementation Principles

The agent must follow these principles:

1. Write modular code.
2. Avoid one large monolithic script.
3. Preserve the existing `data/` folder structure.
4. Read existing notebooks before implementing the pipeline.
5. Refactor useful notebook logic into scripts.
6. Use configuration files and CLI flags.
7. Save all outputs under `runs/`.
8. Use proper logging.
9. Use macro metrics as primary metrics.
10. Select best checkpoint based on validation macro F1-score.
11. Support `nohup`.
12. Support training, evaluation, and inference modes.
13. Make the code easy to extend to other models.
14. Make the code easy to extend to other input schemes.
15. Document all assumptions.

---

## 31. Expected Deliverables

The agent should create or update the following files when appropriate:

```text
context.md
README.md
configs/default.yaml
scripts/run_nohup.sh
scripts/train_eval.sh
scripts/eval_only.sh
scripts/inference_only.sh
src/runner.py
src/config.py
src/logger.py
src/seed.py
src/data/data_discovery.py
src/data/datasets.py
src/data/dataloaders.py
src/data/transforms.py
src/data/lead_parser.py
src/models/build_model.py
src/models/mobilenet_v2.py
src/models/single_input.py
src/models/multibranch.py
src/models/stacked.py
src/engine/train.py
src/engine/evaluate.py
src/engine/inference.py
src/metrics/classification_metrics.py
src/metrics/confusion_matrix.py
src/utils/io.py
src/utils/plots.py
src/utils/environment.py
```

If the current repository already has equivalent files, update the existing files instead of duplicating them.

---

## 32. Acceptance Criteria

The project is considered successful if:

1. The agent reads the current project structure.
2. The agent reads existing notebooks before implementation.
3. The agent does not modify the existing `data/` folder structure.
4. The pipeline can run with:

```bash
python -m src.runner --mode train_eval
```

5. The pipeline can run with:

```bash
bash scripts/run_nohup.sh
```

6. A run directory is automatically created under `runs/`.
7. A data discovery report is saved.
8. The final label mapping is saved.
9. Best and last checkpoints are saved.
10. Training history is saved as CSV and JSON.
11. Test metrics are saved as CSV and JSON.
12. Per-class metrics are saved.
13. Confusion matrix is saved as CSV and PNG.
14. Training curves are saved as PNG.
15. Logger output is saved to files.
16. `train_eval`, `train_only`, `eval_only`, and `inference_only` modes work.
17. MobileNetV2 works for at least one input scheme.
18. The code structure supports single-input, stacked-input, and multi-branch-input schemes.
19. Macro precision, macro recall, and macro F1-score are used as primary performance metrics.
20. Outputs are suitable for IEEE-style tables and figures.

---

## 33. Example Commands

Training and evaluation with default config:

```bash
python -m src.runner \
  --mode train_eval \
  --config configs/default.yaml \
  --run_name mobilenet_v2_default
```

Training only:

```bash
python -m src.runner \
  --mode train_only \
  --config configs/default.yaml \
  --run_name mobilenet_v2_train_only
```

Evaluation only:

```bash
python -m src.runner \
  --mode eval_only \
  --config configs/default.yaml \
  --checkpoint_path runs/previous_run/checkpoints/best.pt \
  --run_name mobilenet_v2_eval_only
```

Inference only:

```bash
python -m src.runner \
  --mode inference_only \
  --config configs/default.yaml \
  --checkpoint_path runs/previous_run/checkpoints/best.pt \
  --inference_dir data/inference \
  --run_name mobilenet_v2_inference_only
```

Run using `nohup`:

```bash
bash scripts/run_nohup.sh
```

Custom `nohup` run:

```bash
MODE=train_eval \
MODEL_NAME=mobilenet_v2 \
INPUT_SCHEME=single_12_lead \
RUN_NAME=mobilenet_v2_single_12_lead \
CONFIG=configs/default.yaml \
bash scripts/run_nohup.sh
```

---

## 34. Final Reminder for the Agent

Do not assume the dataset structure.

The current `data/` folder already exists and must be preserved.

The correct implementation flow is:

```text
Inspect project
Read README
Read notebooks
Inspect data/
Discover dataset format
Generate data discovery report
Build compatible dataloader
Implement modular pipeline
Run smoke test
Save all outputs under runs/
Document assumptions
```

The pipeline must adapt to the current project, not force the project to adapt to the pipeline.

---

## 35. Current Repository Alignment Notes

This section records the current repository state discovered before full pipeline implementation. It corrects the target specification so future implementation work does not accidentally ignore the existing project shape.

### 35.1 Current Code Structure

The current repository is notebook-first and utility-module based:

```text
project_root/
├── README.md
├── context.md
├── pyproject.toml
├── uv.lock
├── notebook-preprocessing.ipynb
├── notebook-modeling.ipynb
├── notebook-modeling-v2.ipynb
├── data/
├── utils/
├── reference/
├── summary/
└── progress_report/
```

The target modular structure in Section 11 is still valid, but implementation must be staged from this current structure rather than replacing it in one unsafe step.

### 35.2 Current Framework

The active project code uses PyTorch and torchvision through `utils/modeling.py`.

Current supported backbones in the utility code include:

```text
resnet18
mobilenet_v2
efficientnet_v2_s
```

Therefore, the full pipeline should be implemented in PyTorch first. TensorFlow/Keras reference notebooks should be treated as methodology references, not as the framework for the new production pipeline.

### 35.3 Current Dataset Structure

The current dataset is folder-based and already preprocessed:

```text
data/
├── raw/
│   ├── ECG_Abnormal/
│   ├── ECG_HistoryMI/
│   ├── ECG_MI/
│   ├── ECG_Normal/
│   └── gwbz3fsgp8-2.zip
└── preprocessed/
    ├── crop_ecg_area/
    ├── extracted_ecg_signal/
    ├── clean_ecg_signal/
    ├── cropped_leads/
    └── text_mask/
```

Current class folders map to target classes as follows:

```json
{
  "ECG_Normal": "Normal",
  "ECG_Abnormal": "Abnormal",
  "ECG_HistoryMI": "HistoryMI",
  "ECG_MI": "MI"
}
```

This mapping must be saved in every run as `artifacts/label_mapping.json`.

### 35.4 Current Input Sources

The existing project already contains the main inputs needed for the target schemes:

```text
single_clean_image:
  data/preprocessed/clean_ecg_signal/

single_raw_image:
  data/raw/

single_12_lead, single_long_lead_ii, stacked, and multibranch variants:
  data/preprocessed/cropped_leads/
```

Each sample in `cropped_leads` is expected to contain 13 lead images:

```text
01_lead_lead_I.png
02_lead_a_VR.png
03_lead_v_1.png
04_lead_v_4.png
05_lead_lead_II.png
06_lead_a_VL.png
07_lead_v_2.png
08_lead_v5.png
09_lead_lead_III.png
10_lead_a_VF.png
11_lead_v_3.png
12_lead_v_6.png
13_long_lead.png
```

### 35.5 Notebook Relocation Requirement

Root-level notebooks should be moved into a dedicated `notebooks/` directory during the cleanup phase:

```text
notebook-preprocessing.ipynb   -> notebooks/preprocessing.ipynb
notebook-modeling.ipynb        -> notebooks/modeling_resnet18.ipynb
notebook-modeling-v2.ipynb     -> notebooks/modeling_multi_backbone.ipynb
```

After relocation, notebooks must still resolve project-root imports and paths. Notebook cells should use a project-root helper or explicit root variable so data and output paths remain stable:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
```

Notebook-generated experiment artifacts should no longer be written into ad hoc output folders. They should use the same `runs/` output convention as the CLI pipeline.

### 35.6 Migration Rule

The migration should happen in small, verifiable steps:

1. Preserve `data/` exactly as-is.
2. Move notebooks only after imports and paths are made root-safe.
3. Keep `utils/` working during the transition.
4. Add `src/`, `configs/`, and `scripts/` without breaking existing notebooks.
5. Once the CLI pipeline is stable, notebooks should become analysis/orchestration entrypoints rather than the only implementation source.
