# TeluxMMU ECG Classification

Pipeline klasifikasi penyakit ECG berbasis PyTorch. Repository ini sekarang diarahkan ke struktur modular: notebook hanya untuk preprocessing/eksplorasi, sedangkan eksperimen model dijalankan melalui CLI di `src.runner`.

## Setup

```bash
uv sync --all-extras
```

Jika memakai notebook, pilih kernel `teluxmmu-research`.

## Struktur Utama

```text
configs/      Konfigurasi eksperimen YAML
notebooks/    Notebook preprocessing yang masih dipertahankan
scripts/      Wrapper CLI dan nohup
src/          Pipeline modular training/evaluation/inference
summary/      Catatan audit dan planning
data/         Dataset lokal, dianggap read-only
runs/         Output eksperimen, metrics, checkpoint, logs
```

Notebook model lama di root sudah dihapus agar alur eksperimen tidak bercabang. Notebook preprocessing dipindahkan ke:

```text
notebooks/preprocessing.ipynb
```

## Dataset Policy

Folder `data/` tidak boleh diubah oleh pipeline training. Split, manifest, label mapping, cache, metrics, dan checkpoint disimpan di dalam folder run:

```text
runs/YYYYMMDD_HHMMSS_run_name/
```

Mapping label aktual:

```json
{
  "ECG_Normal": "Normal",
  "ECG_Abnormal": "Abnormal",
  "ECG_HistoryMI": "HistoryMI",
  "ECG_MI": "MI"
}
```

## Data Discovery

Validasi cepat tanpa training:

```bash
uv run python -m src.runner \
  --config configs/default.yaml \
  --run_name discovery_check \
  --dry_run_discovery
```

Output discovery tersimpan di:

```text
runs/*_discovery_check/artifacts/data_discovery.json
runs/*_discovery_check/artifacts/data_discovery.md
runs/*_discovery_check/artifacts/manifest.csv
runs/*_discovery_check/artifacts/label_mapping.json
```

Split train/validation/test memakai stratified group split berdasarkan hash representasi input. Ini mencegah exact duplicate image atau lead masuk ke split berbeda. Audit split disimpan di:

```text
runs/*/artifacts/splits/split_audit.json
```

## Training dan Evaluation

Default:

```bash
bash scripts/train_eval.sh --run_name mobilenet_v2_default
```

Progress training terbaru selalu ditulis ulang ke file root:

```text
training.log
```

File ini cocok dipantau dengan:

```bash
tail -f training.log
```

Salinan permanen per run juga disimpan di:

```text
runs/YYYYMMDD_HHMMSS_run_name/logs/training.log
```

Smoke test kecil:

```bash
bash scripts/train_eval.sh \
  --epochs 1 \
  --max_samples 40 \
  --run_name smoke_test
```

Default training tidak memakai early stopping:

```yaml
early_stopping: false
```

Jika ingin mengaktifkan early stopping untuk run tertentu:

```bash
bash scripts/train_eval.sh \
  --early_stopping \
  --run_name exp_with_early_stopping
```

LR scheduler default memakai cosine scheduler dengan minimum learning rate:

```yaml
scheduler: cosine
learning_rate: 0.0001
min_learning_rate: 0.000001
```

Override dari CLI:

```bash
bash scripts/train_eval.sh \
  --learning_rate 0.0001 \
  --min_learning_rate 0.000001 \
  --run_name exp_cosine_lr
```

## Nohup

Semua script di `scripts/` berjalan background dan tidak menulis output ke terminal. Setelah command dijalankan, terminal bisa langsung dipakai untuk pekerjaan lain atau ditutup.

```bash
bash scripts/run_nohup.sh
```

## Full Training Semua Skema

Untuk menjalankan semua eksperimen `train_eval` MobileNetV2 untuk seluruh skema input secara serial, satu per satu:

```bash
bash scripts/full_training_all.sh
```

Script ini menjalankan 10 experiment dalam satu background master process. Setiap skema baru mulai setelah skema sebelumnya selesai:

```text
single_raw_image
single_clean_image
single_12_lead
single_long_lead_ii
multibranch_12lead_longlead
multibranch_6lead_6lead_longlead
multibranch_13lead_individual
stacked_12lead_longlead
stacked_6lead_6lead_longlead
stacked_13lead_individual
```

Override hyperparameter:

```bash
EPOCHS=50 \
BATCH_SIZE=32 \
LEARNING_RATE=0.0001 \
bash scripts/full_training_all.sh
```

Setiap skema membuat folder run sendiri di `runs/`. File root `training.log` menunjukkan skema yang sedang berjalan saat ini; untuk log permanen tiap experiment, buka:

```text
runs/<run_name>/logs/training.log
```

Custom:

```bash
MODE=train_eval \
MODEL_NAME=mobilenet_v2 \
INPUT_SCHEME=single_clean_image \
RUN_NAME=mobilenet_v2_single_clean_image \
CONFIG=configs/default.yaml \
bash scripts/run_nohup.sh
```

## Output Run

Setiap run membuat:

```text
runs/YYYYMMDD_HHMMSS_run_name/
├── config_resolved.yaml
├── run_summary.md
├── logs/
├── artifacts/
├── metrics/
├── plots/
└── checkpoints/
```
