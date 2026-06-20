# Full Pipeline Implementation Plan

Dokumen ini adalah planning kerja untuk merapihkan repository `TeluxMMU` menjadi pipeline klasifikasi ECG yang modular, reproducible, dan siap eksperimen. Rencana ini mengikuti `context.md`, tetapi disesuaikan dengan struktur project aktual.

## 1. Prinsip Utama

1. Jangan mengubah struktur, isi, nama file, atau folder di `data/`.
2. Gunakan PyTorch/torchvision sebagai framework utama karena kode aktif saat ini ada di `utils/modeling.py`.
3. Jadikan notebook sebagai entrypoint analisis/orchestration, bukan satu-satunya tempat implementasi logic.
4. Semua output eksperimen baru harus masuk ke `runs/`.
5. Refactor dilakukan bertahap agar notebook lama tetap bisa dipakai selama transisi.

## 2. Struktur Target

Struktur akhir yang akan dibuat:

```text
TeluxMMU/
├── configs/
│   ├── default.yaml
│   ├── mobilenet_v2_single_clean_image.yaml
│   ├── mobilenet_v2_single_raw_image.yaml
│   ├── mobilenet_v2_multibranch_12lead_longlead.yaml
│   └── mobilenet_v2_stacked_13lead_individual.yaml
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── modeling_resnet18.ipynb
│   └── modeling_multi_backbone.ipynb
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
│   ├── models/
│   ├── engine/
│   ├── metrics/
│   └── utils/
├── runs/
├── data/
├── utils/
├── reference/
├── summary/
├── progress_report/
├── README.md
├── context.md
├── pyproject.toml
└── uv.lock
```

`utils/` tidak langsung dihapus. Folder ini tetap dipertahankan sebagai compatibility layer sampai semua logic penting sudah dipindahkan atau dibungkus oleh `src/`.

## 3. Fase 1 - Repository Cleanup dan Notebook Relocation

Tujuan fase ini adalah merapihkan root project tanpa merusak notebook.

Langkah kerja:

1. Buat folder `notebooks/`.
2. Pindahkan notebook root:
   - `notebook-preprocessing.ipynb` menjadi `notebooks/preprocessing.ipynb`
   - `notebook-modeling.ipynb` menjadi `notebooks/modeling_resnet18.ipynb`
   - `notebook-modeling-v2.ipynb` menjadi `notebooks/modeling_multi_backbone.ipynb`
3. Update cell awal notebook agar path tetap mengarah ke project root:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
```

4. Ganti path hardcoded seperti `data/...` menjadi `DATA_DIR / ...` jika notebook dipakai untuk output baru.
5. Hindari menjalankan ulang preprocessing berat saat migrasi. Validasi cukup dengan import/sanity cell.
6. Update README agar menyebut lokasi notebook baru.

Output fase ini:

```text
notebooks/
README.md updated
notebook imports still work
```

## 4. Fase 2 - Data Discovery dan Manifest

Tujuan fase ini adalah membuat pipeline memahami data aktual tanpa memodifikasi `data/`.

File yang dibuat:

```text
src/data/data_discovery.py
src/data/lead_parser.py
src/data/manifest.py
```

Fungsi utama:

1. Deteksi folder kelas di:
   - `data/raw`
   - `data/preprocessed/clean_ecg_signal`
   - `data/preprocessed/cropped_leads`
2. Hitung jumlah sample per kelas.
3. Validasi ketersediaan 13 lead per sample.
4. Buat manifest in-memory atau manifest per run di:

```text
runs/YYYYMMDD_HHMMSS_run_name/artifacts/manifest.csv
```

Kolom manifest minimal:

```text
sample_id
class_folder
target_label
raw_image_path
clean_image_path
lead_01_path
lead_02_path
...
lead_13_path
has_all_leads
```

Output fase ini:

```text
artifacts/data_discovery.json
artifacts/data_discovery.md
artifacts/label_mapping.json
artifacts/manifest.csv
```

## 5. Fase 3 - Config dan Run Directory Standard

Tujuan fase ini adalah membuat konfigurasi dan output konsisten.

File yang dibuat:

```text
configs/default.yaml
src/config.py
src/utils/io.py
src/logger.py
src/seed.py
src/utils/environment.py
```

Aturan:

1. CLI override harus menang atas YAML.
2. Setiap run membuat folder:

```text
runs/YYYYMMDD_HHMMSS_run_name/
```

3. Setiap run menyimpan:

```text
config.yaml
config_resolved.yaml
run_summary.md
logs/
artifacts/
metrics/
plots/
checkpoints/
```

4. Seed dikontrol untuk Python, NumPy, dan PyTorch.
5. Environment info disimpan: Python, torch, torchvision, CUDA/GPU, git commit.

## 6. Fase 4 - Dataset dan Dataloader

Tujuan fase ini adalah memindahkan logic dataset dari `utils/modeling.py` ke modul yang bisa dipakai CLI.

File yang dibuat:

```text
src/data/datasets.py
src/data/dataloaders.py
src/data/transforms.py
```

Input scheme tahap pertama:

1. `single_clean_image`
2. `single_raw_image`
3. `single_long_lead_ii`
4. `single_12_lead`
5. `stacked_13lead_individual`
6. `multibranch_6lead_6lead_longlead`

Split strategy:

1. Split dibuat dari `sample_id`, bukan dari object dataset per scheme.
2. Default split 70/15/15.
3. Stratified split dipakai jika memungkinkan.
4. Split disimpan di:

```text
artifacts/splits/train_split.csv
artifacts/splits/val_split.csv
artifacts/splits/test_split.csv
```

## 7. Fase 5 - Model Layer

Tujuan fase ini adalah membuat model mudah diganti.

File yang dibuat:

```text
src/models/build_model.py
src/models/mobilenet_v2.py
src/models/single_input.py
src/models/multibranch.py
src/models/stacked.py
```

Prioritas implementasi:

1. MobileNetV2 sebagai default sesuai `context.md`.
2. Pertahankan dukungan ResNet-18 dan EfficientNetV2-S sebagai optional karena sudah ada di `utils/modeling.py`.
3. Untuk input 39-channel atau 18-channel, gunakan adapter yang jelas dan terdokumentasi.
4. Untuk multi-branch, setiap branch memproses kelompok lead lalu feature digabung sebelum classifier.

## 8. Fase 6 - Training, Evaluation, dan Metrics

File yang dibuat:

```text
src/engine/train.py
src/engine/evaluate.py
src/engine/inference.py
src/metrics/classification_metrics.py
src/metrics/confusion_matrix.py
src/utils/plots.py
```

Training harus menyimpan:

```text
checkpoints/best.pt
checkpoints/last.pt
metrics/train_history.csv
metrics/train_history.json
plots/training_curve_loss.png
plots/training_curve_accuracy.png
plots/training_curve_f1.png
```

Best checkpoint dipilih berdasarkan:

```text
validation macro F1-score
```

Evaluation harus menyimpan:

```text
metrics/test_metrics.csv
metrics/test_metrics.json
metrics/per_class_metrics.csv
metrics/confusion_matrix.csv
metrics/predictions.csv
plots/confusion_matrix.png
```

Metrics utama:

```text
accuracy
balanced_accuracy
macro_precision
macro_recall
macro_f1
per_class_precision
per_class_recall
per_class_f1
support
```

## 9. Fase 7 - Runner dan Scripts

File yang dibuat:

```text
src/runner.py
scripts/run_nohup.sh
scripts/train_eval.sh
scripts/eval_only.sh
scripts/inference_only.sh
```

Mode yang harus tersedia:

```text
train_eval
train_only
eval_only
inference_only
```

Command minimal yang harus lolos:

```bash
uv run python -m src.runner --mode train_eval --config configs/default.yaml
```

Nohup minimal:

```bash
bash scripts/run_nohup.sh
```

Terminal output harus ringkas. Log detail masuk ke:

```text
runs/YYYYMMDD_HHMMSS_run_name/logs/runner.log
```

## 10. Fase 8 - Notebook Reconnection

Setelah CLI stabil, notebook dipakai untuk:

1. Menjalankan preprocessing preview.
2. Membaca hasil `runs/`.
3. Membuat tabel dan visualisasi perbandingan.
4. Membandingkan input scheme dan backbone.

Notebook tidak boleh lagi menyimpan output eksperimen ke folder acak. Notebook harus membaca atau menulis ke `runs/`.

## 11. Fase 9 - README dan Dokumentasi

README perlu diperbarui dengan:

1. Setup `uv sync --all-extras`.
2. Struktur folder baru.
3. Lokasi notebook baru.
4. Dataset policy: `data/` read-only.
5. Contoh command CLI.
6. Contoh nohup.
7. Penjelasan output `runs/`.
8. Label mapping aktual.

## 12. Urutan Eksekusi yang Direkomendasikan

1. Revisi `context.md` dengan alignment aktual.
2. Buat planning ini.
3. Pindahkan notebook ke `notebooks/` dan perbaiki path imports.
4. Tambahkan `configs/default.yaml`.
5. Tambahkan modul run directory, logger, seed, dan config resolver.
6. Tambahkan data discovery dan manifest.
7. Tambahkan dataset/dataloader berbasis manifest.
8. Tambahkan model builder MobileNetV2.
9. Tambahkan train/eval metrics.
10. Tambahkan `src/runner.py`.
11. Tambahkan shell scripts.
12. Smoke test data discovery.
13. Smoke test training 1 epoch kecil.
14. Update README.

## 13. Validasi Minimum

Sebelum dianggap selesai, minimal harus lolos:

```bash
uv run python -m src.runner --mode train_eval --config configs/default.yaml --epochs 1 --run_name smoke_test
```

Output yang harus ada:

```text
runs/*_smoke_test/
├── config_resolved.yaml
├── run_summary.md
├── logs/runner.log
├── artifacts/data_discovery.json
├── artifacts/label_mapping.json
├── artifacts/manifest.csv
├── metrics/train_history.csv
├── metrics/test_metrics.json
└── checkpoints/best.pt
```

## 14. Risiko dan Catatan

1. `data/raw/ECG_MI` saat inspeksi berisi 239 file, sementara nama folder dataset asli menyebut 240 pasien. Pipeline harus mencatat jumlah aktual, bukan memaksakan angka dari nama folder.
2. `utils/template/notebook-modeling-template.ipynb` saat ini kosong. File ini bisa dipertahankan sementara atau diganti nanti jika template notebook memang dibutuhkan.
3. `notebook-preprocessing.ipynb` sedang tercatat modified di git sebelum planning ini dibuat. Jangan overwrite perubahan tersebut tanpa inspeksi khusus.
4. Migrasi notebook harus menjaga backward compatibility import `utils.*` sampai `src.*` stabil.
