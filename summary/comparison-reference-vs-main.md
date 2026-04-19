# Perbandingan: Reference Notebooks vs. Main Project

> Dibuat: 2026-04-19  
> Referensi utama: `reference/kode-tugas-akhir-preprocess-terbaik-3-model.ipynb`  
> Project utama: `notebook-modeling.ipynb` + `utils/modeling.py`

---

## Daftar Reference Notebooks

| File | Deskripsi |
|------|-----------|
| `reference/kode-tugas-akhir-tanpa-preproses.ipynb` | Baseline: gambar asli tanpa preprocessing |
| `reference/kode-tugas-akhir-preproses-12plus1.ipynb` | 12 short leads + 1 long lead, TF/Keras |
| `reference/kode-tugas-akhir-preproses-6plus6plus1.ipynb` | 6 limb + 6 precordial + 1 long lead, TF/Keras |
| `reference/kode-tugas-akhir-preprocess-terbaik-3-model.ipynb` | Preprocessing terbaik (12+1), 3 model (MobileNetV2, ResNet50, InceptionV3), 3 run |

---

## 1. Dataset Loading dan Transforms

### Reference (TF/Keras)
- Load gambar dengan `cv2.imread()`, resize ke `(224, 224)` menggunakan `cv2.INTER_AREA`.
- Normalisasi sederhana: `/255.0` saja.
- Preprocessing: grayscale → Gaussian blur (5,5) → **Otsu threshold** (`THRESH_BINARY_INV + THRESH_OTSU`) → **morphological opening** (`MORPH_OPEN`) → crop berbasis contour → split lead manual.
- Data split: 80/20 train-test, lalu 20% dari 80% untuk val → final **64/16/20**. Notebook 12+1 dan 6+6+1 menggunakan **60/20/20**.
- **Tidak ada augmentasi** di semua reference.

### Main Project (PyTorch)
- `_build_transform()` di `utils/modeling.py`:
  ```python
  transforms.Grayscale(num_output_channels=3),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ```
- Preprocessing lebih canggih: **adaptive threshold** (Gaussian, block=31) → **MORPH_CLOSE** → connected-component filtering → **text removal step**.
- Data split: **70/15/15** (stratified).

### Perbedaan
| Aspek | Reference | Main Project |
|-------|-----------|--------------|
| Normalisasi | `/255.0` | ImageNet mean/std ✅ lebih tepat untuk pretrained model |
| Thresholding | Global Otsu | Adaptive threshold ✅ lebih robust |
| Morphological op | `MORPH_OPEN` (buang noise) | `MORPH_CLOSE` (sambungkan garis) ✅ lebih cocok untuk ECG |
| Text removal | Tidak ada | Ada ✅ |
| Augmentasi | Tidak ada | Tidak ada (sama) |
| Data split | 60/20/20 | 70/15/15 (test set lebih kecil) |

---

## 2. Arsitektur Model

### Reference
- Backbone: MobileNetV2, ResNet50, InceptionV3 (semua **frozen**).
- Multi-input: setiap branch lead mendapat backbone pretrained tersendiri → `GlobalAveragePooling2D` → `concatenate` → `Dense(256)` → `Dropout(0.5)` → output.
- Strategi: **pretrained features dipertahankan sepenuhnya**, hanya head yang dilatih.

### Main Project
- Backbone: **ResNet-18 saja**.
- Multi-lead: semua lead di-**stack channel-wise** → (39, H, W) → Conv1 diganti untuk menerima 39 channel.
- Seluruh backbone dilatih (tidak ada `requires_grad=False`).

### Perbedaan (SIGNIFIKAN)
| Aspek | Reference | Main Project |
|-------|-----------|--------------|
| Backbone | MobileNetV2/ResNet50/InceptionV3 | ResNet-18 |
| Pretrained weights Conv1 | Dipertahankan (frozen) | **Dibuang (diganti random init)** ⚠️ |
| Fusion strategy | Multi-branch terpisah per lead group | Single model, semua channel digabung |
| Fine-tuning | Frozen (hanya head) | **Full fine-tune ~11M params** ⚠️ |
| Risiko overfitting | Rendah (frozen + dropout 0.5) | **Tinggi** (928 sampel, 11M params) |

---

## 3. Training Loop & Hyperparameter

### Reference
```python
optimizer = Adam(lr=0.0001)
epochs = 50
batch_size = 32
callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
```

### Main Project
```python
optimizer = Adam(lr=1e-4)
num_epochs = 3   # ← ini masalah kritis
batch_size = 16
# Tidak ada early stopping
```

### Perbedaan
| Aspek | Reference | Main Project |
|-------|-----------|--------------|
| Epochs | 50 (dengan early stopping) | **3** ❌ terlalu sedikit |
| Early stopping | Ada (patience=5) | **Tidak ada** ❌ |
| Batch size | 32 | 16 |
| LR scheduler | Tidak ada | Tidak ada (sama) |

---

## 4. Evaluasi & Metrik

### Reference
- Confusion matrix (PNG), `classification_report` per run.
- Best val accuracy & best val loss dilaporkan eksplisit.
- **3 repeated runs** untuk mengukur variance.
- Hasil disimpan sebagai `.txt` dan `.png` lokal.

### Main Project
- `evaluate_and_log()`: accuracy, F1-macro, confusion matrix, classification report.
- Logging ke MLflow (DagsHub) — tidak ada di reference.
- `roc_auc_score` **diimport tapi tidak pernah dipakai**.
- **Tidak ada repeated runs**.

### Perbedaan
| Aspek | Reference | Main Project |
|-------|-----------|--------------|
| AUROC | Tidak ada | Diimport tapi tidak dipakai ⚠️ |
| Repeated runs | 3 kali | Tidak ada |
| Experiment tracking | Lokal (file) | MLflow + DagsHub ✅ lebih lengkap |

---

## 5. Kesalahan yang Ditemukan di Main Project

### ❌ KRITIS

**A. `NUM_EPOCHS = 3` — Model belum converge**  
Notebook `notebook-modeling.ipynb` menset `NUM_EPOCHS = 3`. Reference menggunakan 50 epochs dengan early stopping. 3 epoch hampir pasti tidak cukup untuk convergence. Contoh output: `scheme3_2class Epoch 3: val_loss=2.24` — model diverging.  
**Fix:** Naikkan ke minimal 30–50 dan tambahkan early stopping.

**B. Scheme 2 dan Scheme 3 identik — tidak ada perbedaan**  
`ECGDatasetScheme2` dan `ECGDatasetScheme3` di `utils/modeling.py` menggunakan `ALL_PREFIXES` yang **sama persis** dan logika `__getitem__` yang identik. Di reference, Scheme 3 (6+6+1) memisahkan limb leads dan precordial leads ke branch model berbeda untuk mengeksploitasi anatomical grouping. Di main project keduanya menghasilkan tensor (39, H, W) yang identik — **6 eksperimen yang dijalankan sebenarnya hanya 4 eksperimen unik**.  
**Fix:** Implementasikan ECGDatasetScheme3 yang benar-benar berbeda, atau dokumentasikan bahwa Scheme 2 = Scheme 3 dalam konteks ini.

**C. Tidak ada Early Stopping**  
`train_model()` menyimpan best checkpoint berdasarkan val_acc tapi tetap menjalankan semua epoch. Tanpa early stopping, training berlebih meningkatkan risiko overfitting pada dataset kecil (928 sampel).

### ⚠️ SIGNIFIKAN

**D. Conv1 pretrained weights hilang untuk multi-lead schemes**  
Saat `in_channels != 3`, Conv1 diganti dengan layer baru (random init). Ini menghilangkan pretrained ImageNet features untuk layer paling kritis dari backbone ResNet-18. Reference menghindari ini dengan menggunakan adapter `Conv2D(3, (1,1))` sebelum backbone yang tetap frozen.

**E. `evaluate_and_log` dipanggil tanpa `run_name` di beberapa sel**  
Beberapa sel notebook memanggil `evaluate_and_log()` tanpa parameter `run_name`, sehingga **test metrics tidak ter-log ke MLflow** secara diam-diam. Contoh:
```python
result_s1_2c = evaluate_and_log(
    model_s1_2c, test_s1_2c, DEVICE,
    class_names=CLASS_NAMES_2,
    scheme="scheme1", task="2class"
    # run_name tidak ada → MLflow logging dilewati!
)
```

**F. `roc_auc_score` diimport tapi tidak pernah digunakan**  
`utils/modeling.py` baris 17 mengimport `roc_auc_score` tapi tidak pernah dipanggil. Untuk tugas diagnostik medis, AUROC adalah metrik penting terutama untuk 2-class task.

**G. Nested MLflow run tidak benar-benar nested**  
`train_model()` membuka run MLflow lalu menutupnya (`with` block). Saat `evaluate_and_log()` dipanggil setelahnya, `nested=True` tidak memiliki parent run yang aktif, sehingga test run menjadi **sibling run terpisah**, bukan child. Train metrics dan test metrics tidak terhubung di MLflow UI.

### ℹ️ MINOR

**H. `num_workers=2` di Windows Jupyter dapat menyebabkan error**  
PyTorch multiprocessing di Windows dalam Jupyter memerlukan `if __name__ == '__main__':` guard. Saat ini berjalan tapi berisiko.

**I. Format `date_time` menggunakan titik dua dalam nilai MLflow param**  
`f"{now.day:02d}:{now.month:02d}-{now.hour:02d}:{now.minute:02d}"` — titik dua dalam nilai param dapat menyebabkan masalah di beberapa MLflow backend. Format standar: `YYYYMMDD_HHMM`.

---

## Ringkasan Tabel Perbandingan

| Aspek | Reference (terbaik) | Main Project | Status |
|-------|---------------------|--------------|--------|
| Framework | TF/Keras | PyTorch | Berbeda, tidak salah |
| Preprocessing | Otsu + MORPH_OPEN | Adaptive + MORPH_CLOSE + text removal | ✅ Main lebih canggih |
| Normalisasi | /255.0 | ImageNet mean/std | ✅ Main lebih tepat |
| Model | MobileNetV2/ResNet50/InceptionV3 (frozen) | ResNet-18 (full fine-tune, conv1 diganti) | ⚠️ Deviasi signifikan |
| Fusion multi-lead | Multi-branch terpisah | Single model, channel stack | ⚠️ Berbeda dari reference |
| Epochs | 50 + EarlyStopping(patience=5) | **3, tanpa early stopping** | ❌ Bug kritis |
| Data split | 60/20/20 | 70/15/15 | ℹ️ Minor |
| Batch size | 32 | 16 | ℹ️ Minor |
| Experiment tracking | Tidak ada (lokal) | MLflow + DagsHub | ✅ Main lebih baik |
| Konsistensi MLflow | N/A | run_name hilang di beberapa sel | ❌ Bug |
| Scheme 2 vs Scheme 3 | Berbeda (2 vs 3 branch) | **Identik** | ❌ Bug |
| AUROC | Tidak ada | Diimport tapi tidak dipakai | ⚠️ Metrik hilang |
| Repeated runs | 3 run untuk ukur variance | Tidak ada | ⚠️ Tidak ada |

---

## Rekomendasi Prioritas Perbaikan

1. **[KRITIS]** Naikkan `NUM_EPOCHS` ke 30–50 dan implementasikan early stopping di `train_model()`.
2. **[KRITIS]** Perbaiki atau dokumentasikan perbedaan Scheme 2 vs. Scheme 3 — saat ini identik.
3. **[SIGNIFIKAN]** Tambahkan `run_name` ke semua panggilan `evaluate_and_log()` di notebook.
4. **[SIGNIFIKAN]** Implementasikan `roc_auc_score` di `evaluate_and_log()` atau hapus import-nya.
5. **[SIGNIFIKAN]** Perbaiki struktur nested MLflow run agar test metrics terhubung ke training run.
6. **[MINOR]** Pertimbangkan freeze sebagian backbone (setidaknya layer awal) untuk mengurangi overfitting.
