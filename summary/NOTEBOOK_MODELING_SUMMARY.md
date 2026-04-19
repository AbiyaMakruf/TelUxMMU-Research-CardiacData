# ECG Classification – Notebook Modeling Summary

## Overview

**notebook-modeling.ipynb** adalah notebook PyTorch yang melatih model ResNet-18 untuk klasifikasi ECG dengan **3 skema preprocessing berbeda** dan **2 jenis task klasifikasi**. Hasil dari semua eksperimen di-track menggunakan **MLflow via DagsHub** dan disimpan baik secara lokal maupun remote.

---

## 1. Struktur Eksperimen

### Dimensi Eksperimen
| Dimensi | Nilai | Total Kombinasi |
|---------|-------|-----------------|
| **Skema Preprocessing** | 3 (Scheme 1, 2, 3) | **6 kombinasi** |
| **Task Klasifikasi** | 2 (4-class, 2-class) | |

### Tabel Ringkas

| Skema | Input Representation | In-Channels | Sumber Data |
|-------|----------------------|-------------|------------|
| **Scheme 1** | 1 gambar clean ECG (full image) | **3** | `data/preprocessed/clean_ecg_signal/` |
| **Scheme 2** | 12 short leads + 1 long lead (13 total) | **39** (13×3) | `data/preprocessed/cropped_leads/` |
| **Scheme 3** | 6 limb leads + 6 precordial leads + 1 long lead (13 total) | **39** (13×3) | `data/preprocessed/cropped_leads/` |

### Task Klasifikasi

1. **4-Class (Multi-class):**
   - Normal (0)
   - Abnormal (1)
   - MI – Myocardial Infarction (2)
   - History MI (3)

2. **2-Class (Binary):**
   - Sehat (Healthy) (0) – includes Normal
   - Sakit (Diseased) (1) – includes Abnormal, MI, History MI

---

## 2. Detail Masing-Masing Skema

### **Scheme 1 – Gambar Clean (Clean ECG Signal)**

#### Karakteristik Input
- **Data Source:** `data/preprocessed/clean_ecg_signal/` folder
- **Deskripsi:** Merupakan seluruh gambar ECG yang telah dibersihkan (preprocessing background noise)
- **Format Input:** Satu gambar ECG per sampel
- **Ukuran Channel:** **3 channels** (RGB – converted from grayscale)
- **Dimensi Tensor:** `(1, 3, 224, 224)` per sampel dalam batch

#### Dataset Class
```python
class ECGDatasetScheme1(Dataset):
    """Load satu clean ECG image per sample dari clean_ecg_signal/"""
```

#### Model Architecture
- **Architecture:** ResNet-18 dengan input 3-channel
- **Pretrained:** Menggunakan ImageNet pretrained weights
- **Input Layer:** Conv2d(3, 64, kernel_size=7, ...) – standard

#### Kelebihan & Kelemahan
✅ **Keuntungan:**
- Input sederhana (1 gambar per sampel)
- Menggunakan pretrained ImageNet weights secara optimal (3-channel match)
- Waktu training cepat (fewer parameters)
- Intuitive – menggunakan full visual context

❌ **Keterbatasan:**
- Tidak memisahkan individual leads ECG
- Kehilangan struktur elektrokardiografi yang terorganisir
- Bergantung pada preprocessing clean yang akurat

#### Eksperimen dalam Notebook
```python
# Scheme1 – 4-class
model_s1_4c = build_model(num_classes=4, in_channels=3)
train_model(..., run_name="scheme1_4class", ...)

# Scheme1 – 2-class
model_s1_2c = build_model(num_classes=2, in_channels=3)
train_model(..., run_name="scheme1_2class", ...)
```

---

### **Scheme 2 – 12 Short Leads + 1 Long Lead**

#### Karakteristik Input
- **Data Source:** `data/preprocessed/cropped_leads/` folder
- **Deskripsi:** Masing-masing dari 12 ECG leads yang dipangkas (cropped), plus 1 long lead tambahan
- **Jumlah Leads:** 13 total
  - 6 limb leads: lead I, lead II, lead III, aVR, aVL, aVF
  - 6 precordial leads: V1, V2, V3, V4, V5, V6
  - 1 long lead: rhythmic strip (full duration)
- **Format Input:** Dictionary {lead_prefix → image_path}
- **Ukuran Channel:** **39 channels** (13 leads × 3 channels RGB per lead)
- **Dimensi Tensor:** `(39, 224, 224)` per sampel

#### Dataset Class
```python
class ECGDatasetScheme2(Dataset):
    """Load 13 lead images per sample, stack channel-wise ke (39, H, W)"""
    ALL_PREFIXES = [
        "01_lead_lead_I", "05_lead_lead_II", "09_lead_lead_III",  # Limb
        "02_lead_a_VR", "06_lead_a_VL", "10_lead_a_VF",           # Augmented
        "03_lead_v_1", "07_lead_v_2", "11_lead_v_3",              # Precordial
        "04_lead_v_4", "08_lead_v5", "12_lead_v_6",               # Precordial
        "13_long_lead"                                             # Long Lead
    ]
```

#### Model Architecture
- **Architecture:** ResNet-18 dengan input **39-channel**
- **Pretrained:** Menggunakan ImageNet weights, tapi conv1 layer diganti:
  ```python
  model.conv1 = nn.Conv2d(39, 64, kernel_size=7, stride=2, padding=3, bias=False)
  ```
  Alasan: InputNet hanya support 3-channel, jadi layer pertama di-adapt untuk 39-channel

#### Lead Grouping Logic
- **Limb Leads (6):** Lead I, II, III (frontal plane standard) + augmented (aVR, aVL, aVF)
- **Precordial Leads (6):** V1-V6 (transverse/horizontal plane) – fundamental untuk MI detection
- **Long Lead (1):** Full-duration rhythmic strip untuk pattern analysis

#### Kelebihan & Kelemahan
✅ **Keuntungan:**
- Mempertahankan struktur ECG yang terorganisir (individual leads)
- All 12-lead standard ECG information + long lead
- Model dapat belajar dari spatial relationships antar leads
- Lebih dekat dengan interpretasi klinis

❌ **Keterbatasan:**
- Input channels jauh dari pretrained model (39 ≠ 3), maka conv1 perlu diganti
- Computational overhead meningkat (3.25× lebih banyak channels)
- Complexity dalam data loading (perlu navigate folder structure leads)

#### Eksperimen dalam Notebook
```python
# Scheme2 – 4-class
model_s2_4c = build_model(num_classes=4, in_channels=39)
train_model(..., run_name="scheme2_4class", ...)

# Scheme2 – 2-class
model_s2_2c = build_model(num_classes=2, in_channels=39)
train_model(..., run_name="scheme2_2class", ...)
```

---

### **Scheme 3 – 6 Limb + 6 Precordial + 1 Long Lead**

#### Karakteristik Input
- **Data Source:** `data/preprocessed/cropped_leads/` (sama dengan Scheme 2)
- **Deskripsi:** Subset dari 12 leads yang diatur secara anatomis:
  - 6 limb leads (lead I, II, III, aVR, aVL, aVF)
  - 6 precordial leads (V1-V6)
  - 1 long lead
- **Jumlah Leads:** 13 total (sama dengan Scheme 2 – **SELURUH 12-lead ECG + long lead**)
- **Format Input:** Dictionary {lead_prefix → image_path}, hanya 13 lead yang dipilih
- **Ukuran Channel:** **39 channels** (13 leads × 3 channels)
- **Dimensi Tensor:** `(39, 224, 224)` per sampel

#### Dataset Class
```python
class ECGDatasetScheme3(Dataset):
    """Load 13 lead images dalam anatomical order: Limb → Precordial → Long"""
    ALL_PREFIXES = [
        # Limb leads first (anatomical grouping)
        "01_lead_lead_I", "05_lead_lead_II", "09_lead_lead_III",
        "02_lead_a_VR", "06_lead_a_VL", "10_lead_a_VF",
        # Precordial leads
        "03_lead_v_1", "07_lead_v_2", "11_lead_v_3",
        "04_lead_v_4", "08_lead_v5", "12_lead_v_6",
        # Long lead
        "13_long_lead"
    ]
```

#### Model Architecture
- **Architecture:** ResNet-18 dengan input **39-channel**
- **Pretrained:** Sama dengan Scheme 2 – conv1 layer diganti untuk 39-channel
- **Perbedaan dari Scheme 2:** HANYA dalam urutan channel stacking (anatomical order)

#### Lead Grouping Logic & Clinical Significance
| Grup | Leads | Fungsi Klinis | Deteksi Spesialisasi |
|------|-------|---------------|----------------------|
| **Limb** | I, II, III, aVR, aVL, aVF | Frontal plane view | Lateral MI, bundle branch blocks |
| **Precordial** | V1-V6 | Horizontal/transverse plane | Anterior MI, ventricular issues |
| **Long Lead** | Single long lead | Arrhythmia analysis | Continuous rhythm observation |

#### Perbedaan Scheme 2 vs Scheme 3
| Aspek | Scheme 2 | Scheme 3 |
|------|----------|----------|
| **Total Leads** | 13 (all same) | 13 (all same) |
| **Data Source** | cropped_leads/ | cropped_leads/ |
| **Channels** | 39 | 39 |
| **Channel Order** | Generic order (01→13) | Anatomical order (Limb→Precordial→Long) |
| **Intent** | All 12-lead info + long | All 12-lead info + long, **organized anatomically** |

#### Kelebihan & Kelemahan
✅ **Keuntungan:**
- Channel ordering reflects anatomical/clinical grouping
- Model dapat belajar group-level patterns (limb group, precordial group)
- Memudahkan interpretasi – related leads dijadwalkan berdekatan di channel dimension

❌ **Keterbatasan:**
- Tidak ada bukti kuat anatomical ordering lebih baik daripada generic ordering
- Minimal additional benefit dibanding Scheme 2 (perlu empirical validation)
- Sama-sama computational overhead 39-channel

#### Eksperimen dalam Notebook
```python
# Scheme3 – 4-class
model_s3_4c = build_model(num_classes=4, in_channels=39)
train_model(..., run_name="scheme3_4class", ...)

# Scheme3 – 2-class
model_s3_2c = build_model(num_classes=2, in_channels=39)
train_model(..., run_name="scheme3_2class", ...)
```

---

## 3. Pipeline Training & Evaluation

### 3.1 Data Preparation

#### Step 1: Load Image Records
```python
records_s1 = _get_image_files_scheme1(Config.data_root)
records_s2 = _get_image_files_scheme2(Config.data_root)
records_s3 = _get_image_files_scheme3(Config.data_root)  # Same as records_s2
```

#### Step 2: Create Dataset
```python
dataset = ECGDatasetScheme1(records_s1, task="4class", image_size=224)
# atau
dataset = ECGDatasetScheme2(records_s2, task="4class", image_size=224)
# atau
dataset = ECGDatasetScheme3(records_s3, task="4class", image_size=224)
```

#### Step 3: Build DataLoaders (Stratified Split)
```python
train_loader, val_loader, test_loader = build_dataloaders(
    dataset, 
    seed=42,  # For reproducibility
    batch_size=16,
    val_split=0.15,  # 15% validation
    test_split=0.15  # 15% test, 70% train
)
```

**Stratification Strategy:**
- Splits maintain class distribution across train/val/test
- Pin memory enabled (`pin_memory=True`) untuk GPU optimization
- 2 worker processes (`num_workers=2`) untuk parallel data loading

### 3.2 Model Building
```python
model = build_model(
    num_classes=4,        # atau 2
    in_channels=3,        # Scheme 1
    # in_channels=39,     # Scheme 2/3
    pretrained=True       # ImageNet weights
).to(DEVICE)
```

### 3.3 Training Loop
```python
model, history = train_model(
    model, train_loader, val_loader,
    num_epochs=20,
    learning_rate=1e-4,
    device=DEVICE,
    run_name="scheme1_4class",
    scheme="scheme1",
    task="4class",
    model_name="ResNet-18",
    class_names=CLASS_NAMES_4,
    output_dir="outputs/experiments"
)
```

**Training Details:**
- **Optimizer:** Adam (lr=1e-4)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 20
- **Best Model Selection:** Based on validation accuracy
- **Early Stopping:** Implicit (best model saved when val_acc improves)

**MLflow Logging:**
- Hyperparameters: scheme, task, num_epochs, learning_rate, batch_size, gpu_name, date_time, training_time
- Per-Epoch Metrics: train_loss, train_acc, val_loss, val_acc
- Artifacts:
  - `best_model.pth` – model weights (local + remote)
  - `training_history.png` – loss/accuracy curves
  - `confusion_matrix_val.png` – validation confusion matrix

### 3.4 Evaluation & Testing
```python
result = evaluate_and_log(
    model, test_loader, DEVICE,
    class_names=CLASS_NAMES_4,
    scheme="scheme1",
    task="4class",
    run_name="scheme1_4class",
    output_dir="outputs/experiments"
)
```

**Evaluation Metrics:**
- Test Accuracy
- Test F1 Score (Macro)
- Per-class Precision, Recall, F1
- Confusion Matrix (saved as PNG artifact)

**Artifacts:**
- `confusion_matrix_test.png` – test confusion matrix (local + remote)

---

## 4. Output Structure

### Local Artifact Storage
```
outputs/experiments/
├── scheme1_4class/
│   ├── best_model.pth              # Model weights
│   ├── training_history.png        # Loss/Accuracy curves
│   ├── confusion_matrix_val.png    # Validation confusion matrix
│   └── confusion_matrix_test.png   # Test confusion matrix
├── scheme1_2class/
│   └── ...
├── scheme2_4class/
│   └── ...
└── ... (6 folders total, one per experiment)
```

### Remote Artifact Storage (DagsHub MLflow)
- All artifacts are also logged to remote DagsHub repository
- URL: `https://dagshub.com/abiyamf/TelUxMMU-Research`
- Tracking enables experiment comparison and reproducibility

---

## 5. Hyperparameters (Fixed Across All Experiments)

```python
NUM_EPOCHS    = 20          # Training epochs per experiment
BATCH_SIZE    = 16          # Batch size for all loaders
LEARNING_RATE = 1e-4        # Adam optimizer learning rate
IMAGE_SIZE    = 224         # Resized image/lead size (224×224)
VAL_SPLIT     = 0.15        # Validation fraction
TEST_SPLIT    = 0.15        # Test fraction (70% train)
DEVICE        = cuda        # GPU: NVIDIA GeForce RTX 5070 Ti (17.1 GB)
```

---

## 6. Visualization & Analysis Section (Section 6)

### 6.1 Training History Comparison
Plots training/validation loss and accuracy untuk semua 3 skema dalam 4-class task.

### 6.2 Training History Comparison (2-Class)
Plots untuk 2-class task.

### 6.3 Confusion Matrix Visualization (4-Class)
Side-by-side confusion matrices untuk semua 3 skema, 4-class task.

### 6.4 Confusion Matrix Visualization (2-Class)
Confusion matrices untuk 2-class task.

### 6.5 Scheme Comparison (Accuracy & F1)
Bar chart membandingkan:
- Test Accuracy untuk semua 6 kombinasi eksperimen
- Macro F1 Score untuk semua kombinasi

**Interpretasi:**
- Menunjukkan skema mana yang paling efektif
- Menunjukkan apakah 4-class atau 2-class lebih baik
- Membantu identify best-performing combination

### 6.6 Sample Predictions
Visualisasi prediksi model pada 8 sampel acak dari test set:
- Gambar ECG/lead(s)
- True label
- Predicted label
- Confidence score

**Note:** User perlu customize variabel `best_model`, `best_dataset`, `best_classes` sesuai skema terbaik dari grafik Section 6.5.

---

## 7. Key Implementation Details

### GPU Optimization
- **pin_memory=True** – Data tensors allocated in pinned RAM untuk fast GPU transfer
- **num_workers=2** – Parallel data loading dengan 2 worker processes
- **Result:** Asynchronous data loading prevents GPU idle time

### Model Adaptation for Multi-Channel Input
```python
if in_channels != 3:
    model.conv1 = nn.Conv2d(
        in_channels, 64, 
        kernel_size=7, stride=2, padding=3, bias=False
    )
```
- Scheme 1: Conv1 unchanged (3 channels match pretrained)
- Scheme 2/3: Conv1 replaced (39 channels → adapt to stacked leads)

### Stratified Train/Val/Test Splitting
- Maintains class distribution across splits
- Prevents class imbalance bias
- Important untuk imbalanced ECG dataset

### Training Time Logging
- Actual training duration captured di-log sebagai parameter
- Format: HH:MM:SS
- Useful untuk comparison eksperimen

---

## 8. Interpretasi & Next Steps

### Untuk Memahami Hasil:
1. **Lihat Section 6.5** untuk identifikasi skema terbaik
2. **Analisis confusion matrices** untuk understand misclassifications
3. **Compare training curves** untuk assess overfitting/underfitting
4. **Update best_model reference** di Section 6.6 untuk visualisasi sampel terbaik

### Kemungkinan Research Questions:
- **Apakah multi-lead (Scheme 2/3) lebih baik dari single image (Scheme 1)?**
  - Multi-lead preserves clinical structure tetapi adds complexity
- **Apakah anatomical ordering (Scheme 3) lebih baik dari generic ordering (Scheme 2)?**
  - Memerlukan empirical validation; minimal difference expected
- **Binary (2-class) vs Multi-class (4-class) – mana yang lebih mudah/berguna?**
  - 2-class likely simpler (Healthy vs Diseased)
  - 4-class lebih informatif untuk clinical diagnosis
- **GPU optimization effectiveness?**
  - pin_memory + num_workers = faster training dengan minimal staleness
  - Visible di training time parameter

---

## 9. Technical Stack

| Komponen | Technology |
|----------|-----------|
| **Framework** | PyTorch |
| **Model Architecture** | ResNet-18 (ImageNet pretrained) |
| **Dataset Handling** | torch.utils.data.Dataset + DataLoader |
| **Optimization** | Adam optimizer |
| **Loss Function** | CrossEntropyLoss |
| **Metrics** | Accuracy, F1 Score (macro), Precision, Recall |
| **Experiment Tracking** | MLflow + DagsHub |
| **Visualization** | Matplotlib (confusion matrix, training history, sample predictions) |
| **Computing** | NVIDIA GPU (RTX 5070 Ti – 17.1 GB VRAM) |

---

## 10. Summary Table: 6 Experiments

| Eksperimen | Input | Channels | Model Config | Task | Run Name |
|-----------|-------|----------|--------------|------|----------|
| 1 | Clean image | 3 | ResNet-18 (std) | 4-class | `scheme1_4class` |
| 2 | Clean image | 3 | ResNet-18 (std) | 2-class | `scheme1_2class` |
| 3 | 13 leads (generic order) | 39 | ResNet-18 (39-ch) | 4-class | `scheme2_4class` |
| 4 | 13 leads (generic order) | 39 | ResNet-18 (39-ch) | 2-class | `scheme2_2class` |
| 5 | 13 leads (anatomical order) | 39 | ResNet-18 (39-ch) | 4-class | `scheme3_4class` |
| 6 | 13 leads (anatomical order) | 39 | ResNet-18 (39-ch) | 2-class | `scheme3_2class` |

---

**Document Generated:** 2026-04-19  
**Author:** Abiya Makruf  
**Status:** Active Experiment Tracking via MLflow + DagsHub
