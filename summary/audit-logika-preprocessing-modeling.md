# Audit Logika Preprocessing, Modelling, Utils, dan Reference

Tanggal audit: 2026-04-19

## Ruang Lingkup

File yang diperiksa:

- `notebook-preprocessing.ipynb`
- `notebook-modeling.ipynb`
- `utils/preprocessing.py`
- `utils/modeling.py`
- `utils/eda.py`
- `utils/config.py`
- `reference/kode-tugas-akhir-preproses-12plus1.ipynb`
- `reference/kode-tugas-akhir-preproses-6plus6plus1.ipynb`
- `reference/kode-tugas-akhir-preprocess-terbaik-3-model.ipynb`
- `reference/kode-tugas-akhir-tanpa-preproses.ipynb`

Catatan: audit ini bersifat baca-saja terhadap notebook/kode utama. Perubahan yang dibuat hanya penambahan laporan ini di folder `summary/`.

## Ringkasan Eksekutif

Ada beberapa hal yang sudah benar:

- Alur preprocessing notebook sudah runtut: download/rename data, EDA, crop area ECG, ekstraksi sinyal, pembuatan/pemakaian mask teks, lalu crop 12 short lead dan 1 long lead.
- Struktur output `data/preprocessed/cropped_leads` sudah konsisten: setiap sampel yang dicek memiliki 13 file lead.
- Pembagian data di modelling sudah memakai stratified split sehingga proporsi kelas lebih terjaga.
- Label 2-class sudah masuk akal: `ECG_Normal` sebagai sehat, sedangkan `ECG_Abnormal`, `ECG_MI`, dan `ECG_HistoryMI` sebagai sakit.

Namun ada kesalahan logika penting:

1. **Skema 2 dan Skema 3 pada `utils/modeling.py` saat ini identik secara input model.**
   Skema 2 diklaim sebagai `12 short lead + long lead`, sedangkan Skema 3 diklaim sebagai `6 limb + 6 precordial + long lead`. Tetapi keduanya sama-sama memuat 13 lead, sama-sama menumpuk semua lead ke tensor `(39, H, W)`, dan sama-sama memakai `in_channels=39`. Artinya perbandingan performa Skema 2 vs Skema 3 tidak valid sebagai perbandingan dua preprocessing berbeda.

2. **Implementasi Skema 3 tidak mengikuti reference `6+6+1`.**
   Di reference, `6+6+1` menghasilkan tiga input terpisah: `X_limb`, `X_precordial`, dan `X_long`. Pada kode saat ini, tiga kelompok tersebut digabung menjadi satu tensor channel-stacked, sehingga struktur multi-input reference hilang.

3. **Skema 2 juga tidak mengikuti arsitektur multi-input reference `12+1`.**
   Di reference `12+1`, 12 lead dan long lead diperlakukan sebagai dua input/cabang model terpisah. Pada kode saat ini, 12 short lead dan long lead digabung menjadi satu tensor 39 channel. Ini bukan selalu salah secara PyTorch, tetapi bukan replika metodologi reference.

4. **Ada risiko data leakage antar eksperimen karena split dilakukan ulang per dataset/skema.**
   Walaupun seed sama, split dilakukan terhadap dataset object yang berbeda. Selama urutan record benar-benar sama, hasilnya mungkin sama, tetapi ini tidak dijamin oleh desain. Untuk eksperimen yang dibandingkan antar skema, lebih kuat jika split berbasis `sample_id` dibuat sekali lalu dipakai ulang untuk semua skema.

5. **Notebook modelling tampak memiliki sel eksperimen yang berulang dan evaluasi yang tidak konsisten logging-nya.**
   Ada pola training/evaluasi untuk skema yang muncul lebih dari sekali, dan beberapa `evaluate_and_log` dipanggil tanpa `run_name` atau `output_dir`. Dampaknya bukan selalu salah model, tetapi hasil eksperimen bisa tercampur, tertimpa secara variabel, atau tidak lengkap di MLflow/local output.

## Detail Temuan

### 1. Skema 2 dan Skema 3 Identik

Lokasi terkait:

- `utils/modeling.py`, konstanta lead: `LIMB_LEAD_PREFIXES`, `PRECORDIAL_LEAD_PREFIXES`, `LONG_LEAD_PREFIX`
- `utils/modeling.py`, `_get_image_files_scheme2`
- `utils/modeling.py`, `_get_image_files_scheme3`
- `utils/modeling.py`, `ECGDatasetScheme2`
- `utils/modeling.py`, `ECGDatasetScheme3`

Masalah:

```python
def _get_image_files_scheme3(data_root: str):
    return _get_image_files_scheme2(data_root)
```

Selain itu:

```python
class ECGDatasetScheme2(Dataset):
    ALL_PREFIXES = LIMB_LEAD_PREFIXES + PRECORDIAL_LEAD_PREFIXES + [LONG_LEAD_PREFIX]
```

dan:

```python
class ECGDatasetScheme3(Dataset):
    ALL_PREFIXES = LIMB_LEAD_PREFIXES + PRECORDIAL_LEAD_PREFIXES + [LONG_LEAD_PREFIX]
```

Kedua class memakai daftar lead yang sama, transform yang sama, urutan loading yang sama, dan output tensor yang sama:

```python
img_tensor = torch.cat(tensors, dim=0)  # shape: (39, H, W)
```

Dampak:

- Skema 2 dan Skema 3 bukan eksperimen yang berbeda secara data input.
- Jika hasil akurasi/F1 berbeda, perbedaannya kemungkinan berasal dari random training, bukan dari perbedaan preprocessing.
- Klaim "Skema 3 = 6 tungkai + 6 precordial + long lead" tidak benar secara implementasi saat ini, karena akhirnya tetap menjadi 13 lead channel-stacked seperti Skema 2.

Rekomendasi:

- Pilih salah satu arah metodologi:
  - Jika ingin mengikuti reference, buat Skema 2 sebagai model 2-input: blok 12 lead dan long lead; buat Skema 3 sebagai model 3-input: limb, precordial, long.
  - Jika tetap ingin single-input PyTorch, ubah deskripsi eksperimen agar jujur: Skema 2 dan Skema 3 saat ini hanya berbeda nama/urutan konseptual, bukan berbeda input aktual. Tetapi ini membuat perbandingan Skema 2 vs Skema 3 tidak bermakna.

### 2. Perbandingan Dengan Reference `12+1`

Reference:

- `reference/kode-tugas-akhir-preproses-12plus1.ipynb`

Logika reference:

- Preprocessing menghasilkan dua set data:
  - `X_12_lead_processed.npy`
  - `X_long_lead_processed.npy`
- Model menerima dua input:
  - `input_short_lead`
  - `input_long_lead`
- Masing-masing cabang diadaptasi ke model transfer learning, lalu fitur digabung.

Implementasi saat ini:

- File individual 12 short lead dan 1 long lead disimpan di folder per sampel.
- Dataset memuat 13 gambar lead dan melakukan `torch.cat` menjadi 39 channel.
- Model hanya satu input ResNet-18 dengan conv pertama diganti menjadi `in_channels=39`.

Kesimpulan:

- Tidak ada kesalahan fatal jika memang ingin pendekatan single-input channel-stacking.
- Tetapi secara metodologi, ini **bukan implementasi yang setara** dengan reference `12+1`.
- Long lead kehilangan status sebagai branch terpisah; model memperlakukannya sebagai channel tambahan bersama lead lain.

### 3. Perbandingan Dengan Reference `6+6+1`

Reference:

- `reference/kode-tugas-akhir-preproses-6plus6plus1.ipynb`

Logika reference:

- Preprocessing menghasilkan tiga set data:
  - `X_limb_processed.npy`
  - `X_precordial_processed.npy`
  - `X_long_lead_processed.npy`
- Model menerima list input:
  - limb lead
  - precordial lead
  - long lead
- Tiap kelompok dapat memiliki cabang feature extractor sendiri.

Implementasi saat ini:

- `LIMB_LEAD_PREFIXES` dan `PRECORDIAL_LEAD_PREFIXES` memang didefinisikan.
- Tetapi keduanya langsung digabung dengan long lead menjadi satu list `ALL_PREFIXES`.
- Output akhir tetap satu tensor `(39, H, W)`.

Kesimpulan:

- Pengelompokan limb/precordial hanya dipakai untuk urutan channel, bukan sebagai desain input/model.
- Ini menyimpang dari reference `6+6+1`.

### 4. Preprocessing: Secara Umum Runtut, Tetapi Ada Catatan

Lokasi:

- `notebook-preprocessing.ipynb`
- `utils/preprocessing.py`
- `utils/config.py`

Alur saat ini:

1. Download dan rename folder.
2. Crop area ECG dari raw image.
3. Ekstrak sinyal ECG dengan adaptive threshold dan connected components.
4. Buat text mask.
5. Hapus teks dari sinyal ECG.
6. Crop short lead dan long lead.

Catatan:

- Notebook membuat mask melalui `create_text_mask(..., data/preprocessed/text_mask/text_mask.png)`, tetapi tahap preview dan delete text memakai `utils/text_mask_new.png`.
- Ini bukan pasti salah jika `utils/text_mask_new.png` memang mask final yang sengaja dipakai. Namun dari sisi reproducibility, notebook menjadi kurang self-contained karena output mask yang baru dibuat tidak dipakai langsung.
- `create_text_mask` membuat mask dari satu gambar `ECG_Normal` pertama. Kalau posisi teks antar kelas/sampel tidak selalu identik, mask ini bisa menghapus bagian sinyal atau gagal menghapus teks pada sampel lain.
- Fungsi `_clean_and_remove_text` menjalankan threshold ulang pada gambar yang sudah berada di folder `extracted_ecg_signal`. Ini bisa mengubah hasil ekstraksi awal, bukan sekadar menghapus teks dari hasil yang sudah ada.

Rekomendasi:

- Tentukan satu sumber mask yang eksplisit. Jika `utils/text_mask_new.png` adalah mask manual/final, tulis alasannya di notebook.
- Simpan visual audit beberapa sampel per kelas setelah hapus teks dan setelah crop lead.
- Validasi bahwa mask tidak menghapus bagian sinyal ECG penting.

### 5. Crop Lead dan Nama Lead

Lokasi:

- `utils/config.py`
- `utils/preprocessing.py`
- `data/preprocessed/cropped_leads`

Hasil inspeksi:

- Setiap sample folder yang dicek berisi 13 file `.png`.
- Nama file mengikuti pola seperti:
  - `01_lead_lead_I.png`
  - `02_lead_a_VR.png`
  - `03_lead_v_1.png`
  - ...
  - `13_long_lead.png`

Catatan:

- Nama `01_lead_lead_I` terlihat redundant karena `lead_filename = f"{lead_idx:02d}_lead_{lead_name}.png"` dan `lead_name` untuk index 1 adalah `lead_I`.
- Ini bukan salah logika selama prefix di `utils/modeling.py` sama. Saat ini memang sudah sama.
- Reference memakai nama lebih standar: `lead_I`, `lead_II`, `lead_III`, `lead_aVR`, `lead_aVL`, `lead_aVF`, `lead_V1`, dst. Kode saat ini memakai variasi `a_VR`, `v_1`, `v5`. Ini tidak salah secara teknis, tetapi membuat integrasi dengan reference lebih sulit.

### 6. Split Data dan Validitas Perbandingan

Lokasi:

- `utils/modeling.py`, `build_dataloaders`
- `notebook-modeling.ipynb`

Yang sudah benar:

- Split memakai `train_test_split(..., stratify=labels, random_state=seed)`.
- Proporsi default 70/15/15 jika `val_split=0.15` dan `test_split=0.15`.

Masalah konseptual:

- Split dibuat dari dataset object masing-masing skema.
- Untuk membandingkan Skema 1, 2, dan 3 secara fair, sample yang masuk train/val/test sebaiknya identik berdasarkan `sample_id`.
- Saat ini urutan record bergantung pada cara file/folder dibaca oleh masing-masing `_get_image_files_scheme*`.

Rekomendasi:

- Buat manifest sample berisi `sample_id`, `class`, path clean image, path 13 lead.
- Buat `train_ids`, `val_ids`, `test_ids` sekali saja.
- Dataset tiap skema menerima daftar ID yang sama.

### 7. Model Transfer Learning Dengan 39 Channel

Lokasi:

- `utils/modeling.py`, `build_model`

Masalah:

```python
if in_channels != 3:
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
```

Ketika `in_channels=39`, layer conv pertama diganti baru. Akibatnya pretrained weight ImageNet pada conv pertama tidak dipakai untuk Skema 2/3. Backbone lain tetap pretrained, tetapi adaptasi awalnya random.

Ini bukan bug fatal, tetapi perlu dicatat karena:

- Perbandingan Skema 1 vs Skema 2/3 tidak sepenuhnya apple-to-apple.
- Skema 1 memakai conv pertama pretrained 3-channel.
- Skema 2/3 memakai conv pertama baru 39-channel.

Rekomendasi:

- Jika tetap memakai 39 channel, pertimbangkan inisialisasi conv1 dari rerata weight pretrained RGB yang direplikasi ke 39 channel.
- Alternatif lebih dekat reference: pakai branch 3-channel/1-channel per kelompok lead, lalu gabungkan embedding.

### 8. Evaluasi dan Logging

Lokasi:

- `notebook-modeling.ipynb`
- `utils/modeling.py`, `train_model`, `evaluate_and_log`

Catatan:

- `train_model` menyimpan best model berdasarkan validation accuracy.
- `evaluate_and_log` menghitung test accuracy dan macro F1.
- Beberapa cell notebook memanggil `evaluate_and_log` tanpa `run_name` dan `output_dir`, sehingga tidak semua evaluasi tercatat ke MLflow atau tersimpan confusion matrix test-nya.
- Notebook memiliki pola training/evaluasi berulang untuk skema yang sama. Ini berisiko membuat variabel seperti `model_s1_4c`, `history_s1_4c`, atau `run_output_dir` tertimpa oleh run berikutnya.

Rekomendasi:

- Rapikan notebook modelling menjadi satu cell training dan satu cell evaluasi per kombinasi eksperimen.
- Gunakan `run_name` unik untuk setiap eksperimen.
- Hindari menjalankan ulang eksperimen dengan nama variabel yang sama kecuali memang disengaja.

### 9. Jumlah Data

Hasil inspeksi folder lokal:

| Folder | Raw | Clean ECG | Cropped Leads |
|---|---:|---:|---:|
| `ECG_Abnormal` | 233 | 233 | 233 sample folders |
| `ECG_HistoryMI` | 172 | 172 | 172 sample folders |
| `ECG_MI` | 239 | 239 | 239 sample folders |
| `ECG_Normal` | 284 | 284 | 284 sample folders |

Catatan:

- `Config.folder_mapping` menyebut sumber MI sebagai `240x12=2880`, tetapi folder lokal berisi 239 file raw setelah rename/extract.
- Ini perlu dicek apakah memang ada 1 file hilang/gagal ekstrak atau dataset asli yang tersedia berjumlah 239 gambar untuk kelas MI.

## Kesimpulan

Ada kesalahan logika utama pada bagian modelling: **Skema 2 dan Skema 3 saat ini tidak benar-benar berbeda secara input model**, sehingga hasil perbandingan antar skema tersebut tidak bisa dianggap membandingkan `12+1` vs `6+6+1`.

Pipeline preprocessing secara umum masuk akal dan output folder terlihat konsisten, tetapi reproducibility mask teks perlu diperjelas. Dibandingkan reference, implementasi saat ini lebih merupakan adaptasi PyTorch single-input channel-stacking, bukan replikasi multi-input seperti reference.

## Prioritas Perbaikan

1. Pisahkan desain Skema 2 dan Skema 3 agar benar-benar berbeda.
2. Buat split berbasis `sample_id` yang dipakai ulang semua skema.
3. Rapikan notebook modelling agar tidak ada run ganda/tumpang tindih variabel.
4. Jelaskan sumber `utils/text_mask_new.png` atau gunakan mask yang dibuat notebook secara konsisten.
5. Cek penyebab jumlah `ECG_MI` lokal hanya 239, bukan 240 seperti deskripsi folder asli.

