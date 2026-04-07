# Review Preprocessing `ecg-miner`

Dokumen ini merangkum cara kerja pipeline pada folder `ecg-miner`, lalu membandingkannya dengan pipeline preprocessing yang kamu buat sebelumnya.

Sumber kode utama yang ditinjau:

- `ecg-miner/src/digitization/Preprocessor.py`
- `ecg-miner/src/digitization/SignalExtractor.py`
- `ecg-miner/src/digitization/Postprocessor.py`
- `ecg-miner/src/digitization/Digitizer.py`
- `ecg-miner/validation/render.py`
- `ecg-miner/src/app/model/Model.py`

## 1. Bagaimana cara kerja preprocessing-nya?

Penting: di `ecg-miner`, "preprocessing" bukan tujuan akhir. Preprocessing hanyalah tahap awal dari pipeline digitization penuh. Setelah preprocessing, tool ini lanjut ke ekstraksi sinyal, kalibrasi, dan konversi menjadi data numerik.

### Urutan pipeline lengkap

1. Input image dibaca sebagai objek `Image`.
   - Bisa dari file gambar biasa atau PDF.
   - Implementasi ada di `utils/graphics/Image.py`.

2. ECGMiner menjalankan `Preprocessor.preprocess()`.
   - Lihat `ecg-miner/src/digitization/Preprocessor.py:25-40`.
   - Tahap ini mengembalikan dua hal:
     - citra ECG yang sudah di-crop dan dibinarisasi
     - rectangle area crop terhadap gambar asli

3. Tahap `__img_partitioning`: mencari area utama ECG di dalam halaman.
   - Lihat `Preprocessor.py:42-91`.
   - Langkahnya:
     - konversi ke BGR
     - deteksi edge dengan Canny (`cv.Canny`)
     - cari contour eksternal (`cv.findContours`)
     - aproksimasi contour menjadi polygon
     - ubah tiap polygon menjadi bounding rectangle
     - pilih rectangle terbesar sebagai area grid ECG
   - Artinya, pendekatannya bukan split manual per lead, tetapi mendeteksi blok ECG terbesar otomatis.

4. Gambar di-crop ke rectangle hasil deteksi.
   - `Preprocessor.py:37-39`

5. Tahap `__gridline_removal`: menghapus grid/background dan membinarisasi.
   - Lihat `Preprocessor.py:138-166`.
   - Urutannya:
     - citra diubah dari BGR ke HSV
     - dibuat mask HSV dengan rentang:
       - `lower = [0, 0, 168]`
       - `upper = [255, 255, 255]`
     - mask ini mempertahankan piksel yang cukup terang
     - hasil mask dipakai sebagai citra grayscale baru
     - lalu diterapkan threshold Otsu
   - Intinya:
     - mereka tidak pakai adaptive threshold seperti pipeline kamu
     - mereka lebih dulu menyaring background berdasarkan brightness di HSV
     - baru sesudah itu pakai Otsu global

6. Tahap `__outline_borders`: merapikan border dan menyambung sinyal yang terputus.
   - Lihat `Preprocessor.py:200-233`.
   - Yang dilakukan:
     - jika ada garis hitam tebal di 10 piksel terluar border, garis itu dihapus
     - lalu dicari baris paling atas dan paling bawah yang punya piksel hitam
     - jika ada gap kecil antarsinyal pada baris itu, gap tersebut diisi hitam
   - Tujuannya:
     - menghapus frame tepi
     - menyambung trace yang terpotong akibat crop atau keterbatasan ruang gambar

7. Tahap `SignalExtractor.extract_signals()`: mendeteksi row/ROI sinyal.
   - Lihat `ecg-miner/src/digitization/SignalExtractor.py:31-80` dan `:82-111`.
   - Algoritmanya:
     - lakukan sliding window vertikal sepanjang 10 piksel
     - hitung standar deviasi tiap window sepanjang lebar gambar
     - cari peak pada kurva standar deviasi
     - peak tertinggi dianggap sebagai pusat row/ROI sinyal
   - Jumlah ROI yang dicari = `layout[0] + len(rhythm)`.
     - contoh layout `3x4` dan 1 rhythm strip menghasilkan 4 ROI.

8. Untuk tiap kolom gambar, ia mendeteksi cluster piksel hitam.
   - Lihat `SignalExtractor.py:113-136`.
   - Cluster = rentang piksel hitam berurutan pada satu kolom.

9. Cluster antarkolom dihubungkan dengan dynamic programming.
   - Lihat `SignalExtractor.py:44-79`.
   - Skor koneksi mempertimbangkan:
     - skor sebelumnya
     - jarak vertikal cluster ke ROI
     - gap putih antara cluster lama dan cluster baru
   - Rumus idenya:
     - semakin dekat ke ROI, lebih baik
     - semakin kecil gap putih, lebih baik

10. Lalu dilakukan backtracking untuk mengambil jalur sinyal terbaik.
    - Lihat `SignalExtractor.py:164-213`.
    - Hasil tahap ini adalah sekumpulan titik `(x, y)` untuk tiap row sinyal.

11. Tahap `Postprocessor.__segment()`: memisahkan reference pulse digital dari sinyal utama.
    - Lihat `ecg-miner/src/digitization/Postprocessor.py:138-215`.
    - Yang dicari:
      - level datar awal = `0 mV`
      - level datar berikutnya = `1 mV`
    - Setelah reference pulse ditemukan:
      - bagian pulse dibuang dari sinyal
      - kalibrasi pixel-to-mV dihitung dari selisih `0mV` dan `1mV`
      - dipakai median selisih antar row agar lebih stabil

12. Tahap `__vectorize()`: ubah trace piksel menjadi sinyal numerik per lead.
    - Lihat `Postprocessor.py:217-290`.
    - Yang dilakukan:
      - semua sinyal diinterpolasi ke panjang yang sama
      - sinyal dibagi sesuai layout ECG
      - lead dipetakan ke urutan standard atau Cabrera
      - koordinat y diubah menjadi satuan relatif mV berdasarkan reference pulse
      - hasil disimpan ke `pandas.DataFrame`

13. Tahap `__get_trace()`: buat visualisasi hasil digitization.
    - Lihat `Postprocessor.py:292-361`.
    - Ia menggambar ulang sinyal hasil ekstraksi di atas citra crop.

14. Tahap akhir `Digitizer.digitize()`: simpan hasil.
    - Lihat `ecg-miner/src/digitization/Digitizer.py:52-88`.
    - Output yang disimpan:
      - file CSV berisi sinyal 12 lead
      - file PNG `_trace.png` berisi overlay trace
      - file metadata TXT jika OCR diaktifkan

### Asumsi format input yang cukup kuat

Dari `validation/render.py:15-159` terlihat bahwa data validasi mereka dirender dengan asumsi:

- layout default `3 x 4`
- rhythm strip tambahan, default `II`
- sample rate 500 Hz
- reference pulse bisa di kiri atau kanan
- ada lead separator dan lead name
- grid merah muda/merah dan sinyal hitam

Di GUI, default model mereka adalah:

- `layout = (3, 4)`
- `rhythm = Lead.II`
- `rp_at_right = False`
- `cabrera = False`

Lihat `ecg-miner/src/app/model/Model.py:14-22`.

## 2. Apa perbedaannya dengan preprocessing buatan saya sebelumnya?

Pipeline kamu sebelumnya, dari percakapan dan kode yang kamu buat, secara garis besar adalah:

1. crop area ECG secara manual berdasarkan koordinat
2. binarisasi dengan adaptive threshold
3. hapus noise kecil dengan connected components
4. hapus teks dengan mask manual
5. upscale menjadi versi "HD"
6. split menjadi `12 short lead + 1 long lead`
7. simpan hasil sebagai gambar per bagian

Sedangkan `ecg-miner` melakukan:

1. crop area ECG otomatis dengan contour terbesar
2. buang grid/background dengan HSV mask + Otsu
3. rapikan border dan sambung sinyal di tepi
4. deteksi ROI row otomatis
5. tracing sinyal antar kolom dengan dynamic programming
6. deteksi reference pulse
7. konversi trace menjadi data numerik per lead
8. simpan CSV dan trace visual

### Perbedaan paling penting

#### A. Tujuan pipeline

- Pipeline kamu:
  - tujuan utamanya menghasilkan gambar ECG yang lebih bersih dan terpotong per lead
  - cocok untuk workflow berbasis image
- ECGMiner:
  - tujuan utamanya digitization
  - bukan sekadar membersihkan gambar, tapi mengubah gambar menjadi sinyal numerik

#### B. Cara lokalisasi area ECG

- Pipeline kamu: manual, berbasis koordinat tetap
- ECGMiner: otomatis, berbasis contour terbesar

#### C. Cara menghilangkan grid/noise

- Pipeline kamu: adaptive threshold + connected components + mask manual
- ECGMiner: HSV mask brightness + Otsu + border outlining

#### D. Cara memisahkan lead

- Pipeline kamu: split geometris tetap berdasarkan koordinat short/long
- ECGMiner: tidak split kotak-kotak gambar; ia menelusuri trace sinyal lalu memetakan ke lead berdasarkan layout dan rhythm strip

#### E. Kebutuhan intervensi manual

- Pipeline kamu: tinggi, terutama di koordinat crop/split dan mask teks
- ECGMiner: lebih otomatis, tapi lebih sensitif ke asumsi format gambar

## 3. Apa output akhir dari preprocessing? Apakah sebuah gambar, atau apa?

Kalau yang dimaksud adalah output akhir pipeline `ecg-miner`, maka output utamanya bukan gambar.

### Output akhir utamanya

1. CSV sinyal 12-lead
   - Disimpan di `Digitizer.py:76-77`
   - Ini adalah output paling penting
   - Isinya dataframe dengan kolom lead:
     - `I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`-`V6`

2. PNG trace hasil digitization
   - Disimpan di `Digitizer.py:78-82`
   - Ini semacam visualisasi/overlay hasil ekstraksi

3. TXT metadata OCR
   - Opsional
   - Disimpan di `Digitizer.py:83-88`

### Output preprocessing internal

Tahap preprocessing sendiri memang menghasilkan gambar antara:

- crop area ECG
- citra binarized tanpa grid dominan

Tetapi itu hanya intermediate result. Hasil akhir sistemnya adalah data sinyal numerik, bukan sekadar image cleaning.

## 4. Lebih unggul mana antara preprocessing dia dengan preprocessing saya sebelumnya?

Jawabannya tergantung tujuan kamu.

### Jika tujuan kamu adalah image-based workflow

Contoh:

- klasifikasi berbasis CNN dari citra lead
- segmentasi visual
- dataset yang akan tetap dipakai sebagai gambar
- layout ECG antar image konsisten dan kamu sudah tahu koordinatnya

Maka pipeline kamu bisa lebih unggul secara praktis, karena:

- lebih sederhana
- lebih mudah dikontrol
- lebih mudah diadaptasi ke dataset kamu
- text removal manual memberi hasil lebih konsisten jika format sangat tetap
- split short/long yang kamu buat cocok untuk training image model

### Jika tujuan kamu adalah signal digitization

Contoh:

- ingin memperoleh waveform numerik dari gambar ECG
- ingin menghitung fitur ECG di domain sinyal
- ingin membandingkan hasil digitized dengan sinyal asli

Maka `ecg-miner` jauh lebih unggul, karena:

- ia melakukan tracing sinyal, bukan sekadar crop gambar
- ia mendeteksi reference pulse untuk kalibrasi
- ia menghasilkan CSV sinyal per lead
- ia mendukung layout, rhythm strip, Cabrera, dan OCR metadata

### Penilaian yang lebih jujur

`ecg-miner` bukan "preprocessing yang lebih bagus" dalam semua kasus. Ia adalah sistem yang menyelesaikan masalah yang lebih besar.

- Untuk tugas image preparation: pipeline kamu lebih langsung dan sering lebih robust jika dataset sangat fixed.
- Untuk tugas digitization penuh: `ecg-miner` jauh lebih kaya dan secara metodologis lebih kuat.

## 5. Jika saya ingin memodifikasi codingannya agar bisa saya gunakan untuk dataset saya, bagaimana caranya?

Karena dataset kamu berbeda pada format letak ECG di dalam gambar, ada beberapa bagian yang hampir pasti harus diubah.

### A. Ubah tahap penentuan area ECG

File utama:

- `ecg-miner/src/digitization/Preprocessor.py:42-91`

Saat ini ECGMiner memilih bounding box terbesar dari hasil contour.

Jika dataset kamu:

- posisi ECG selalu tetap
- ada teks/header besar
- border kertas atau frame kadang lebih dominan daripada area sinyal

maka langkah ini kemungkinan tidak cukup stabil.

Yang bisa kamu lakukan:

1. Ganti `__img_partitioning()` dengan koordinat tetap seperti pipeline kamu.
2. Atau gunakan pendekatan hybrid:
   - crop kasar manual dulu
   - baru jalankan deteksi otomatis di dalam crop itu
3. Jika ada sedikit pergeseran antargambar:
   - pakai anchor/template matching
   - atau deteksi reference pulse sebagai landmark

Untuk dataset kamu, opsi paling realistis biasanya:

- tetap gunakan crop manual/semimanual yang sudah kamu buat
- masukkan hasil crop itu ke tahap digitization berikutnya

### B. Ubah tahap grid/background removal

File utama:

- `ecg-miner/src/digitization/Preprocessor.py:138-166`

Saat ini mereka pakai HSV threshold:

- `lower = [0, 0, 168]`
- `upper = [255, 255, 255]`

Ini cocok jika:

- background terang
- grid berwarna cukup terang
- sinyal hitam cukup kontras

Kalau dataset kamu berbeda, kamu mungkin perlu:

1. ubah threshold HSV
2. ganti ke adaptive threshold seperti pipeline kamu
3. tambahkan text mask removal seperti pipeline kamu
4. tambahkan deskew jika gambar miring

Praktiknya, untuk dataset kamu, bagian ini justru kandidat terbaik untuk diganti dengan preprocessing milikmu sendiri.

### C. Ubah asumsi layout ECG

File terkait:

- `ecg-miner/src/app/model/Model.py:14-22`
- `ecg-miner/src/digitization/Digitizer.py:21-50`
- `ecg-miner/src/digitization/Postprocessor.py:237-289`

Kamu harus set dengan benar:

- `layout`
- `rhythm`
- `rp_at_right`
- `cabrera`

Kalau dataset kamu adalah format 12 short lead + 1 long lead standar, maka kemungkinan:

- `layout = (3, 4)`
- `rhythm = [Lead.II]` atau lead lain sesuai strip panjang sebenarnya
- `cabrera = False`
- `rp_at_right` harus disesuaikan dengan posisi pulse di dataset

Kalau salah di parameter ini, hasil mapping lead akan salah meskipun preprocessing benar.

### D. Ubah deteksi ROI jika posisi baris tidak serapi dataset paper

File utama:

- `ecg-miner/src/digitization/SignalExtractor.py:82-111`

Saat ini ROI baris ditentukan dari peak standar deviasi per row.

Masalah yang mungkin muncul pada dataset kamu:

- jarak antarbaris tidak konsisten
- ada teks yang mengganggu row statistics
- ada long lead yang tinggi/offset-nya berbeda
- garis sinyal lebih tipis/putus

Yang bisa diubah:

1. ubah `WINDOW` dan `min_distance`
2. hitung ROI dari hasil split manual row dulu
3. bypass ROI otomatis dan langsung kirim row yang sudah diketahui

Kalau layout dataset kamu memang sedikit bergeser tapi masih cukup konsisten, strategi terbaik sering kali:

- crop area utama manual
- split row manual
- baru jalankan tracing sinyal per row

### E. Ubah logika segmentasi reference pulse jika bentuk pulse berbeda atau tidak ada

File utama:

- `ecg-miner/src/digitization/Postprocessor.py:138-215`

Pipeline ini sangat bergantung pada reference pulse untuk konversi pixel ke mV.

Kalau pada dataset kamu:

- pulse bentuknya berbeda
- pulse tidak konsisten
- pulse tertutup teks
- pulse tidak ada

maka fungsi `__segment()` perlu diubah.

Alternatifnya:

1. gunakan kalibrasi tetap jika skala semua gambar sama
2. deteksi pulse dengan rule khusus dataset kamu
3. jika hanya butuh image dataset, lewati tahap ini seluruhnya

### F. Tentukan apakah kamu benar-benar perlu seluruh ECGMiner

Ini keputusan paling penting.

#### Opsi 1: Pakai preprocessing kamu + model image

Pilih ini jika targetmu tetap berbasis image.

Yang dipakai:

- crop manual
- binarisasi/adaptive threshold
- hapus teks
- HD
- split short/long

Dalam skenario ini, kamu tidak perlu memaksa memakai `SignalExtractor` dan `Postprocessor`.

#### Opsi 2: Pakai preprocessing kamu sebagai front-end, lalu sambungkan ke ECGMiner

Pilih ini jika kamu ingin digitization numerik tetapi format gambar dataset kamu berbeda.

Strateginya:

1. pakai crop dan text removal milikmu untuk menormalkan input
2. hasil normalized image diberikan ke pipeline ekstraksi sinyal ECGMiner
3. modifikasi `Preprocessor` agar tidak lagi mencari contour terbesar, atau langsung bypass tahap itu

Secara teknis, ini sering jadi kompromi terbaik.

#### Opsi 3: Adaptasi penuh ECGMiner ke dataset kamu

Pilih ini jika target akhir kamu memang waveform numerik per lead.

Bagian yang paling mungkin harus diubah:

1. `Preprocessor.__img_partitioning()`
2. `Preprocessor.__gridline_removal()`
3. `SignalExtractor.__get_roi()`
4. `Postprocessor.__segment()`
5. parameter `layout`, `rhythm`, `rp_at_right`, `cabrera`

## Rekomendasi praktis untuk dataset kamu

Berdasarkan pipeline yang sudah kamu bangun sebelumnya, rekomendasi paling pragmatis adalah:

1. Pertahankan preprocessing milikmu untuk:
   - crop area ECG
   - hapus teks
   - split short/long

2. Jangan langsung mengganti semuanya dengan ECGMiner.
   - Arsitekturnya berbeda tujuan.

3. Kalau kamu ingin belajar dari paper itu, adopsi bagian yang paling berguna saja:
   - ide crop area otomatis berbasis contour
   - ide background removal berbasis HSV
   - ide tracing sinyal antar kolom
   - ide reference-pulse calibration

4. Kalau targetmu nanti bergeser ke digitization sinyal numerik, baru gunakan ECGMiner sebagai dasar utama.

## Kesimpulan singkat

- `ecg-miner` bukan hanya preprocessing image; ia adalah pipeline digitization penuh.
- Output utamanya adalah CSV sinyal ECG, bukan gambar split.
- Pipeline kamu lebih cocok untuk kebutuhan image-based dataset preparation.
- ECGMiner lebih unggul jika target akhirnya adalah waveform extraction.
- Untuk memakai ECGMiner pada dataset kamu, bagian yang paling perlu diubah adalah:
  - penentuan area crop
  - background/grid removal
  - deteksi ROI
  - segmentasi reference pulse
  - parameter layout/rhythm
