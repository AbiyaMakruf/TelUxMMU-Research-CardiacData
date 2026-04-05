import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def preview_cropped_ecg_area(data_path, coordinate, random):
    list_folder = [f for f in os.listdir(data_path) if not f.endswith('.zip')]
    all_images_data = []
    
    for folder in list_folder:
        folder_path = os.path.join(data_path, folder)
        images = os.listdir(folder_path)
        
        if random:
            images = np.random.choice(images, 1, replace=False)
        else:
            images = images[:1]
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            all_images_data.append({
                "path": img_path,
                "folder": folder,
                "name": img_name
            })
    
    # Create subplots for all images in one canvas
    total_rows = len(all_images_data)
    fig, axes = plt.subplots(total_rows, 2, figsize=(12, 4 * total_rows))
    
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    left, top, right, bottom = map(int, coordinate.values())

    for idx, data in enumerate(all_images_data):
        img = Image.open(data["path"])
        
        # Kolom 0: Gambar Asli + Kotak Merah
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"[{data['folder']}] - {data['name']}")
        rect = plt.Rectangle((left, top), right - left, bottom - top, 
                            linewidth=2, edgecolor="red", facecolor="none")
        axes[idx, 0].add_patch(rect)
        axes[idx, 0].axis("off")
        
        # Kolom 1: Hasil Crop
        img_crop = img.crop((left, top, right, bottom))
        axes[idx, 1].imshow(img_crop)
        axes[idx, 1].set_title(f"Cropped {data['folder']}")
        axes[idx, 1].axis("off")

    plt.suptitle("ECG Cropping Preview Across All Folders", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def crop_ecg_area(folder_source, folder_target, coordinate):
    folder_target = f'{folder_target}/crop_ecg_area'
    os.makedirs(folder_target, exist_ok=True)

    list_folder = [f for f in os.listdir(folder_source) if not f.endswith('.zip')]
    left, top, right, bottom = map(int, coordinate.values())

    for folder in list_folder:
        folder_path = os.path.join(folder_source, folder)
        images = os.listdir(folder_path)
        os.makedirs(os.path.join(folder_target, folder), exist_ok=True)
        
        for img_name in tqdm.tqdm(images, desc=f"Cropping {folder}"):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path)
            img_crop = img.crop((left, top, right, bottom))
            img_crop.save(os.path.join(folder_target, folder, img_name))

def clean_ecg_signal(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1. Blur ringan agar noise berkurang
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # 2. Ambil objek gelap (garis ECG) dengan adaptive threshold
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,   # block size
        10     # constant C
    )

    # 3. Sambungkan garis yang tipis / sedikit putus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. Hapus titik-titik kecil dengan connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean = np.zeros_like(binary)

    min_area = 100
    min_width = 60

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # simpan komponen yang cukup besar / memanjang
        if area >= min_area or w >= min_width:
            clean[labels == i] = 255

    # 5. Ubah jadi garis hitam di background putih
    result = np.ones_like(clean) * 255
    result[clean == 255] = 0
    return result

def preview_extract_ecg_signal(data_path, random):
    list_folder = [f for f in os.listdir(data_path) if not f.endswith('.zip')]
    all_images_data = []
    
    for folder in list_folder:
        folder_path = os.path.join(data_path, folder)
        images = os.listdir(folder_path)
        
        if random:
            images = np.random.choice(images, 1, replace=False)
        else:
            images = images[:1]
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            all_images_data.append({
                "path": img_path,
                "folder": folder,
                "name": img_name
            })
    
    # Create subplots for all images in one canvas
    total_rows = len(all_images_data)
    fig, axes = plt.subplots(total_rows, 2, figsize=(12, 4 * total_rows))
    
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, data in enumerate(all_images_data):
        img = Image.open(data["path"])
        
        # Kolom 0: Gambar Asli
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"[{data['folder']}] - {data['name']}")
        axes[idx, 0].axis("off")
        
        # Kolom 1: Hasil Cleaning
        img_clean = clean_ecg_signal(data["path"])
        axes[idx, 1].imshow(img_clean, cmap='gray')
        axes[idx, 1].set_title(f"Cleaned {data['folder']}")
        axes[idx, 1].axis("off")

    plt.suptitle("ECG Cleaning Preview Across All Folders", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def extract_ecg_signal(folder_source, folder_target):
    folder_target = f'{folder_target}/extracted_ecg_signal'
    os.makedirs(folder_target, exist_ok=True)

    list_folder = [f for f in os.listdir(folder_source) if not f.endswith('.zip')]
    
    for folder in list_folder:
        folder_path = os.path.join(folder_source, folder)
        images = os.listdir(folder_path)
        os.makedirs(os.path.join(folder_target, folder), exist_ok=True)
        
        for img_name in tqdm.tqdm(images, desc=f"Cleaning {folder}"):
            img_path = os.path.join(folder_path, img_name)
            img_clean = clean_ecg_signal(img_path)
            cv2.imwrite(os.path.join(folder_target, folder, img_name), img_clean)

def create_text_mask(data_path, folder_target):
    img_path = os.path.join(data_path, "ECG_Normal", os.listdir(os.path.join(data_path, "ECG_Normal"))[0])
    result = clean_ecg_signal(img_path)
    result_invert = cv2.bitwise_not(result)
    os.makedirs(os.path.join(f'{folder_target}', "text_mask"), exist_ok=True)
    cv2.imwrite(os.path.join(f'{folder_target}/text_mask', "text_mask.png"), result_invert)
    
    # Preview hasil
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].imshow(result, cmap='gray')
    axes[0].set_title("Cleaned ECG Signal")
    axes[0].axis("off")
    
    axes[1].imshow(result_invert, cmap='gray')
    axes[1].set_title("Text Mask")
    axes[1].axis("off")
    
    plt.suptitle("Text Mask Preview", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def preview_delete_text(data_path, text_mask_path, random):
    # 1. Load Mask Utama (Grayscale)
    text_mask_raw = cv2.imread(text_mask_path, cv2.IMREAD_GRAYSCALE)
    if text_mask_raw is None:
        print(f"Error: Mask tidak ditemukan di {text_mask_path}")
        return

    # 2. List Folder (kecuali .zip dan folder mask itu sendiri)
    list_folder = [f for f in os.listdir(data_path) if not f.endswith('.zip') and os.path.isdir(os.path.join(data_path, f))]
    list_folder = [f for f in list_folder if f != 'text_mask']
    
    all_images_data = []
    for folder in list_folder:
        folder_path = os.path.join(data_path, folder)
        images = [i for i in os.listdir(folder_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images: continue
        
        img_name = np.random.choice(images) if random else images[0]
        all_images_data.append({
            "path": os.path.join(folder_path, img_name),
            "folder": folder,
            "name": img_name
        })

    total_rows = len(all_images_data)
    if total_rows == 0: return

    fig, axes = plt.subplots(total_rows, 2, figsize=(16, 6 * total_rows), dpi=100)
    if total_rows == 1: axes = np.expand_dims(axes, axis=0)

    for idx, data in enumerate(all_images_data):
        # --- PROSES EKSTRAKSI SINYAL (Sesuai Logika Anda yang Berhasil) ---
        img = cv2.imread(data["path"], cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        scale = 2
        img_res = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Adaptive Threshold untuk ambil garis ECG (hitam)
        blur = cv2.GaussianBlur(img_res, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 8)
        
        # Pembersihan komponen kecil (Noise Removal)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        min_area = 100 * (scale ** 2)
        min_width = 60 * scale

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= min_area or w >= min_width:
                clean[labels == i] = 255

        # Ubah ke Sinyal Hitam di Background Putih
        result = np.ones_like(clean) * 255
        result[clean == 255] = 0

        # --- PROSES PENGHAPUSAN TEKS BERDASARKAN MASK ---
        # Resize mask ke ukuran gambar hasil resize (HD)
        mask_res = cv2.resize(text_mask_raw, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)
        
        # Perlebar mask teks sedikit agar bersih
        kernel_text = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_wide = cv2.dilate(mask_bin, kernel_text, iterations=2)

        # LOGIKA KRUSIAL: Ubah area mask menjadi PUTIH (Background)
        result_no_text = result.copy()
        result_no_text[mask_wide == 255] = 255 

        # --- VISUALISASI ---
        axes[idx, 0].imshow(result, cmap="gray")
        axes[idx, 0].set_title(f"Original Signal: {data['folder']}")
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(result_no_text, cmap="gray")
        axes[idx, 1].set_title("Text Removed")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()

def delete_text_from_ecg(data_path, folder_target, text_mask_path):
    text_mask_raw = cv2.imread(text_mask_path, cv2.IMREAD_GRAYSCALE)
    if text_mask_raw is None:
        print(f"Error: Mask tidak ditemukan di {text_mask_path}")
        return

    os.makedirs(folder_target, exist_ok=True)

    list_folder = [f for f in os.listdir(data_path) if not f.endswith('.zip') and os.path.isdir(os.path.join(data_path, f))]
    list_folder = [f for f in list_folder if f != 'text_mask']
    
    for folder in list_folder:
        folder_path = os.path.join(data_path, folder)
        images = [i for i in os.listdir(folder_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]
        os.makedirs(os.path.join(folder_target, folder), exist_ok=True)

        for img_name in tqdm.tqdm(images, desc=f"Removing Text {folder}"):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            scale = 2
            img_res = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            blur = cv2.GaussianBlur(img_res, (3, 3), 0)
            binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 31, 8)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            clean = np.zeros_like(binary)
            min_area = 100 * (scale ** 2)
            min_width = 60 * scale

            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area >= min_area or w >= min_width:
                    clean[labels == i] = 255

            result = np.ones_like(clean) * 255
            result[clean == 255] = 0

            mask_res = cv2.resize(text_mask_raw, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, mask_bin = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)
            
            kernel_text = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_wide = cv2.dilate(mask_bin, kernel_text, iterations=2)

            result_no_text = result.copy()
            result_no_text[mask_wide == 255] = 255

            output_path = os.path.join(folder_target, folder, img_name)
            cv2.imwrite(output_path, result_no_text)
    
def preview_cropped_short_and_long_leads(data_path, random, coordinate_short, coordinate_long):
    """Preview how images will be cropped into short and long lead areas."""
    list_folder = [
        f for f in os.listdir(data_path)
        if not f.endswith('.zip') and os.path.isdir(os.path.join(data_path, f))
    ]
    list_folder = [f for f in list_folder if f != 'text_mask']

    if not list_folder:
        print("No folders found")
        return

    all_images_data = []
    for folder in list_folder:
        folder_path = os.path.join(data_path, folder)
        images = [i for i in os.listdir(folder_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not images:
            continue

        img_name = np.random.choice(images) if random else images[0]
        all_images_data.append({
            "path": os.path.join(folder_path, img_name),
            "folder": folder,
            "name": img_name
        })

    if not all_images_data:
        print("No images found")
        return

    total_rows = len(all_images_data)
    fig, axes = plt.subplots(total_rows, 3, figsize=(18, 6 * total_rows), dpi=100)
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    left_s, top_s, right_s, bottom_s = map(int, coordinate_short.values())
    left_l, top_l, right_l, bottom_l = map(int, coordinate_long.values())

    for idx, data in enumerate(all_images_data):
        img = Image.open(data["path"])

        # Original with short and long lead rectangles
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Original: {data['folder']}")

        rect_short = plt.Rectangle(
            (left_s, top_s),
            right_s - left_s,
            bottom_s - top_s,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        axes[idx, 0].add_patch(rect_short)

        rect_long = plt.Rectangle(
            (left_l, top_l),
            right_l - left_l,
            bottom_l - top_l,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none"
        )
        axes[idx, 0].add_patch(rect_long)

        axes[idx, 0].axis("off")

        # Short lead with 3x4 grid
        img_short = img.crop((left_s, top_s, right_s, bottom_s))
        axes[idx, 1].imshow(img_short)
        axes[idx, 1].set_title("Short Lead (3x4 grid)")
        axes[idx, 1].axis("off")

        short_w = right_s - left_s
        short_h = bottom_s - top_s
        cell_w = short_w / 4
        cell_h = short_h / 3

        for c in range(4):
            for r in range(3):
                x = c * cell_w
                y = r * cell_h
                rect = plt.Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none"
                )
                axes[idx, 1].add_patch(rect)

                n = r * 4 + c + 1
                axes[idx, 1].text(
                    x + cell_w / 2,
                    y + cell_h / 2,
                    f"S{n}",
                    color="yellow",
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1)
                )

        # Long lead
        img_long = img.crop((left_l, top_l, right_l, bottom_l))
        axes[idx, 2].imshow(img_long)
        axes[idx, 2].set_title("Long Lead")
        axes[idx, 2].axis("off")

    plt.suptitle("ECG Lead Cropping Preview", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()

def crop_short_and_long_leads(folder_source, folder_target, coordinate_short, coordinate_long, lead_mapping):
    """
    Crops ECG images into 12 short leads and 1 long lead, saved into sample-specific folders.
    
    Output structure:
    folder_target/
    └── [class_folder]/
        └── [sample_name]/           <-- Folder per gambar asli
            ├── 01_lead_I.png
            ├── 02_lead_II.png
            ├── ...
            └── 13_long_lead.png
    """
    
    # Ambil daftar folder kelas (ECG_Abnormal, dll)
    list_folder = [f for f in os.listdir(folder_source) 
                  if not f.endswith('.zip') and os.path.isdir(os.path.join(folder_source, f))]
    list_folder = [f for f in list_folder if f != 'text_mask']
    
    for class_folder in list_folder:
        class_source_path = os.path.join(folder_source, class_folder)
        images = [i for i in os.listdir(class_source_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm.tqdm(images, desc=f"Processing {class_folder}"):
            img_path = os.path.join(class_source_path, img_name)
            img = Image.open(img_path)
            
            # 1. Buat folder khusus untuk sampel ini (Sample ID-Based)
            # Contoh: folder_target/ECG_Abnormal/HB(1)/
            sample_name = os.path.splitext(img_name)[0]
            sample_dir = os.path.join(folder_target, class_folder, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            # --- PROSES SHORT LEADS (12 leads) ---
            # Ambil koordinat area 3x4
            s_left, s_top, s_right, s_bottom = map(int, coordinate_short.values())
            
            short_w = s_right - s_left
            short_h = s_bottom - s_top
            cell_w = short_w / 4
            cell_h = short_h / 3
            
            for lead_idx, lead_name in lead_mapping.items():
                # Tentukan posisi sel dalam grid 3x4
                # lead_idx 1-4 (baris 0), 5-8 (baris 1), 9-12 (baris 2)
                row = (lead_idx - 1) // 4
                col = (lead_idx - 1) % 4
                
                # Koordinat relatif terhadap gambar asli
                left = s_left + (col * cell_w)
                top = s_top + (row * cell_h)
                right = left + cell_w
                bottom = top + cell_h
                
                img_lead = img.crop((int(left), int(top), int(right), int(bottom)))
                
                # Simpan dengan nomor urut agar mudah di-sort saat loading (01, 02, ... 12)
                lead_filename = f"{lead_idx:02d}_lead_{lead_name}.png"
                img_lead.save(os.path.join(sample_dir, lead_filename))

            # --- PROSES LONG LEAD (1 lead) ---
            l_left, l_top, l_right, l_bottom = map(int, coordinate_long.values())
            
            img_long = img.crop((l_left, l_top, l_right, l_bottom))
            
            # Simpan sebagai urutan ke-13
            long_filename = "13_long_lead.png"
            img_long.save(os.path.join(sample_dir, long_filename))