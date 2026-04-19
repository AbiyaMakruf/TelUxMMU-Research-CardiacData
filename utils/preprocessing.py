import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def _list_class_folders(data_path, exclude_text_mask=True):
    """Get list of class folders, excluding .zip files and optionally text_mask folder.

    Helper function to standardize folder listing across preprocessing functions.

    Args:
        data_path (str): Path to the data directory.
        exclude_text_mask (bool): If True, exclude 'text_mask' folder. Defaults to True.

    Returns:
        list: List of class folder names (sorted).
    """
    list_folder = [f for f in os.listdir(data_path)
                   if not f.endswith('.zip') and os.path.isdir(os.path.join(data_path, f))]
    if exclude_text_mask:
        list_folder = [f for f in list_folder if f != 'text_mask']
    return sorted(list_folder)


def _clean_and_remove_text(img_path, text_mask_raw, scale=2):
    """Apply upscaled cleaning and text mask removal to ECG image.

    Reads image, scales it up, applies adaptive threshold to extract ECG signal,
    removes noise via connected components, then removes text using provided mask.

    Args:
        img_path (str): Path to the ECG image file.
        text_mask_raw (np.ndarray): Grayscale text mask (from cv2.imread).
        scale (int): Upscaling factor. Defaults to 2.

    Returns:
        np.ndarray: Cleaned ECG signal with text removed (white background, black signal).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img_res = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(img_res, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 8)

    # Connected components to filter out noise: keep only large/wide components
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

    # Apply text mask: set masked areas to white (remove text)
    mask_res = cv2.resize(text_mask_raw, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)

    kernel_text = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_wide = cv2.dilate(mask_bin, kernel_text, iterations=2)

    result_no_text = result.copy()
    result_no_text[mask_wide == 255] = 255

    return result_no_text


def preview_cropped_ecg_area(data_path, coordinate, random):
    """Display preview of ECG area cropping on sample images.

    Shows original image with red bounding box and cropped result side-by-side
    for one image from each class folder.

    Args:
        data_path (str): Path to raw data directory containing class folders.
        coordinate (dict): Dictionary with 'left', 'top', 'right', 'bottom' pixel coordinates.
        random (bool): If True, select random image; otherwise use first image.
    """
    list_folder = _list_class_folders(data_path, exclude_text_mask=False)
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
    """Crop ECG area from all images in all class folders.

    Extracts the ECG signal area (defined by coordinates) from each image
    and saves cropped versions to folder_target/crop_ecg_area/.

    Args:
        folder_source (str): Source directory with raw ECG images in class subfolders.
        folder_target (str): Target base directory where cropped images will be saved.
        coordinate (dict): Dictionary with 'left', 'top', 'right', 'bottom' pixel coordinates.
    """
    folder_target = f'{folder_target}/crop_ecg_area'
    os.makedirs(folder_target, exist_ok=True)

    list_folder = _list_class_folders(folder_source, exclude_text_mask=False)
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
    """Extract clean ECG signal from image using adaptive threshold and morphology.

    Reads grayscale ECG image, applies Gaussian blur, adaptive threshold to isolate
    dark ECG lines, morphological closing to connect broken lines, and connected
    components filtering to remove noise. Returns black ECG signal on white background.

    Args:
        img_path (str): Path to the ECG image file.

    Returns:
        np.ndarray: Cleaned ECG signal (grayscale, black signal on white background).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Adaptive threshold inverted (BINARY_INV) to extract dark ECG lines
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,   # block size
        10     # constant C
    )

    # Morphological closing to connect thin or broken ECG lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components: label each connected region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean = np.zeros_like(binary)

    min_area = 100
    min_width = 60

    # Keep only components that are large enough or wide enough (filters out noise)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area >= min_area or w >= min_width:
            clean[labels == i] = 255

    # Convert to black signal on white background
    result = np.ones_like(clean) * 255
    result[clean == 255] = 0
    return result

def preview_extract_ecg_signal(data_path, random):
    """Display preview of ECG signal extraction on sample images.

    Shows original cropped ECG image and cleaned signal result side-by-side
    for one image from each class folder.

    Args:
        data_path (str): Path to cropped ECG data directory containing class folders.
        random (bool): If True, select random image; otherwise use first image.
    """
    list_folder = _list_class_folders(data_path, exclude_text_mask=False)
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
    """Extract clean ECG signals from all cropped images in all class folders.

    Applies clean_ecg_signal() to each image and saves cleaned versions to
    folder_target/extracted_ecg_signal/.

    Args:
        folder_source (str): Source directory with cropped ECG images in class subfolders.
        folder_target (str): Target base directory where cleaned signals will be saved.
    """
    folder_target = f'{folder_target}/extracted_ecg_signal'
    os.makedirs(folder_target, exist_ok=True)

    list_folder = _list_class_folders(folder_source, exclude_text_mask=False)
    
    for folder in list_folder:
        folder_path = os.path.join(folder_source, folder)
        images = os.listdir(folder_path)
        os.makedirs(os.path.join(folder_target, folder), exist_ok=True)
        
        for img_name in tqdm.tqdm(images, desc=f"Cleaning {folder}"):
            img_path = os.path.join(folder_path, img_name)
            img_clean = clean_ecg_signal(img_path)
            cv2.imwrite(os.path.join(folder_target, folder, img_name), img_clean)

def create_text_mask(data_path, folder_target):
    """Create a text mask template from a Normal ECG image.

    Cleans an ECG image and inverts it to create a mask of text regions.
    This mask is used later to remove text from other ECG images.

    Args:
        data_path (str): Path to cropped ECG data containing ECG_Normal folder.
        folder_target (str): Target directory where text_mask will be saved.
    """
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
    """Display preview of text removal on sample ECG images.

    Shows extracted ECG signal and cleaned result (with text removed) side-by-side
    for one image from each class folder.

    Args:
        data_path (str): Path to extracted ECG signals directory containing class folders.
        text_mask_path (str): Path to the text mask image file.
        random (bool): If True, select random image; otherwise use first image.
    """
    text_mask_raw = cv2.imread(text_mask_path, cv2.IMREAD_GRAYSCALE)
    if text_mask_raw is None:
        print(f"Error: Mask tidak ditemukan di {text_mask_path}")
        return

    list_folder = _list_class_folders(data_path, exclude_text_mask=True)

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

    total_rows = len(all_images_data)
    if total_rows == 0:
        return

    fig, axes = plt.subplots(total_rows, 2, figsize=(16, 6 * total_rows), dpi=100)
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, data in enumerate(all_images_data):
        result_no_text = _clean_and_remove_text(data["path"], text_mask_raw, scale=2)
        if result_no_text is None:
            continue

        # Read original for comparison
        img = cv2.imread(data["path"], cv2.IMREAD_GRAYSCALE)
        result = np.ones_like(img) * 255
        result[img == 0] = 0  # Approximate original signal for display

        axes[idx, 0].imshow(result, cmap="gray")
        axes[idx, 0].set_title(f"Original Signal: {data['folder']}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(result_no_text, cmap="gray")
        axes[idx, 1].set_title("Text Removed")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()

def delete_text_from_ecg(data_path, folder_target, text_mask_path):
    """Remove text from all ECG images in all class folders.

    Applies text removal using the provided mask to each image and saves
    cleaned versions to folder_target, preserving class folder structure.

    Args:
        data_path (str): Source directory with extracted ECG signals in class subfolders.
        folder_target (str): Target directory where cleaned images will be saved.
        text_mask_path (str): Path to the text mask image file.
    """
    text_mask_raw = cv2.imread(text_mask_path, cv2.IMREAD_GRAYSCALE)
    if text_mask_raw is None:
        print(f"Error: Mask tidak ditemukan di {text_mask_path}")
        return

    os.makedirs(folder_target, exist_ok=True)

    list_folder = _list_class_folders(data_path, exclude_text_mask=True)

    for folder in list_folder:
        folder_path = os.path.join(data_path, folder)
        images = [i for i in os.listdir(folder_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]
        os.makedirs(os.path.join(folder_target, folder), exist_ok=True)

        for img_name in tqdm.tqdm(images, desc=f"Removing Text {folder}"):
            img_path = os.path.join(folder_path, img_name)
            result_no_text = _clean_and_remove_text(img_path, text_mask_raw, scale=2)

            if result_no_text is None:
                continue

            output_path = os.path.join(folder_target, folder, img_name)
            cv2.imwrite(output_path, result_no_text)
    
def preview_cropped_short_and_long_leads(data_path, random, coordinate_short, coordinate_long):
    """Display preview of short and long lead cropping on sample images.

    Shows original image with short (red, 3x4) and long (cyan) lead rectangles,
    plus the cropped short leads grid and long lead separately for one image
    from each class folder.

    Args:
        data_path (str): Path to cleaned ECG signals directory containing class folders.
        random (bool): If True, select random image; otherwise use first image.
        coordinate_short (dict): Dictionary with x1, y1, x2, y2 for short leads area.
        coordinate_long (dict): Dictionary with x1, y1, x2, y2 for long lead area.
    """
    list_folder = _list_class_folders(data_path, exclude_text_mask=True)

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
    """Crop ECG images into 12 short leads and 1 long lead per sample.

    Extracts short leads (12 leads in 3x4 grid) and long lead (1 lead) from each ECG image,
    creating per-sample folders to organize individual leads for downstream analysis.

    Output structure:
        folder_target/
        └── [class_folder]/
            └── [sample_name]/
                ├── 01_lead_I.png
                ├── 02_lead_II.png
                ├── ...
                └── 13_long_lead.png

    Args:
        folder_source (str): Source directory with cleaned ECG images in class subfolders.
        folder_target (str): Target base directory where per-sample lead folders will be created.
        coordinate_short (dict): Dictionary with x1, y1, x2, y2 for short leads area (3x4 grid).
        coordinate_long (dict): Dictionary with x1, y1, x2, y2 for long lead area.
        lead_mapping (dict): Dictionary mapping lead indices (1-12) to lead names.
    """
    list_folder = _list_class_folders(folder_source, exclude_text_mask=True)

    for class_folder in list_folder:
        class_source_path = os.path.join(folder_source, class_folder)
        images = [i for i in os.listdir(class_source_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in tqdm.tqdm(images, desc=f"Processing {class_folder}"):
            img_path = os.path.join(class_source_path, img_name)
            img = Image.open(img_path)

            sample_name = os.path.splitext(img_name)[0]
            sample_dir = os.path.join(folder_target, class_folder, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            s_left, s_top, s_right, s_bottom = map(int, coordinate_short.values())

            short_w = s_right - s_left
            short_h = s_bottom - s_top
            cell_w = short_w / 4
            cell_h = short_h / 3

            # Extract 12 short leads from 3x4 grid
            for lead_idx, lead_name in lead_mapping.items():
                # Map lead index (1-12) to grid position: row = index // 4, col = index % 4
                # Indices 1-4 in row 0, 5-8 in row 1, 9-12 in row 2
                row = (lead_idx - 1) // 4
                col = (lead_idx - 1) % 4

                left = s_left + (col * cell_w)
                top = s_top + (row * cell_h)
                right = left + cell_w
                bottom = top + cell_h

                img_lead = img.crop((int(left), int(top), int(right), int(bottom)))

                lead_filename = f"{lead_idx:02d}_lead_{lead_name}.png"
                img_lead.save(os.path.join(sample_dir, lead_filename))

            # Extract 1 long lead
            l_left, l_top, l_right, l_bottom = map(int, coordinate_long.values())

            img_long = img.crop((l_left, l_top, l_right, l_bottom))

            long_filename = "13_long_lead.png"
            img_long.save(os.path.join(sample_dir, long_filename))