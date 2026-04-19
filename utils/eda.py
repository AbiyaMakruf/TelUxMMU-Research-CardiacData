import os
import zipfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def download_and_extract_dataset(url, zip_name, data_path):
    """Download and extract ECG dataset from remote URL.

    Downloads a zip file from the specified URL with Mozilla User-Agent header,
    then extracts it to the specified data path. Skips download if file already exists.

    Args:
        url (str): URL to download the dataset from.
        zip_name (str): Name of the zip file to create/use.
        data_path (str): Path where the zip will be saved and extracted.
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    if not os.path.exists(os.path.join(data_path, zip_name)):
        print(f"Downloading dataset from {url}...")
        
        # Buat opener untuk menambahkan header User-Agent
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, os.path.join(data_path, zip_name))
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

    print("Extracting dataset...")
    with zipfile.ZipFile(os.path.join(data_path, zip_name), 'r') as zip_ref:
        zip_ref.extractall(data_path)
        print(f"Dataset extracted to {data_path}.")
        
def rename_folder(data_path, folder_mapping):
    """Rename folders in data directory according to provided mapping.

    Iterates through the folder_mapping dictionary and renames folders in data_path
    from old names to new names. Handles cases where source folder doesn't exist.

    Args:
        data_path (str): Path containing folders to rename.
        folder_mapping (dict): Dictionary mapping old folder names to new names.
    """
    for old_name, new_name in folder_mapping.items():
        old_path = os.path.join(data_path, old_name)
        new_path = os.path.join(data_path, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed '{old_name}' to '{new_name}'.")
        else:
            print(f"Folder '{old_name}' not found. Skipping rename.")
    
def count_dataset(data_path):
    """Count number of images in each class folder.

    Scans the data directory and counts the number of files in each subdirectory,
    returning results as a pandas DataFrame.

    Args:
        data_path (str): Path to the data directory containing class folders.

    Returns:
        pd.DataFrame: DataFrame with columns 'folder' and 'count'.
    """
    data = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            num_files = len(os.listdir(folder_path))
            data.append({"folder": folder, "count": num_files})
    
    return pd.DataFrame(data)

def preview_images(data_path, num_images, random):
    """Display a grid of sample images from each class folder.

    Creates a matplotlib figure showing num_images from each class folder.
    Images can be selected sequentially or randomly.

    Args:
        data_path (str): Path to the data directory containing class folders.
        num_images (int): Number of images to display per class.
        random (bool): If True, randomly select images; otherwise use first N images.
    """
    list_folder = [f for f in os.listdir(data_path) if not f.endswith('.zip')]
    num_folder = len(list_folder)
    image_files = []

    for folder in list_folder:
        if not random:
            image = [f for f in os.listdir(os.path.join(data_path, folder))][:num_images]
            image_files.append(image)
        else:
            image = [f for f in os.listdir(os.path.join(data_path, folder))]
            image_files.append(np.random.choice(image, num_images, replace=False))

    plt.figure(figsize=(20, 20))
    for i in range(num_folder):
        for j in range(num_images):
            # Create subplot at row i, column j; num_folder rows, num_images columns
            img = plt.imread(os.path.join(data_path, list_folder[i], image_files[i][j]))
            plt.subplot(num_folder, num_images, i*num_images + j + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(list_folder[i])
    plt.tight_layout()
    plt.show()