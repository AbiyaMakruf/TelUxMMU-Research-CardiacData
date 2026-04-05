import os
import zipfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def download_and_extract_dataset(url, zip_name, data_path):
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
    
    for old_name, new_name in folder_mapping.items():
        old_path = os.path.join(data_path, old_name)
        new_path = os.path.join(data_path, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed '{old_name}' to '{new_name}'.")
        else:
            print(f"Folder '{old_name}' not found. Skipping rename.")
    
def count_dataset(data_path):
    data = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            num_files = len(os.listdir(folder_path))
            data.append({"folder": folder, "count": num_files})
    
    return pd.DataFrame(data)

def preview_images(data_path, num_images, random):
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

    # plot kesamping sebanyak num_images, plot kebawah sebanyak num_folder
    # buat subplot dengan jumlah num_folder x num_images
    plt.figure(figsize=(20, 20))    
    for i in range(num_folder):
        for j in range(num_images):
            img = plt.imread(os.path.join(data_path, list_folder[i], image_files[i][j]))
            plt.subplot(num_folder, num_images, i*num_images + j + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(list_folder[i])
    plt.tight_layout()
    plt.show()