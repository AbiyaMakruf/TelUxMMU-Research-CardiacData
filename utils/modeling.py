import os
import time
import tempfile
import shutil
import contextlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlflow
import dagshub

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    MobileNet_V2_Weights,
    EfficientNet_V2_S_Weights,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIMB_LEAD_PREFIXES = [
    "01_lead_lead_I",
    "05_lead_lead_II",
    "09_lead_lead_III",
    "02_lead_a_VR",
    "06_lead_a_VL",
    "10_lead_a_VF",
]

PRECORDIAL_LEAD_PREFIXES = [
    "03_lead_v_1",
    "07_lead_v_2",
    "11_lead_v_3",
    "04_lead_v_4",
    "08_lead_v5",
    "12_lead_v_6",
]

LONG_LEAD_PREFIX = "13_long_lead"

SICK_FOLDERS = {"ECG_MI", "ECG_Abnormal", "ECG_HistoryMI"}
HEALTHY_FOLDERS = {"ECG_Normal"}

LABEL_4CLASS = {
    "ECG_Normal": 0,
    "ECG_Abnormal": 1,
    "ECG_MI": 2,
    "ECG_HistoryMI": 3,
}

LABEL_2CLASS = {
    "ECG_Normal": 0,
    "ECG_Abnormal": 1,
    "ECG_MI": 1,
    "ECG_HistoryMI": 1,
}

CLASS_NAMES_4 = ["Normal", "Abnormal", "MI", "History MI"]
CLASS_NAMES_2 = ["Sehat", "Sakit"]

SCHEME_NAMES = {
    "scheme1": "Skema 1 – Gambar Clean",
    "scheme2": "Skema 2 – 12 Short Lead + Long Lead",
    "scheme3": "Skema 3 – 6 Tungkai + 6 Precordial + Long Lead",
}


# ---------------------------------------------------------------------------
# Dagshub / MLflow Setup
# ---------------------------------------------------------------------------

def init_dagshub(repo_owner: str, repo_name: str):
    """Initialize DagsHub MLflow tracking for experiment logging.

    Authenticates with DagsHub and sets the MLflow tracking URI to the
    remote DagsHub repository so all runs are logged there.

    Args:
        repo_owner (str): DagsHub username / organization name.
        repo_name (str): DagsHub repository name.
    """
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    print(f"DagsHub initialized: https://dagshub.com/{repo_owner}/{repo_name}")


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def _get_image_files_scheme1(data_root: str):
    """Collect (image_path, folder_name) pairs for Scheme 1 (clean images).

    Walks through data_root/preprocessed/clean_ecg_signal/ and collects
    all image files along with their class folder name.

    Args:
        data_root (str): Root data directory (e.g. 'data').

    Returns:
        list[tuple[str, str]]: List of (absolute_image_path, class_folder_name).
    """
    base = os.path.join(data_root, "preprocessed", "clean_ecg_signal")
    records = []
    for cls_folder in sorted(os.listdir(base)):
        cls_path = os.path.join(base, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                records.append((os.path.join(cls_path, fname), cls_folder))
    return records


def _get_image_files_scheme2(data_root: str):
    """Collect (lead_paths_dict, folder_name) pairs for Scheme 2 (12 short + long lead).

    Walks through data_root/preprocessed/cropped_leads/ and for each sample
    collects all 13 lead file paths grouped by lead name prefix.

    Args:
        data_root (str): Root data directory (e.g. 'data').

    Returns:
        list[tuple[dict, str]]: Each element is (lead_dict, class_folder_name),
            where lead_dict maps lead_prefix -> absolute_path.
    """
    base = os.path.join(data_root, "preprocessed", "cropped_leads")
    records = []
    for cls_folder in sorted(os.listdir(base)):
        cls_path = os.path.join(base, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        for sample in sorted(os.listdir(cls_path)):
            sample_path = os.path.join(cls_path, sample)
            if not os.path.isdir(sample_path):
                continue
            files = {os.path.splitext(f)[0]: os.path.join(sample_path, f)
                     for f in os.listdir(sample_path)
                     if f.lower().endswith(".png")}
            records.append((files, cls_folder))
    return records


def _get_image_files_scheme3(data_root: str):
    """Collect (lead_paths_dict, folder_name) pairs for Scheme 3 (6 limb + 6 precordial + long).

    Same source as Scheme 2 (cropped_leads/) but returns only the subset
    of lead files needed for Scheme 3: limb leads, precordial leads, and long lead.

    Args:
        data_root (str): Root data directory (e.g. 'data').

    Returns:
        list[tuple[dict, str]]: Each element is (lead_dict, class_folder_name),
            where lead_dict contains only the 13 relevant lead paths.
    """
    return _get_image_files_scheme2(data_root)


# ---------------------------------------------------------------------------
# Dataset Classes
# ---------------------------------------------------------------------------

def _build_transform(image_size: int = 224):
    """Build standard torchvision transform pipeline for ECG images.

    Resizes to image_size x image_size, converts to 3-channel tensor,
    and normalizes with ImageNet mean/std (compatible with pretrained models).

    Args:
        image_size (int): Target size (square). Defaults to 224.

    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ECGDatasetScheme1(Dataset):
    """PyTorch Dataset for Scheme 1: single clean ECG image per sample.

    Loads one image per sample from clean_ecg_signal/ and assigns label
    based on task type (2-class or 4-class).

    Args:
        records (list[tuple[str, str]]): Output of _get_image_files_scheme1().
        task (str): '2class' or '4class'.
        image_size (int): Target image size. Defaults to 224.
    """

    def __init__(self, records, task: str = "4class", image_size: int = 224):
        self.records = records
        self.label_map = LABEL_2CLASS if task == "2class" else LABEL_4CLASS
        self.transform = _build_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, cls_folder = self.records[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.label_map[cls_folder]
        return img, label


class ECGDatasetScheme2(Dataset):
    """PyTorch Dataset for Scheme 2: 12 short leads + 1 long lead concatenated.

    Loads all 13 lead images per sample, stacks them channel-wise into a
    single tensor of shape (39, H, W), and assigns label based on task type.

    Args:
        records (list[tuple[dict, str]]): Output of _get_image_files_scheme2().
        task (str): '2class' or '4class'.
        image_size (int): Target image size per lead. Defaults to 224.
    """

    ALL_PREFIXES = LIMB_LEAD_PREFIXES + PRECORDIAL_LEAD_PREFIXES + [LONG_LEAD_PREFIX]

    def __init__(self, records, task: str = "4class", image_size: int = 224):
        self.records = records
        self.label_map = LABEL_2CLASS if task == "2class" else LABEL_4CLASS
        self.transform = _build_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        files, cls_folder = self.records[idx]
        tensors = []
        # Load each lead in fixed order; stack along channel dimension
        for prefix in self.ALL_PREFIXES:
            path = files.get(prefix)
            img = Image.open(path).convert("RGB") if path else Image.new("RGB", (224, 224), 255)
            tensors.append(self.transform(img))
        img_tensor = torch.cat(tensors, dim=0)  # shape: (39, H, W)
        label = self.label_map[cls_folder]
        return img_tensor, label


class ECGDatasetScheme3(Dataset):
    """PyTorch Dataset for Scheme 3: multi-branch input (limb / precordial / long lead).

    Returns three separate tensors per sample (one per anatomical group) for use
    with the multi-branch MultiBranchResNet18 model:
      - tensor_limb       : (18, H, W) — 6 limb leads x 3 ch
      - tensor_precordial : (18, H, W) — 6 precordial leads x 3 ch
      - tensor_long       : (3,  H, W) — 1 long lead x 3 ch

    Args:
        records (list[tuple[dict, str]]): Output of _get_image_files_scheme3().
        task (str): '2class' or '4class'.
        image_size (int): Target image size per lead. Defaults to 224.
    """

    def __init__(self, records, task: str = "4class", image_size: int = 224):
        self.records = records
        self.label_map = LABEL_2CLASS if task == "2class" else LABEL_4CLASS
        self.transform = _build_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        files, cls_folder = self.records[idx]
        label = self.label_map[cls_folder]

        def _load_group(prefixes):
            tensors = []
            for prefix in prefixes:
                path = files.get(prefix)
                img = Image.open(path).convert("RGB") if path else Image.new("RGB", (224, 224), 255)
                tensors.append(self.transform(img))
            return torch.cat(tensors, dim=0)

        tensor_limb       = _load_group(LIMB_LEAD_PREFIXES)        # (18, H, W)
        tensor_precordial = _load_group(PRECORDIAL_LEAD_PREFIXES)   # (18, H, W)
        tensor_long       = _load_group([LONG_LEAD_PREFIX])         # (3,  H, W)

        return (tensor_limb, tensor_precordial, tensor_long), label


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_dataloaders(dataset, seed: int = 42, batch_size: int = 16,
                      val_split: float = 0.15, test_split: float = 0.15):
    """Split dataset into train/val/test and wrap each in a DataLoader.

    Performs stratified splitting based on labels to maintain class balance
    across splits. Uses pin_memory=True for GPU-optimized data loading.

    Args:
        dataset (Dataset): PyTorch Dataset instance.
        seed (int): Random seed for reproducibility. Defaults to 42.
        batch_size (int): Batch size for all loaders. Defaults to 16.
        val_split (float): Fraction of data for validation. Defaults to 0.15.
        test_split (float): Fraction of data for test. Defaults to 0.15.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_split, stratify=labels, random_state=seed
    )
    labels_trainval = [labels[i] for i in idx_trainval]
    val_fraction = val_split / (1 - test_split)

    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_fraction, stratify=labels_trainval, random_state=seed
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, idx_train),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, idx_val),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, idx_test),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_model(num_classes: int, in_channels: int = 3, pretrained: bool = True,
                freeze_layers: bool = True, backbone: str = "resnet18"):
    """Build a classification model adapted for ECG input (Schemes 1 & 2).

    Supports ResNet-18, MobileNetV2, and EfficientNetV2-S backbones.
    The first convolution layer is replaced when in_channels != 3 (e.g. multi-lead stacked input).
    The final FC layer is replaced to output num_classes logits.
    Optionally freezes early layers to reduce overfitting on small datasets.

    Args:
        num_classes (int): Number of output classes (2 or 4).
        in_channels (int): Number of input channels. 3 for Scheme 1,
            39 for Scheme 2 (13 leads x 3 channels). Defaults to 3.
        pretrained (bool): Use ImageNet pretrained weights. Defaults to True.
        freeze_layers (bool): If True, freeze early layers. Defaults to True.
        backbone (str): Backbone model: "resnet18", "mobilenet_v2", "efficientnet_v2_s".
            Defaults to "resnet18".

    Returns:
        torch.nn.Module: Modified model.
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_layers:
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False

    elif backbone == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(1280, num_classes)
        if freeze_layers:
            for param in model.features[0].parameters():
                param.requires_grad = False
            for param in model.features[1].parameters():
                param.requires_grad = False

    elif backbone == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(1280, num_classes)
        if freeze_layers:
            for param in model.features[0].parameters():
                param.requires_grad = False
            for param in model.features[1].parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


class _BranchBackbone(nn.Module):
    """Feature extractor branch with configurable backbone (ResNet-18, MobileNetV2, EfficientNetV2-S).

    Used internally by MultiBranchModel for Scheme 3. Exposes self.feature_dim
    to indicate output feature size for dynamic classifier sizing.

    Args:
        in_channels (int): Number of input channels.
        pretrained (bool): Load ImageNet weights. Defaults to True.
        freeze_layers (bool): Freeze early layers. Defaults to True.
        backbone (str): Backbone architecture: "resnet18", "mobilenet_v2", "efficientnet_v2_s".
            Defaults to "resnet18".
    """

    def __init__(self, in_channels: int, pretrained: bool = True,
                 freeze_layers: bool = True, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            if in_channels != 3:
                model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            if freeze_layers:
                for param in model.layer1.parameters():
                    param.requires_grad = False
                for param in model.layer2.parameters():
                    param.requires_grad = False
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool
            )
            self.feature_dim = 512

        elif backbone == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            if in_channels != 3:
                model.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
            if freeze_layers:
                for param in model.features[0].parameters():
                    param.requires_grad = False
                for param in model.features[1].parameters():
                    param.requires_grad = False
            self.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
            self.feature_dim = 1280

        elif backbone == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_v2_s(weights=weights)
            if in_channels != 3:
                model.features[0][0] = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False)
            if freeze_layers:
                for param in model.features[0].parameters():
                    param.requires_grad = False
                for param in model.features[1].parameters():
                    param.requires_grad = False
            self.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))
            self.feature_dim = 1280

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return torch.flatten(self.features(x), 1)


class _BranchResNet18(nn.Module):
    """Backward-compatibility alias for _BranchBackbone with ResNet-18."""

    def __init__(self, in_channels: int, pretrained: bool = True,
                 freeze_layers: bool = True):
        super().__init__()
        self._branch = _BranchBackbone(in_channels, pretrained, freeze_layers, backbone="resnet18")
        self.features = self._branch.features
        self.feature_dim = self._branch.feature_dim

    def forward(self, x):
        return self._branch(x)


class MultiBranchModel(nn.Module):
    """Multi-branch model for anatomically grouped ECG leads (Scheme 3).

    Three separate branches process limb leads (18ch), precordial leads (18ch),
    and long lead (3ch) independently. Feature vectors are concatenated and
    classified. Supports ResNet-18, MobileNetV2, and EfficientNetV2-S backbones.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet weights. Defaults to True.
        freeze_layers (bool): Freeze early layers. Defaults to True.
        backbone (str): Backbone architecture. Defaults to "resnet18".
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_layers: bool = True, backbone: str = "resnet18"):
        super().__init__()
        self.branch_limb       = _BranchBackbone(18, pretrained, freeze_layers, backbone)
        self.branch_precordial = _BranchBackbone(18, pretrained, freeze_layers, backbone)
        self.branch_long       = _BranchBackbone(3,  pretrained, freeze_layers, backbone)
        feature_dim = self.branch_limb.feature_dim
        self.classifier = nn.Linear(feature_dim * 3, num_classes)

    def forward(self, inputs):
        t_limb, t_precordial, t_long = inputs
        feat = torch.cat([
            self.branch_limb(t_limb),
            self.branch_precordial(t_precordial),
            self.branch_long(t_long),
        ], dim=1)
        return self.classifier(feat)


class MultiBranchResNet18(MultiBranchModel):
    """Backward-compatibility alias for MultiBranchModel with ResNet-18 backbone."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 freeze_layers: bool = True):
        super().__init__(num_classes, pretrained, freeze_layers, backbone="resnet18")


def build_model_multibranch(num_classes: int, pretrained: bool = True,
                             freeze_layers: bool = True, backbone: str = "resnet18"):
    """Construct a MultiBranchModel for Scheme 3.

    Three separate branches process limb leads (18ch), precordial leads (18ch),
    and long lead (3ch) independently. Feature vectors are concatenated and
    classified. Supports ResNet-18, MobileNetV2, and EfficientNetV2-S backbones.

    Args:
        num_classes (int): Number of output classes (2 or 4).
        pretrained (bool): Use ImageNet pretrained weights. Defaults to True.
        freeze_layers (bool): Freeze early layers. Defaults to True.
        backbone (str): Backbone architecture. Defaults to "resnet18".

    Returns:
        MultiBranchModel: Multi-branch model instance.
    """
    return MultiBranchModel(num_classes, pretrained, freeze_layers, backbone)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full training epoch over the provided DataLoader.

    Supports both single-tensor input (Schemes 1 & 2) and multi-branch tuple
    input (Scheme 3). Detects input type automatically from batch structure.

    Args:
        model (nn.Module): Model to train.
        loader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computation on.

    Returns:
        tuple[float, float]: (average_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_inputs, labels in loader:
        labels = labels.to(device)

        if isinstance(batch_inputs, (list, tuple)):
            inputs = tuple(t.to(device) for t in batch_inputs)
            batch_size = labels.size(0)
        else:
            inputs = batch_inputs.to(device)
            batch_size = inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader without gradient updates.

    Supports both single-tensor input (Schemes 1 & 2) and multi-branch tuple
    input (Scheme 3). Detects input type automatically from batch structure.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Validation or test DataLoader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computation on.

    Returns:
        tuple[float, float, list, list, np.ndarray]:
            (average_loss, accuracy, all_preds, all_labels, all_probs)
            where all_probs has shape (N, num_classes).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_inputs, labels in loader:
            labels = labels.to(device)

            if isinstance(batch_inputs, (list, tuple)):
                inputs = tuple(t.to(device) for t in batch_inputs)
                batch_size = labels.size(0)
            else:
                inputs = batch_inputs.to(device)
                batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, num_epochs: int,
                learning_rate: float, device, run_name: str,
                scheme: str, task: str, model_name: str = "ResNet-18",
                class_names: list = None, output_dir: str = "outputs/experiments"):
    """Train a model with comprehensive MLflow logging and artifact saving.

    Logs hyperparameters (including model name, GPU info, date/time, training time),
    per-epoch metrics, training/validation plots, confusion matrix, and final model.
    Uses CrossEntropyLoss and Adam optimizer. Saves the best model checkpoint
    based on validation accuracy to both MLflow (remote) and local output folder.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        device (torch.device): Device to run computation on.
        run_name (str): MLflow run name for this experiment.
        scheme (str): Scheme identifier ('scheme1', 'scheme2', 'scheme3').
        task (str): Task identifier ('2class' or '4class').
        model_name (str): Human-readable model name. Defaults to "ResNet-18".
        class_names (list): Class names for confusion matrix. Defaults to None.
        output_dir (str): Local directory to save models and artifacts. Defaults to "outputs/experiments".

    Returns:
        tuple[nn.Module, dict, str, str]: (best_model, history, run_output_dir, mlflow_run_id)
            where history contains lists of train_loss, train_acc, val_loss, val_acc per epoch,
            run_output_dir is the timestamped output directory for artifacts, and mlflow_run_id
            is the MLflow run ID to pass to evaluate_and_log() as parent_run_id.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -float('inf')
    best_model_state = None
    best_val_preds = []
    best_val_labels = []

    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    # Get date and time in format DD:MM-HH:MM
    now = datetime.now()
    date_time = f"{now.day:02d}:{now.month:02d}-{now.hour:02d}:{now.minute:02d}"

    # Create local output directory with timestamp to avoid overwriting
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Record start time for training duration
    start_time = time.time()

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow_run_id = active_run.info.run_id
        mlflow.log_params({
            "scheme": scheme,
            "task": task,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "model_name": model_name,
            "gpu_name": gpu_name,
            "date_time": date_time,
        })

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(f"[{run_name}] Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Save best model checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                # Store val preds/labels for best model's confusion matrix
                best_val_preds = val_preds
                best_val_labels = val_labels

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Save model locally
        local_model_path = os.path.join(run_output_dir, "best_model.pth")
        torch.save(model.state_dict(), local_model_path)
        print(f"Model saved locally to: {local_model_path}")

        # Log training time in HH:MM:SS format
        elapsed = time.time() - start_time
        training_time = str(timedelta(seconds=int(elapsed)))
        mlflow.log_param("training_time", training_time)

        # Log training history plot
        tmp_hist_path = os.path.join(tempfile.gettempdir(), f"hist_{run_name}.png")
        _plot_and_save_training_history(history, tmp_hist_path)
        mlflow.log_artifact(tmp_hist_path, artifact_path="plots")
        local_hist_path = os.path.join(run_output_dir, "training_history.png")
        shutil.copy2(tmp_hist_path, local_hist_path)
        os.remove(tmp_hist_path)
        print(f"Training history saved to: {local_hist_path}")

        # Log validation confusion matrix
        if class_names is not None:
            tmp_cm_path = os.path.join(tempfile.gettempdir(), f"cm_val_{run_name}.png")
            _plot_and_save_confusion_matrix(
                best_val_labels, best_val_preds, class_names, tmp_cm_path
            )
            mlflow.log_artifact(tmp_cm_path, artifact_path="plots")
            local_cm_path = os.path.join(run_output_dir, "confusion_matrix_val.png")
            shutil.copy2(tmp_cm_path, local_cm_path)
            os.remove(tmp_cm_path)
            print(f"Confusion matrix saved to: {local_cm_path}")

    return model, history, run_output_dir, mlflow_run_id


def _plot_and_save_training_history(history: dict, filepath: str):
    """Plot training history (loss and accuracy) and save as PNG.

    Args:
        history (dict): History dict with keys train_loss, val_loss, train_acc, val_acc.
        filepath (str): Path to save the PNG file.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_and_save_confusion_matrix(labels: list, preds: list, class_names: list,
                                    filepath: str):
    """Plot confusion matrix and save as PNG.

    Args:
        labels (list): True labels.
        preds (list): Predicted labels.
        class_names (list): Human-readable class names.
        filepath (str): Path to save the PNG file.
    """
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix (Validation Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()


def evaluate_and_log(model, test_loader, device, class_names: list,
                     scheme: str, task: str, run_name: str = None,
                     output_dir: str = None, parent_run_id: str = None):
    """Evaluate model on test set, log metrics to MLflow, and print report.

    Computes accuracy, macro F1, ROC-AUC, and classification report on the test set.
    Logs test metrics and confusion matrix artifact to MLflow as a true nested run
    under the parent training run when parent_run_id is provided.

    Supports both single-tensor (Schemes 1 & 2) and multi-branch tuple (Scheme 3)
    DataLoaders transparently via the shared evaluate() function.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test DataLoader.
        device (torch.device): Device to run computation on.
        class_names (list[str]): Human-readable class names.
        scheme (str): Scheme identifier for display.
        task (str): Task identifier for display.
        run_name (str): MLflow run name for nested eval run. If None, no MLflow logging.
        output_dir (str): Local directory to save test confusion matrix.
        parent_run_id (str): MLflow run_id from train_model() 4th return value.
            When provided, reopens the parent run so this eval run is a true nested child,
            linking train and test metrics in the MLflow UI. Defaults to None.

    Returns:
        dict: Dictionary with keys 'accuracy', 'f1', 'roc_auc', 'preds', 'labels'.
    """
    criterion = nn.CrossEntropyLoss()
    _, acc, preds, labels, all_probs = evaluate(model, test_loader, criterion, device)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Compute ROC-AUC (binary: use positive class prob; multi-class: OvR macro)
    num_classes = all_probs.shape[1]
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(labels, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    print(f"\n=== Test Results | {scheme} | {task} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # Log test metrics directly into the parent training run (no separate child run)
    if parent_run_id is not None:
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_metrics({
                "test_accuracy": acc,
                "test_f1_macro": f1,
                "test_roc_auc":  roc_auc,
            })

            tmp_cm_path = os.path.join(tempfile.gettempdir(), f"cm_test_{scheme}_{task}.png")
            _plot_and_save_confusion_matrix(labels, preds, class_names, tmp_cm_path)
            mlflow.log_artifact(tmp_cm_path, artifact_path="plots")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                local_cm_path = os.path.join(output_dir, "confusion_matrix_test.png")
                shutil.copy2(tmp_cm_path, local_cm_path)
                os.remove(tmp_cm_path)
                print(f"Test confusion matrix saved to: {local_cm_path}")

    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc, "preds": preds, "labels": labels}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_history(histories: dict, title_prefix: str = ""):
    """Plot training and validation loss/accuracy curves for multiple runs.

    Creates a 2-row grid: top row for loss curves, bottom row for accuracy
    curves. Each column corresponds to one training run.

    Args:
        histories (dict): Dictionary mapping run_name -> history dict
            (each history has keys: train_loss, val_loss, train_acc, val_acc).
        title_prefix (str): Optional prefix added to the figure title.
    """
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))

    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (run_name, history) in enumerate(histories.items()):
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0, col].plot(epochs, history["train_loss"], label="Train")
        axes[0, col].plot(epochs, history["val_loss"], label="Val")
        axes[0, col].set_title(f"{run_name}\nLoss")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Loss")
        axes[0, col].legend()

        axes[1, col].plot(epochs, history["train_acc"], label="Train")
        axes[1, col].plot(epochs, history["val_acc"], label="Val")
        axes[1, col].set_title(f"{run_name}\nAccuracy")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Accuracy")
        axes[1, col].legend()

    fig.suptitle(f"{title_prefix}Training History", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results: dict, class_names: list):
    """Plot confusion matrices for multiple trained models side-by-side.

    Args:
        results (dict): Dictionary mapping run_name -> result dict
            (each result has keys: preds, labels).
        class_names (list[str]): Human-readable class names.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (run_name, result) in zip(axes, results.items()):
        cm = confusion_matrix(result["labels"], result["preds"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(run_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_yticklabels(class_names)

        # Annotate each cell with count
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.colorbar(im, ax=ax)

    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_scheme_comparison(all_results: dict):
    """Plot bar chart comparing accuracy and F1 score across all schemes and tasks.

    Creates a grouped bar chart where each group is a (scheme, task) combination.
    Two bars per group: Accuracy and Macro F1.

    Args:
        all_results (dict): Nested dict mapping run_name -> result dict
            (each result must have keys 'accuracy' and 'f1').
    """
    run_names = list(all_results.keys())
    accuracies = [all_results[r]["accuracy"] for r in run_names]
    f1_scores = [all_results[r]["f1"] for r in run_names]

    x = np.arange(len(run_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(run_names) * 2), 6))

    bars_acc = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="steelblue")
    bars_f1 = ax.bar(x + width / 2, f1_scores, width, label="Macro F1", color="darkorange")

    ax.set_title("Perbandingan Performa Antar Skema Preprocessing & Task", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=20, ha="right")
    ax.legend()

    # Annotate bar values
    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_sample_predictions(model, dataset, class_names: list,
                            device, num_samples: int = 8):
    """Visualize model predictions on random samples from a dataset.

    Displays a grid of ECG images with their true and predicted labels.
    For multi-lead datasets (Scheme 2/3), only the first channel is shown.

    Args:
        model (nn.Module): Trained model in eval mode.
        dataset (Dataset): Dataset to sample from.
        class_names (list[str]): Human-readable class names.
        device (torch.device): Device to run computation on.
        num_samples (int): Number of samples to display. Defaults to 8.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(3 * (num_samples // 2), 6))
    axes = axes.flatten()

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img_tensor, true_label = dataset[idx]
            output = model(img_tensor.unsqueeze(0).to(device))
            pred_label = output.argmax(dim=1).item()

            # Show only first 3 channels (first lead or grayscale)
            img_show = img_tensor[:3].permute(1, 2, 0).cpu().numpy()
            img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)

            color = "green" if pred_label == true_label else "red"
            ax.imshow(img_show)
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                color=color, fontsize=8
            )
            ax.axis("off")

    plt.suptitle("Sample Predictions", fontsize=13)
    plt.tight_layout()
    plt.show()
