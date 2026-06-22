from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.lead_parser import (
    ALL_13_LEADS,
    GRID_13_LEADS,
    LIMB_LEADS,
    LONG_LEAD,
    PRECORDIAL_LEADS,
    SHORT_12_LEADS,
    lead_column_name,
)
from src.data.transforms import build_image_transform


def _load_image(path: str, fallback_size: int = 224) -> Image.Image:
    if path and Path(path).exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (fallback_size, fallback_size), 255)


class ECGManifestDataset(Dataset):
    def __init__(self, rows: list[dict], class_names: list[str], input_scheme: str, image_size: int = 224):
        self.rows = rows
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.input_scheme = input_scheme
        self.image_size = image_size
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _image_tensor(self, path: str) -> torch.Tensor:
        return self.transform(_load_image(path, self.image_size))

    def _stack_leads(self, row: dict, leads: list[str]) -> torch.Tensor:
        tensors = [self._image_tensor(row.get(lead_column_name(prefix), "")) for prefix in leads]
        return torch.cat(tensors, dim=0)

    def _sequence_leads(self, row: dict, leads: list[str]) -> torch.Tensor:
        tensors = [self._image_tensor(row.get(lead_column_name(prefix), "")) for prefix in leads]
        return torch.stack(tensors, dim=0)

    def __getitem__(self, index: int):
        row = self.rows[index]
        scheme = self.input_scheme

        if scheme == "single_clean_image":
            inputs = self._image_tensor(row.get("clean_image_path", ""))
        elif scheme == "single_raw_image":
            inputs = self._image_tensor(row.get("raw_image_path", "") or row.get("clean_image_path", ""))
        elif scheme == "single_long_lead_ii":
            inputs = self._image_tensor(row.get(lead_column_name(LONG_LEAD), ""))
        elif scheme == "single_12_lead":
            inputs = self._stack_leads(row, SHORT_12_LEADS)
        elif scheme == "stacked_12lead_longlead":
            inputs = self._stack_leads(row, GRID_13_LEADS)
        elif scheme == "stacked_6lead_6lead_longlead":
            inputs = self._stack_leads(row, ALL_13_LEADS)
        elif scheme == "stacked_13lead_individual":
            inputs = self._sequence_leads(row, GRID_13_LEADS)
        elif scheme == "multibranch_12lead_longlead":
            inputs = (self._stack_leads(row, SHORT_12_LEADS), self._stack_leads(row, [LONG_LEAD]))
        elif scheme == "multibranch_6lead_6lead_longlead":
            inputs = (
                self._stack_leads(row, LIMB_LEADS),
                self._stack_leads(row, PRECORDIAL_LEADS),
                self._stack_leads(row, [LONG_LEAD]),
            )
        elif scheme == "multibranch_13lead_individual":
            inputs = tuple(self._image_tensor(row.get(lead_column_name(prefix), "")) for prefix in ALL_13_LEADS)
        else:
            raise ValueError(f"Unsupported input_scheme: {scheme}")

        label = self.class_to_idx[row["target_label"]]
        return inputs, label, row["sample_id"]
