from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import build_image_transform
from src.models.build_model import build_model, infer_input_shape
from src.utils.io import write_csv

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def run_inference(config: dict, checkpoint_path: str | Path, output_path: str | Path, device):
    input_scheme = config["data"]["input_scheme"]
    kind, channels = infer_input_shape(input_scheme)
    if kind != "single" or channels != 3:
        raise ValueError("inference_only currently supports 3-channel single-image schemes.")

    paths = []
    inference_file = config.get("run", {}).get("inference_file")
    inference_dir = config.get("run", {}).get("inference_dir")
    if inference_file:
        paths.append(Path(inference_file))
    if inference_dir:
        paths.extend(sorted(p for p in Path(inference_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS))

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_image_transform(config["data"]["image_size"])
    class_names = config["data"]["class_names"]
    rows = []
    with torch.no_grad():
        for path in paths:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            row = {
                "input_path": str(path),
                "predicted_label": class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
            }
            for idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probs[idx])
            rows.append(row)
    write_csv(rows, output_path)
    return rows
