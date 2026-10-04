"""Batch bright-field segmentation: TIFF folder -> per-image JSON annotations."""
import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .annotations import build_annotation, remove_small_objects

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_images(input_dir: Path, skip_dir_markers=("_Tsf", "_Cell")) -> List[Path]:
    """Recursively list .tif/.tiff files, skipping folders whose name contains a marker."""
    found = []
    for path in sorted(Path(input_dir).rglob("*")):
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        rel_dirs = path.relative_to(input_dir).parts[:-1]
        if any(m in d for d in rel_dirs for m in skip_dir_markers):
            continue
        found.append(path)
    return found


def load_image(path: Path) -> np.ndarray:
    """Read an image as a normalised float32 CHW array (ImageNet mean/std)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _batches(items: List[Path], size: int) -> Iterator[List[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def segment_folder(
    input_dir,
    output_dir,
    weights,
    threshold: float = 0.5,
    min_area: int = 80,
    batch_size: int = 8,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Segment every image under ``input_dir`` and write one JSON per image.

    Output mirrors the input folder structure. Images whose JSON already exists are
    skipped unless ``overwrite`` is set, so interrupted runs can be resumed.
    Returns the number of images processed.
    """
    import torch  # imported lazily so the tracking step does not need PyTorch

    from .model import build_model

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model()
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    todo: List[Tuple[Path, Path]] = []
    for img in find_images(input_dir):
        out = output_dir / img.relative_to(input_dir).with_suffix(".json")
        if overwrite or not out.exists():
            todo.append((img, out))

    done = 0
    with torch.inference_mode():
        for batch in _batches(todo, batch_size):
            tensor = torch.from_numpy(np.stack([load_image(p) for p, _ in batch])).to(device)
            probs = torch.sigmoid(model(tensor))[:, 0].cpu().numpy()
            for (img_path, out_path), prob in zip(batch, probs):
                mask = remove_small_objects((prob > threshold).astype(np.uint8), min_area)
                annotation = build_annotation(str(img_path), mask)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(annotation))
                done += 1
            print(f"{done}/{len(todo)} images", flush=True)
    return done
