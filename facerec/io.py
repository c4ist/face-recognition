from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

#lucky guess on how documentation worked + a couple other examples i saw on github taught me a lot
def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def list_image_files(input_path: Path | str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if is_image_file(path) else []
    if not path.exists() or not path.is_dir():
        return []
    return sorted([item for item in path.rglob("*") if item.is_file() and is_image_file(item)])


def list_identity_dirs(known_dir: Path | str) -> List[Path]:
    directory = Path(known_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted([item for item in directory.iterdir() if item.is_dir()], key=lambda p: p.name.lower())


def load_bgr_image(image_path: Path | str) -> np.ndarray:
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_bgr_image(image_path: Path | str, image: np.ndarray) -> None:
    path = Path(image_path)
    ensure_directory(path.parent)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Failed to write image: {path}")


def draw_face_annotation(image: np.ndarray, bbox: tuple[float, float, float, float], label: str, is_unknown: bool) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    color = (0, 0, 255) if is_unknown else (0, 200, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text = label.strip() if label.strip() else "unknown"
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(
        image,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
