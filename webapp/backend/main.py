#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


APP_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = APP_ROOT / "tools"
FRONTEND_DIR = APP_ROOT / "webapp" / "frontend"
TMP_DIR = APP_ROOT / "webapp" / "tmp"

YOLO_TO_COCO_SCRIPT = TOOLS_DIR / "yolo_to_coco.py"
EXPORT_SCRIPT = TOOLS_DIR / "export_dataset.py"

ALLOWED_FORMATS = {"yolo", "coco"}
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_PREVIEW_FORMATS = {"source_yolo", *ALLOWED_FORMATS}
ALLOWED_PREVIEW_MODES = {"translated", "direct"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PALETTE = [
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#e9c46a",
    "#6d597a",
    "#f4a261",
    "#1d3557",
    "#8ecae6",
]

MODEL_CATALOG: list[dict[str, str]] = [
    {
        "id": "yolov5",
        "name": "YOLOv5",
        "required_format": "yolo",
        "description": "YOLO txt labels for YOLOv5 training pipelines.",
    },
    {
        "id": "yolov6",
        "name": "YOLOv6",
        "required_format": "yolo",
        "description": "YOLO txt labels for YOLOv6 training pipelines.",
    },
    {
        "id": "yolov7",
        "name": "YOLOv7",
        "required_format": "yolo",
        "description": "YOLO txt labels for YOLOv7 training pipelines.",
    },
    {
        "id": "yolov8",
        "name": "YOLOv8",
        "required_format": "yolo",
        "description": "YOLO txt labels for YOLOv8 training pipelines.",
    },
    {
        "id": "yolov11",
        "name": "YOLOv11",
        "required_format": "yolo",
        "description": "YOLO txt labels for YOLOv11 training pipelines.",
    },
    {
        "id": "darknet_yolo",
        "name": "Darknet YOLO",
        "required_format": "yolo",
        "description": "YOLO txt labels for Darknet-style training pipelines.",
    },
    {
        "id": "kerascv_yolo",
        "name": "KerasCV YOLO",
        "required_format": "yolo",
        "description": "YOLO-format labels for KerasCV YOLO workflows.",
    },
    {
        "id": "detectron2",
        "name": "Detectron2 (Faster/Mask R-CNN)",
        "required_format": "coco",
        "description": "COCO JSON used by Detectron2 data loaders.",
    },
    {
        "id": "mmdetection",
        "name": "MMDetection",
        "required_format": "coco",
        "description": "COCO format is the common input.",
    },
]
MODEL_INDEX: dict[str, dict[str, str]] = {item["id"]: item for item in MODEL_CATALOG}
ALLOWED_MODELS = set(MODEL_INDEX.keys())


class ExportRequest(BaseModel):
    dataset_root: str = Field(..., description="Path to source YOLO dataset root")
    models: list[str] = Field(default_factory=lambda: ["yolov8"])
    formats: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "test"])
    include_images: bool = True


class PreviewRequest(BaseModel):
    dataset_root: str = Field(..., description="Path to source YOLO dataset root")
    split: str = Field(default="train")
    image_name: str = Field(..., description="Image filename inside split/images")
    preview_format: str = Field(default="source_yolo")
    preview_mode: str = Field(default="translated")
    line_width: int = Field(default=2, ge=1, le=8)
    show_labels: bool = True


app = FastAPI(title="SCN Dataset Export API", version="1.0.0")


def is_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data.yaml").exists()
        and (path / "train" / "images").exists()
        and (path / "train" / "labels").exists()
    )


def dataset_score(path: Path) -> int:
    score = 0
    lowered = path.name.lower()
    if "scn_cyst_counting" in lowered:
        score += 3
    if "yolo" in lowered:
        score += 2
    return score


def discover_dataset_roots(search_root: Path) -> list[str]:
    if not search_root.exists() or not search_root.is_dir():
        return []

    candidates: list[tuple[int, str, Path]] = []

    if is_dataset_root(search_root):
        candidates.append((dataset_score(search_root), search_root.name, search_root))

    for child in search_root.iterdir():
        if not child.is_dir():
            continue
        if is_dataset_root(child):
            candidates.append((dataset_score(child), child.name, child))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [str(item[2].resolve()) for item in candidates]


def guess_dataset_root() -> str:
    discovered = discover_dataset_roots(APP_ROOT.parent)
    return discovered[0] if discovered else ""


def ensure_scripts_exist() -> None:
    if not YOLO_TO_COCO_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {YOLO_TO_COCO_SCRIPT}")
    if not EXPORT_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {EXPORT_SCRIPT}")


def validate_selection(selected: Sequence[str], allowed: set[str], label: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in selected:
        cleaned = value.strip().lower()
        if cleaned not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {label}: {value}. Allowed: {', '.join(sorted(allowed))}",
            )
        if cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    if not ordered:
        raise HTTPException(status_code=400, detail=f"At least one {label} is required.")
    return ordered


def resolve_formats(models: Sequence[str], formats: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for model_id in models:
        mapped = MODEL_INDEX[model_id]["required_format"]
        if mapped not in seen:
            ordered.append(mapped)
            seen.add(mapped)

    for fmt in formats:
        if fmt not in seen:
            ordered.append(fmt)
            seen.add(fmt)

    if not ordered:
        raise HTTPException(
            status_code=400,
            detail="Select at least one model or one format.",
        )
    return ordered


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "valid":
        return "val"
    return normalized


def parse_names_from_yaml(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        return []

    text = yaml_path.read_text(encoding="utf-8")
    list_match = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, flags=re.M)
    if list_match:
        raw = list_match.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        names: list[str] = []
        for part in parts:
            if (part.startswith("'") and part.endswith("'")) or (
                part.startswith('"') and part.endswith('"')
            ):
                part = part[1:-1]
            names.append(part)
        if names:
            return names

    block_match = re.search(r"^\s*names\s*:\s*$", text, flags=re.M)
    if not block_match:
        return []

    names_dict: dict[int, str] = {}
    lines = text[block_match.end() :].splitlines()
    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^[^\s]", line):
            break
        entry = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if not entry:
            continue
        idx = int(entry.group(1))
        value = entry.group(2).strip()
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]
        names_dict[idx] = value

    return [names_dict[i] for i in sorted(names_dict)]


def validate_dataset_root_or_400(dataset_root: Path) -> None:
    if not dataset_root.exists():
        raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {dataset_root}")
    if not (dataset_root / "data.yaml").exists():
        raise HTTPException(
            status_code=400,
            detail=f"Expected file not found: {dataset_root / 'data.yaml'}",
        )


def validate_path_exists_or_400(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")


def list_images(dataset_root: Path, split: str) -> list[str]:
    candidate_dirs = [
        dataset_root / split / "images",
        dataset_root / split,
        dataset_root / "images" / split,
        dataset_root / "images",
    ]
    for images_dir in candidate_dirs:
        if not images_dir.exists() or not images_dir.is_dir():
            continue
        files = [
            path.name
            for path in sorted(images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if files:
            return files
    return []


def resolve_image_path(dataset_root: Path, split: str, image_name: str) -> Path | None:
    candidate_paths = [
        dataset_root / split / "images" / image_name,
        dataset_root / split / image_name,
        dataset_root / "images" / split / image_name,
        dataset_root / "images" / image_name,
    ]
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def class_label(class_id: int, class_names: Sequence[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def parse_yolo_label_file(label_path: Path, width: int, height: int, class_names: Sequence[str]) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    result: list[dict[str, Any]] = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        row = row.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            bw = float(parts[3]) * width
            bh = float(parts[4]) * height
        except ValueError:
            continue
        x1 = cx - (bw / 2.0)
        y1 = cy - (bh / 2.0)
        x2 = x1 + bw
        y2 = y1 + bh
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width, height)
        result.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "label": class_label(class_id, class_names),
            }
        )
    return result


def parse_voc_file(xml_path: Path, class_names: Sequence[str]) -> list[dict[str, Any]]:
    if not xml_path.exists():
        return []
    result: list[dict[str, Any]] = []
    class_index = {name: idx for idx, name in enumerate(class_names)}
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            x1 = float(bbox.findtext("xmin", "0"))
            y1 = float(bbox.findtext("ymin", "0"))
            x2 = float(bbox.findtext("xmax", "0"))
            y2 = float(bbox.findtext("ymax", "0"))
        except ValueError:
            continue
        class_id = class_index.get(name, -1)
        result.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "label": name or class_label(class_id, class_names),
            }
        )
    return result


def parse_coco_file(
    coco_path: Path,
    image_rel_path: str,
    image_name: str,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    if not coco_path.exists():
        return []
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_id = None
    for image in images:
        file_name = str(image.get("file_name", ""))
        if file_name == image_rel_path or Path(file_name).name == image_name:
            image_id = int(image.get("id"))
            break
    if image_id is None:
        return []

    category_to_name = {int(c.get("id")): str(c.get("name", "")) for c in categories}
    class_index = {name: idx for idx, name in enumerate(class_names)}
    result: list[dict[str, Any]] = []

    for ann in annotations:
        if int(ann.get("image_id", -1)) != image_id:
            continue
        bbox = ann.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x = float(bbox[0])
            y = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
        except (ValueError, TypeError):
            continue
        category_id = int(ann.get("category_id", -1))
        name = category_to_name.get(category_id, str(category_id))
        class_id = class_index.get(name, category_id - 1)
        result.append(
            {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "class_id": class_id,
                "label": name,
            }
        )
    return result


def parse_csv_file(csv_path: Path, image_rel_path: str, image_name: str) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    result: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_image = str(row.get("image_file", ""))
            if row_image != image_rel_path and Path(row_image).name != image_name:
                continue
            try:
                class_id = int(float(row.get("class_id", "-1")))
                x1 = float(row.get("x1", "0"))
                y1 = float(row.get("y1", "0"))
                x2 = float(row.get("x2", "0"))
                y2 = float(row.get("y2", "0"))
            except ValueError:
                continue
            label = str(row.get("class_name") or class_id)
            result.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": class_id,
                    "label": label,
                }
            )
    return result


def draw_preview_image(
    image_path: Path,
    annotations: Sequence[dict[str, Any]],
    line_width: int,
    show_labels: bool,
) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Preview requires Pillow. Install with: pip install pillow",
        ) from exc

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for ann in annotations:
        class_id = int(ann.get("class_id", -1))
        color = PALETTE[class_id % len(PALETTE)] if class_id >= 0 else "#ff006e"
        x1 = float(ann.get("x1", 0.0))
        y1 = float(ann.get("y1", 0.0))
        x2 = float(ann.get("x2", 0.0))
        y2 = float(ann.get("y2", 0.0))
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        if show_labels:
            label = str(ann.get("label", class_id))
            try:
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                tw, th = right - left, bottom - top
            except Exception:
                tw, th = draw.textsize(label, font=font)
            tx = x1
            ty = max(0.0, y1 - th - 4)
            draw.rectangle([(tx, ty), (tx + tw + 6, ty + th + 4)], fill=color)
            draw.text((tx + 3, ty + 2), label, fill="#ffffff", font=font)

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def run_command(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "Unknown command error"
        cmd = " ".join(args)
        raise RuntimeError(f"Command failed: {cmd}\n{detail}")
    return completed.stdout


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def model_export_tag(model_id: str) -> str:
    model = MODEL_INDEX.get(model_id)
    base = model["name"] if model else model_id
    tag = slugify(base)
    for suffix in [
        "_yolo",
        "_models",
        "_family",
        "_workflow",
        "_api",
        "_detection",
    ]:
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
            break
    return tag or "dataset"


def build_export_base_name(selected_models: Sequence[str], formats: Sequence[str]) -> str:
    if len(selected_models) > 1 or len(formats) > 1:
        return "scn_multi_dataset"
    if selected_models:
        return f"scn_{model_export_tag(selected_models[0])}_dataset"
    if formats:
        return f"scn_{slugify(formats[0])}_dataset"
    return "scn_dataset"


def zip_export(export_dir: Path, zip_path: Path, root_dir_name: str) -> None:
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(export_dir.rglob("*")):
            if file_path.is_file():
                arcname = Path(root_dir_name) / file_path.relative_to(export_dir)
                zf.write(file_path, arcname=str(arcname))


def safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index_html = FRONTEND_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=500, detail=f"Missing UI file: {index_html}")
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/defaults")
def defaults() -> dict[str, object]:
    discovered = discover_dataset_roots(APP_ROOT.parent)
    default_root = discovered[0] if discovered else ""
    return {
        "dataset_root": default_root,
        "dataset_candidates": discovered,
        "models": ["yolov8"],
        "formats": [],
        "splits": ["train", "test"],
        "include_images": True,
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {
        "models": MODEL_CATALOG,
        "formats": sorted(ALLOWED_FORMATS),
    }


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    discovered = discover_dataset_roots(APP_ROOT.parent)
    return {
        "datasets": discovered,
        "default_dataset_root": discovered[0] if discovered else "",
    }


@app.get("/api/images")
def images(
    dataset_root: str = Query(..., description="Path to source YOLO dataset root"),
    split: str = Query("train", description="Split name"),
) -> dict[str, Any]:
    normalized_split = normalize_split(split)
    if normalized_split not in ALLOWED_SPLITS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid split: {split}. Allowed: {', '.join(sorted(ALLOWED_SPLITS))}",
        )

    root = Path(dataset_root).expanduser().resolve()
    validate_path_exists_or_400(root)

    return {
        "split": normalized_split,
        "images": list_images(root, normalized_split),
    }


@app.post("/api/preview")
def preview_annotation(payload: PreviewRequest) -> Response:
    normalized_split = normalize_split(payload.split)
    if normalized_split not in ALLOWED_SPLITS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid split: {payload.split}. Allowed: {', '.join(sorted(ALLOWED_SPLITS))}",
        )

    preview_format = payload.preview_format.strip().lower()
    if preview_format not in ALLOWED_PREVIEW_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid preview_format: {payload.preview_format}. "
                f"Allowed: {', '.join(sorted(ALLOWED_PREVIEW_FORMATS))}"
            ),
        )

    preview_mode = payload.preview_mode.strip().lower()
    if preview_mode not in ALLOWED_PREVIEW_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid preview_mode: {payload.preview_mode}. "
                f"Allowed: {', '.join(sorted(ALLOWED_PREVIEW_MODES))}"
            ),
        )

    image_name = payload.image_name.strip()
    if not image_name:
        raise HTTPException(status_code=400, detail="image_name is required.")
    if Path(image_name).name != image_name:
        raise HTTPException(status_code=400, detail="image_name must be a file name, not a path.")

    dataset_root = Path(payload.dataset_root).expanduser().resolve()
    if preview_mode == "translated":
        validate_dataset_root_or_400(dataset_root)
    else:
        validate_path_exists_or_400(dataset_root)

    image_path = dataset_root / normalized_split / "images" / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Preview requires Pillow. Install with: pip install pillow",
        ) from exc

    with Image.open(image_path) as img:
        width, height = img.size

    class_names = parse_names_from_yaml(dataset_root / "data.yaml")
    image_stem = Path(image_name).stem
    image_rel_path = f"{normalized_split}/images/{image_name}"

    annotations: list[dict[str, Any]] = []
    if preview_mode == "direct":
        if preview_format in {"source_yolo", "yolo"}:
            label_path = dataset_root / normalized_split / "labels" / f"{image_stem}.txt"
            annotations = parse_yolo_label_file(label_path, width, height, class_names)
        elif preview_format == "voc":
            xml_path = dataset_root / normalized_split / "annotations" / f"{image_stem}.xml"
            annotations = parse_voc_file(xml_path, class_names)
        elif preview_format == "coco":
            coco_path = dataset_root / "annotations.coco.json"
            annotations = parse_coco_file(coco_path, image_rel_path, image_name, class_names)
        elif preview_format == "csv":
            csv_path = dataset_root / "annotations.csv"
            annotations = parse_csv_file(csv_path, image_rel_path, image_name)
    elif preview_format == "source_yolo":
        source_label = dataset_root / normalized_split / "labels" / f"{image_stem}.txt"
        annotations = parse_yolo_label_file(source_label, width, height, class_names)
    else:
        ensure_scripts_exist()
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="preview_job_", dir=str(TMP_DIR)))
        canonical_dir = job_dir / "canonical"
        exports_dir = job_dir / "exports"
        canonical_coco = canonical_dir / "annotations.coco.json"
        canonical_raw = canonical_dir / "annotations.raw.jsonl"

        try:
            step1 = [
                sys.executable,
                str(YOLO_TO_COCO_SCRIPT),
                "--dataset-root",
                str(dataset_root),
                "--splits",
                normalized_split,
                "--output",
                str(canonical_coco),
                "--raw-output",
                str(canonical_raw),
                "--strict",
            ]
            run_command(step1)

            step2 = [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--dataset-root",
                str(dataset_root),
                "--raw",
                str(canonical_raw),
                "--image-manifest",
                str(canonical_coco),
                "--yaml",
                str(dataset_root / "data.yaml"),
                "--formats",
                preview_format,
                "--splits",
                normalized_split,
                "--output-dir",
                str(exports_dir),
                "--no-copy-images",
            ]
            run_command(step2)

            if preview_format == "yolo":
                label_path = exports_dir / "yolo" / normalized_split / "labels" / f"{image_stem}.txt"
                annotations = parse_yolo_label_file(label_path, width, height, class_names)
            elif preview_format == "voc":
                xml_path = exports_dir / "voc" / normalized_split / "annotations" / f"{image_stem}.xml"
                annotations = parse_voc_file(xml_path, class_names)
            elif preview_format == "coco":
                coco_path = exports_dir / "coco" / "annotations.coco.json"
                annotations = parse_coco_file(coco_path, image_rel_path, image_name, class_names)
            elif preview_format == "csv":
                csv_path = exports_dir / "csv" / "annotations.csv"
                annotations = parse_csv_file(csv_path, image_rel_path, image_name)
        finally:
            safe_rmtree(job_dir)

    image_bytes = draw_preview_image(
        image_path=image_path,
        annotations=annotations,
        line_width=payload.line_width,
        show_labels=payload.show_labels,
    )

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "X-Annotation-Count": str(len(annotations)),
            "X-Preview-Format": preview_format,
            "X-Preview-Mode": preview_mode,
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/export")
def export_dataset(payload: ExportRequest) -> FileResponse:
    ensure_scripts_exist()

    dataset_root = Path(payload.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {dataset_root}")
    if not (dataset_root / "data.yaml").exists():
        raise HTTPException(
            status_code=400,
            detail=f"Expected file not found: {dataset_root / 'data.yaml'}",
        )

    selected_models: list[str] = []
    selected_formats: list[str] = []
    if payload.models:
        selected_models = validate_selection(payload.models, ALLOWED_MODELS, "model")
    if payload.formats:
        selected_formats = validate_selection(payload.formats, ALLOWED_FORMATS, "format")

    formats = resolve_formats(selected_models, selected_formats)
    splits = validate_selection(payload.splits, ALLOWED_SPLITS, "split")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="export_job_", dir=str(TMP_DIR)))

    canonical_dir = job_dir / "canonical"
    exports_dir = job_dir / "exports"
    canonical_coco = canonical_dir / "annotations.coco.json"
    canonical_raw = canonical_dir / "annotations.raw.jsonl"

    try:
        step1 = [
            sys.executable,
            str(YOLO_TO_COCO_SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--splits",
            *splits,
            "--output",
            str(canonical_coco),
            "--raw-output",
            str(canonical_raw),
            "--strict",
        ]
        run_command(step1)

        step2 = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--raw",
            str(canonical_raw),
            "--image-manifest",
            str(canonical_coco),
            "--yaml",
            str(dataset_root / "data.yaml"),
            "--formats",
            *formats,
            "--splits",
            *splits,
            "--output-dir",
            str(exports_dir),
        ]
        if not payload.include_images:
            step2.append("--no-copy-images")
        run_command(step2)

        manifest = {
            "dataset_root": str(dataset_root),
            "selected_models": selected_models,
            "selected_formats": selected_formats,
            "resolved_formats": formats,
            "splits": splits,
            "include_images": payload.include_images,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        export_base_name = build_export_base_name(selected_models, formats)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"{export_base_name}_{timestamp}.zip"
        manifest["export_base_name"] = export_base_name
        manifest["zip_name"] = zip_name
        (exports_dir / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        zip_path = job_dir / zip_name
        zip_export(exports_dir, zip_path, root_dir_name=export_base_name)
    except HTTPException:
        safe_rmtree(job_dir)
        raise
    except Exception as exc:
        safe_rmtree(job_dir)
        raise HTTPException(status_code=500, detail=str(exc))

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_name,
        background=BackgroundTask(safe_rmtree, job_dir),
    )
