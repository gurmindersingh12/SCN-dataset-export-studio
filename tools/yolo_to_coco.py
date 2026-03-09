#!/usr/bin/env python3
"""Convert a YOLO detection dataset into canonical annotation files.

This script expects a dataset layout like:
  dataset_root/
    data.yaml
    train/images/*.jpg
    train/labels/*.txt
    test/images/*.jpg
    test/labels/*.txt

It reads class names from data.yaml and writes:
1) a COCO JSON (widely interoperable)
2) a raw JSONL file with explicit xyxy/xywh boxes for model-agnostic exports
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels to canonical COCO JSON."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Path to YOLO dataset root (contains data.yaml and split folders).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Path to data.yaml (defaults to <dataset-root>/data.yaml).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to include (e.g. train val test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("canonical") / "annotations.coco.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("canonical") / "annotations.raw.jsonl",
        help="Output raw JSONL path with explicit xyxy boxes.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip writing the raw JSONL output file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a label file has no matching image or malformed rows.",
    )
    return parser.parse_args()


def parse_names_from_yaml(yaml_path: Path) -> List[str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing YAML file: {yaml_path}")

    text = yaml_path.read_text(encoding="utf-8")

    list_match = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, flags=re.M)
    if list_match:
        raw = list_match.group(1)
        # Split comma list while preserving quoted strings.
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        names: List[str] = []
        for part in parts:
            if (part.startswith("'") and part.endswith("'")) or (
                part.startswith('"') and part.endswith('"')
            ):
                part = part[1:-1]
            names.append(part)
        if names:
            return names

    # Fallback for block format:
    # names:
    #   0: class_a
    #   1: class_b
    block_match = re.search(r"^\s*names\s*:\s*$", text, flags=re.M)
    if not block_match:
        raise ValueError("Could not parse class names from YAML 'names' field.")

    names_dict: Dict[int, str] = {}
    lines = text[block_match.end() :].splitlines()
    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^[^\s]", line):
            # Reached next top-level key.
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

    if not names_dict:
        raise ValueError("Could not parse class names from YAML 'names' field.")

    return [names_dict[i] for i in sorted(names_dict)]


def image_size(path: Path) -> Tuple[int, int]:
    # Parse common image headers without external dependencies.
    with path.open("rb") as f:
        header = f.read(32)

        # PNG signature + IHDR width/height.
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            f.seek(16)
            width = int.from_bytes(f.read(4), "big")
            height = int.from_bytes(f.read(4), "big")
            return width, height

        # JPEG: scan for SOF markers.
        if header[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue

                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    break

                if marker in {b"\xd8", b"\xd9"}:
                    continue

                length_bytes = f.read(2)
                if len(length_bytes) != 2:
                    break
                seg_len = int.from_bytes(length_bytes, "big")
                if seg_len < 2:
                    break

                # SOF0, SOF1, SOF2, etc. excluding DHT/DAC/JPG.
                if marker in {
                    b"\xc0",
                    b"\xc1",
                    b"\xc2",
                    b"\xc3",
                    b"\xc5",
                    b"\xc6",
                    b"\xc7",
                    b"\xc9",
                    b"\xca",
                    b"\xcb",
                    b"\xcd",
                    b"\xce",
                    b"\xcf",
                }:
                    _precision = f.read(1)
                    height = int.from_bytes(f.read(2), "big")
                    width = int.from_bytes(f.read(2), "big")
                    return width, height

                f.seek(seg_len - 2, 1)

        # BMP width/height (little-endian DIB header at offset 18).
        if header.startswith(b"BM"):
            f.seek(18)
            width = int.from_bytes(f.read(4), "little", signed=True)
            height = int.from_bytes(f.read(4), "little", signed=True)
            return abs(width), abs(height)

    raise ValueError(f"Unsupported image format or corrupt file: {path}")


def build_image_map(images_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not images_dir.exists():
        return mapping

    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mapping[path.stem] = path
    return mapping


def parse_yolo_row(row: str, line_num: int, label_path: Path) -> Tuple[int, float, float, float, float]:
    pieces = row.strip().split()
    if len(pieces) != 5:
        raise ValueError(
            f"Malformed row in {label_path} line {line_num}: expected 5 columns, got {len(pieces)}"
        )
    cls_id = int(float(pieces[0]))
    cx = float(pieces[1])
    cy = float(pieces[2])
    w = float(pieces[3])
    h = float(pieces[4])
    return cls_id, cx, cy, w, h


def yolo_to_coco_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    abs_w = max(0.0, min(float(img_w), w * img_w))
    abs_h = max(0.0, min(float(img_h), h * img_h))
    x_min = (cx * img_w) - (abs_w / 2.0)
    y_min = (cy * img_h) - (abs_h / 2.0)
    x_min = max(0.0, min(x_min, float(img_w)))
    y_min = max(0.0, min(y_min, float(img_h)))

    # Keep bbox inside image.
    if x_min + abs_w > img_w:
        abs_w = max(0.0, img_w - x_min)
    if y_min + abs_h > img_h:
        abs_h = max(0.0, img_h - y_min)

    return x_min, y_min, abs_w, abs_h


def iter_split_items(split_dir: Path) -> Iterable[Tuple[Path, Path]]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_map = build_image_map(images_dir)

    if not labels_dir.exists():
        return

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = image_map.get(label_path.stem)
        yield label_path, image_path


def convert(
    dataset_root: Path,
    yaml_path: Path,
    splits: Sequence[str],
    output_path: Path,
    raw_output_path: Optional[Path],
    strict: bool,
) -> Dict[str, int]:
    class_names = parse_names_from_yaml(yaml_path)
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    raw_annotations: List[Dict[str, object]] = []

    image_id = 1
    ann_id = 1

    missing_image_count = 0
    malformed_row_count = 0

    for split in splits:
        split_dir = dataset_root / split
        for label_path, image_path in iter_split_items(split_dir):
            if image_path is None:
                missing_image_count += 1
                msg = f"No image matched label: {label_path}"
                if strict:
                    raise FileNotFoundError(msg)
                print(f"WARN: {msg}")
                continue

            width, height = image_size(image_path)
            image_rel_path = str(image_path.relative_to(dataset_root)).replace("\\", "/")
            label_rel_path = str(label_path.relative_to(dataset_root)).replace("\\", "/")

            images.append(
                {
                    "id": image_id,
                    "file_name": image_rel_path,
                    "width": width,
                    "height": height,
                    "split": split,
                    "label_file": label_rel_path,
                }
            )

            rows = label_path.read_text(encoding="utf-8").splitlines()
            for line_num, row in enumerate(rows, start=1):
                if not row.strip():
                    continue
                try:
                    cls_id, cx, cy, w, h = parse_yolo_row(row, line_num, label_path)
                except Exception:
                    malformed_row_count += 1
                    if strict:
                        raise
                    print(
                        f"WARN: Skipping malformed row in {label_path} line {line_num}: {row.strip()}"
                    )
                    continue

                if cls_id < 0 or cls_id >= len(class_names):
                    malformed_row_count += 1
                    msg = (
                        f"Class id {cls_id} out of range in {label_path} line {line_num} "
                        f"(classes: 0..{len(class_names)-1})"
                    )
                    if strict:
                        raise ValueError(msg)
                    print(f"WARN: {msg}")
                    continue

                x, y, bw, bh = yolo_to_coco_bbox(cx, cy, w, h, width, height)
                x2 = x + bw
                y2 = y + bh
                area = bw * bh

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": [x, y, bw, bh],
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                if raw_output_path:
                    raw_annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "split": split,
                            "image_file": image_rel_path,
                            "label_file": label_rel_path,
                            "image_width": width,
                            "image_height": height,
                            "class_id": cls_id,
                            "class_name": class_names[cls_id],
                            "x1": x,
                            "y1": y,
                            "x2": x2,
                            "y2": y2,
                            "w": bw,
                            "h": bh,
                            "x_center_norm": cx,
                            "y_center_norm": cy,
                            "w_norm": w,
                            "h_norm": h,
                            "area": area,
                        }
                    )
                ann_id += 1

            image_id += 1

    output = {
        "info": {
            "description": "Canonical COCO exported from YOLO labels",
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    if raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_output_path.open("w", encoding="utf-8") as raw_file:
            for row in raw_annotations:
                raw_file.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "images": len(images),
        "annotations": len(annotations),
        "raw_annotations": len(raw_annotations),
        "categories": len(categories),
        "missing_images": missing_image_count,
        "malformed_rows": malformed_row_count,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    yaml_path = args.yaml.resolve() if args.yaml else (dataset_root / "data.yaml").resolve()
    output_path = args.output.resolve() if args.output.is_absolute() else (dataset_root / args.output)
    raw_output_path = None
    if not args.skip_raw:
        raw_output_path = (
            args.raw_output.resolve()
            if args.raw_output.is_absolute()
            else (dataset_root / args.raw_output)
        )

    stats = convert(
        dataset_root=dataset_root,
        yaml_path=yaml_path,
        splits=args.splits,
        output_path=output_path,
        raw_output_path=raw_output_path,
        strict=args.strict,
    )

    print("Wrote canonical COCO file:")
    print(f"  {output_path}")
    if raw_output_path:
        print("Wrote canonical raw JSONL file:")
        print(f"  {raw_output_path}")
    print("Summary:")
    for key in [
        "images",
        "annotations",
        "raw_annotations",
        "categories",
        "missing_images",
        "malformed_rows",
    ]:
        print(f"  {key}: {stats[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
