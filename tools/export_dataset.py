#!/usr/bin/env python3
"""Export canonical detection annotations into multiple dataset formats.

Supported formats:
- yolo: split/images + split/labels + data.yaml
- voc: Pascal VOC XML files + split/images
- csv: flat table with explicit xyxy/xywh fields
- coco: COCO JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET


ImageInfo = Dict[str, object]
AnnotationRow = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical raw annotations into multiple dataset formats."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Dataset root path containing images and canonical files.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("canonical") / "annotations.raw.jsonl",
        help="Path to canonical raw JSONL annotations.",
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=Path("canonical") / "annotations.coco.json",
        help="Optional COCO file used to keep images with zero annotations.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("data.yaml"),
        help="Path to dataset YAML for class names.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["yolo", "voc", "csv", "coco"],
        default=["yolo"],
        help="Target formats to export.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split filter (example: train test). Defaults to all splits found.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory where exported datasets are written.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy image files in exports.",
    )
    return parser.parse_args()


def parse_names_from_yaml(yaml_path: Path) -> List[str]:
    if not yaml_path.exists():
        return []

    text = yaml_path.read_text(encoding="utf-8")
    list_match = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, flags=re.M)
    if list_match:
        raw = list_match.group(1)
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

    block_match = re.search(r"^\s*names\s*:\s*$", text, flags=re.M)
    if not block_match:
        return []

    names_dict: Dict[int, str] = {}
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


def infer_class_names(rows: Sequence[AnnotationRow]) -> List[str]:
    class_map: Dict[int, str] = {}
    for row in rows:
        class_id = int(row["class_id"])
        class_name = str(row.get("class_name", class_id))
        if class_id not in class_map:
            class_map[class_id] = class_name
    return [class_map[i] for i in sorted(class_map)]


def infer_split_from_path(image_file: str) -> str:
    first_part = Path(image_file).parts[0] if Path(image_file).parts else ""
    if first_part in {"train", "val", "test", "valid"}:
        return "val" if first_part == "valid" else first_part
    return "train"


def load_raw_rows(raw_path: Path) -> List[AnnotationRow]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw annotations file: {raw_path}")

    rows: List[AnnotationRow] = []
    for idx, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {idx} in {raw_path}: {exc}") from exc
        rows.append(row)

    if not rows:
        raise ValueError(f"No annotations found in {raw_path}")
    return rows


def load_manifest_images(manifest_path: Path) -> List[ImageInfo]:
    if not manifest_path.exists():
        return []

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    if not isinstance(images, list):
        return []

    result: List[ImageInfo] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = int(image.get("id", -1))
        file_name = str(image.get("file_name", ""))
        if image_id < 0 or not file_name:
            continue
        split = str(image.get("split") or infer_split_from_path(file_name))
        if split == "valid":
            split = "val"
        result.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "split": split,
            }
        )
    return result


def filter_rows(rows: Sequence[AnnotationRow], split_filter: Optional[set[str]]) -> List[AnnotationRow]:
    if not split_filter:
        return list(rows)
    filtered: List[AnnotationRow] = []
    for row in rows:
        split = str(row.get("split") or infer_split_from_path(str(row.get("image_file", ""))))
        if split == "valid":
            split = "val"
        if split in split_filter:
            row_copy = dict(row)
            row_copy["split"] = split
            filtered.append(row_copy)
    return filtered


def build_image_index(
    rows: Sequence[AnnotationRow],
    manifest_images: Sequence[ImageInfo],
    split_filter: Optional[set[str]],
) -> List[ImageInfo]:
    images_by_id: Dict[int, ImageInfo] = {}

    for img in manifest_images:
        split = str(img.get("split", "train"))
        if split_filter and split not in split_filter:
            continue
        image_id = int(img["id"])
        images_by_id[image_id] = {
            "id": image_id,
            "file_name": str(img["file_name"]),
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
            "split": split,
        }

    for row in rows:
        image_id = int(row["image_id"])
        split = str(row.get("split") or infer_split_from_path(str(row["image_file"])))
        if split == "valid":
            split = "val"
        if split_filter and split not in split_filter:
            continue
        if image_id not in images_by_id:
            images_by_id[image_id] = {
                "id": image_id,
                "file_name": str(row["image_file"]),
                "width": int(row["image_width"]),
                "height": int(row["image_height"]),
                "split": split,
            }

    ordered = list(images_by_id.values())
    ordered.sort(key=lambda x: (str(x["split"]), str(x["file_name"])))
    return ordered


def build_annotations_by_image(rows: Sequence[AnnotationRow]) -> DefaultDict[int, List[AnnotationRow]]:
    grouped: DefaultDict[int, List[AnnotationRow]] = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(row)
    return grouped


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_requested(src: Path, dst: Path, copy_images: bool) -> None:
    if not copy_images:
        return
    mkdir(dst.parent)
    shutil.copy2(src, dst)


def copy_images_for_relative_paths(
    dataset_root: Path,
    out_dir: Path,
    image_rel_paths: Sequence[str],
    copy_images: bool,
) -> int:
    copied = 0
    seen: set[str] = set()
    for rel in image_rel_paths:
        rel_str = str(rel)
        if rel_str in seen:
            continue
        seen.add(rel_str)

        src_image = dataset_root / rel_str
        dst_image = out_dir / rel_str
        copy_if_requested(src_image, dst_image, copy_images=copy_images)
        if copy_images:
            copied += 1
    return copied


def write_yolo_yaml(output_dir: Path, class_names: Sequence[str], splits_present: Sequence[str]) -> None:
    lines: List[str] = []
    for split in ["train", "val", "test"]:
        if split in splits_present:
            lines.append(f"{split}: ./{split}/images")
    lines.append("")
    lines.append(f"nc: {len(class_names)}")
    joined_names = ", ".join(f"'{name}'" for name in class_names)
    lines.append(f"names: [{joined_names}]")
    lines.append("")
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def export_yolo(
    dataset_root: Path,
    out_dir: Path,
    image_infos: Sequence[ImageInfo],
    anns_by_image: DefaultDict[int, List[AnnotationRow]],
    class_names: Sequence[str],
    copy_images: bool,
) -> Dict[str, int]:
    ann_count = 0
    splits_present: set[str] = set()

    for image in image_infos:
        image_id = int(image["id"])
        split = str(image["split"])
        image_file = str(image["file_name"])
        image_rel = Path(image_file)

        splits_present.add(split)

        src_image = dataset_root / image_rel
        dst_image = out_dir / split / "images" / image_rel.name
        dst_label = out_dir / split / "labels" / f"{image_rel.stem}.txt"

        copy_if_requested(src_image, dst_image, copy_images=copy_images)
        mkdir(dst_label.parent)

        lines: List[str] = []
        anns = sorted(anns_by_image.get(image_id, []), key=lambda r: int(r["id"]))
        for ann in anns:
            cls_id = int(ann["class_id"])
            xc = float(ann["x_center_norm"])
            yc = float(ann["y_center_norm"])
            w = float(ann["w_norm"])
            h = float(ann["h_norm"])
            lines.append(f"{cls_id} {xc:.16g} {yc:.16g} {w:.16g} {h:.16g}")
            ann_count += 1

        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_yolo_yaml(out_dir, class_names, sorted(splits_present))
    return {"images": len(image_infos), "annotations": ann_count}


def make_voc_xml(
    folder: str,
    filename: str,
    width: int,
    height: int,
    depth: int,
    anns: Sequence[AnnotationRow],
) -> ET.Element:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"

    for ann in anns:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(ann["class_name"])
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        width_i = int(width)
        height_i = int(height)
        x1 = float(ann["x1"])
        y1 = float(ann["y1"])
        x2 = float(ann["x2"])
        y2 = float(ann["y2"])

        xmin = max(1, min(width_i, int(round(x1))))
        ymin = max(1, min(height_i, int(round(y1))))
        xmax = max(xmin, min(width_i, int(round(x2))))
        ymax = max(ymin, min(height_i, int(round(y2))))

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    return root


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def export_voc(
    dataset_root: Path,
    out_dir: Path,
    image_infos: Sequence[ImageInfo],
    anns_by_image: DefaultDict[int, List[AnnotationRow]],
    copy_images: bool,
) -> Dict[str, int]:
    split_stems: Dict[str, List[str]] = defaultdict(list)
    ann_count = 0

    for image in image_infos:
        image_id = int(image["id"])
        split = str(image["split"])
        image_file = str(image["file_name"])
        width = int(image["width"])
        height = int(image["height"])

        image_rel = Path(image_file)
        stem = image_rel.stem

        src_image = dataset_root / image_rel
        dst_image = out_dir / split / "images" / image_rel.name
        dst_xml = out_dir / split / "annotations" / f"{stem}.xml"

        copy_if_requested(src_image, dst_image, copy_images=copy_images)
        mkdir(dst_xml.parent)

        anns = sorted(anns_by_image.get(image_id, []), key=lambda r: int(r["id"]))
        root = make_voc_xml(
            folder=split,
            filename=image_rel.name,
            width=width,
            height=height,
            depth=3,
            anns=anns,
        )
        indent_xml(root)
        ET.ElementTree(root).write(dst_xml, encoding="utf-8", xml_declaration=True)

        split_stems[split].append(stem)
        ann_count += len(anns)

    image_sets_dir = out_dir / "ImageSets" / "Main"
    mkdir(image_sets_dir)
    for split, stems in split_stems.items():
        (image_sets_dir / f"{split}.txt").write_text(
            "\n".join(sorted(stems)) + "\n", encoding="utf-8"
        )

    return {"images": len(image_infos), "annotations": ann_count}


def export_csv(
    dataset_root: Path,
    out_dir: Path,
    rows: Sequence[AnnotationRow],
    image_infos: Sequence[ImageInfo],
    copy_images: bool,
) -> Dict[str, int]:
    mkdir(out_dir)
    csv_path = out_dir / "annotations.csv"
    fields = [
        "id",
        "image_id",
        "split",
        "image_file",
        "label_file",
        "image_width",
        "image_height",
        "class_id",
        "class_name",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "x_center_norm",
        "y_center_norm",
        "w_norm",
        "h_norm",
        "area",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["id"])):
            writer.writerow({field: row.get(field) for field in fields})

    image_rel_paths = [str(img["file_name"]) for img in image_infos]
    copy_images_for_relative_paths(
        dataset_root=dataset_root,
        out_dir=out_dir,
        image_rel_paths=image_rel_paths,
        copy_images=copy_images,
    )

    return {"images": len(image_infos), "annotations": len(rows)}


def export_coco(
    dataset_root: Path,
    out_dir: Path,
    image_infos: Sequence[ImageInfo],
    anns_by_image: DefaultDict[int, List[AnnotationRow]],
    class_names: Sequence[str],
    copy_images: bool,
) -> Dict[str, int]:
    mkdir(out_dir)

    images = [
        {
            "id": int(img["id"]),
            "file_name": str(img["file_name"]),
            "width": int(img["width"]),
            "height": int(img["height"]),
            "split": str(img["split"]),
        }
        for img in image_infos
    ]

    annotations: List[Dict[str, object]] = []
    for img in image_infos:
        image_id = int(img["id"])
        anns = sorted(anns_by_image.get(image_id, []), key=lambda r: int(r["id"]))
        for row in anns:
            annotations.append(
                {
                    "id": int(row["id"]),
                    "image_id": image_id,
                    "category_id": int(row["class_id"]) + 1,
                    "bbox": [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["w"]),
                        float(row["h"]),
                    ],
                    "area": float(row["area"]),
                    "iscrowd": 0,
                }
            )

    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    coco = {
        "info": {
            "description": "COCO exported from canonical raw annotations",
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    out_path = out_dir / "annotations.coco.json"
    out_path.write_text(json.dumps(coco, ensure_ascii=True, indent=2), encoding="utf-8")

    image_rel_paths = [str(img["file_name"]) for img in image_infos]
    copy_images_for_relative_paths(
        dataset_root=dataset_root,
        out_dir=out_dir,
        image_rel_paths=image_rel_paths,
        copy_images=copy_images,
    )

    return {"images": len(images), "annotations": len(annotations)}


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    raw_path = args.raw if args.raw.is_absolute() else (dataset_root / args.raw)
    manifest_path = (
        args.image_manifest if args.image_manifest.is_absolute() else (dataset_root / args.image_manifest)
    )
    yaml_path = args.yaml if args.yaml.is_absolute() else (dataset_root / args.yaml)
    output_root = args.output_dir if args.output_dir.is_absolute() else (dataset_root / args.output_dir)

    split_filter = None
    if args.splits:
        split_filter = {s.strip().replace("valid", "val") for s in args.splits}

    rows = load_raw_rows(raw_path)
    rows = filter_rows(rows, split_filter)
    if not rows:
        raise ValueError("No rows matched the requested split filter.")

    manifest_images = load_manifest_images(manifest_path)
    image_infos = build_image_index(rows, manifest_images, split_filter)
    anns_by_image = build_annotations_by_image(rows)

    class_names = parse_names_from_yaml(yaml_path)
    if not class_names:
        class_names = infer_class_names(rows)

    copy_images = not args.no_copy_images
    results: Dict[str, Dict[str, int]] = {}

    for fmt in args.formats:
        out_dir = output_root / fmt
        if out_dir.exists():
            shutil.rmtree(out_dir)
        mkdir(out_dir)

        if fmt == "yolo":
            results[fmt] = export_yolo(
                dataset_root=dataset_root,
                out_dir=out_dir,
                image_infos=image_infos,
                anns_by_image=anns_by_image,
                class_names=class_names,
                copy_images=copy_images,
            )
        elif fmt == "voc":
            results[fmt] = export_voc(
                dataset_root=dataset_root,
                out_dir=out_dir,
                image_infos=image_infos,
                anns_by_image=anns_by_image,
                copy_images=copy_images,
            )
        elif fmt == "csv":
            results[fmt] = export_csv(
                dataset_root=dataset_root,
                out_dir=out_dir,
                rows=rows,
                image_infos=image_infos,
                copy_images=copy_images,
            )
        elif fmt == "coco":
            results[fmt] = export_coco(
                dataset_root=dataset_root,
                out_dir=out_dir,
                image_infos=image_infos,
                anns_by_image=anns_by_image,
                class_names=class_names,
                copy_images=copy_images,
            )
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    print("Export complete.")
    print(f"Output root: {output_root}")
    print(f"Image count (selected): {len(image_infos)}")
    for fmt in args.formats:
        stats = results[fmt]
        print(f"- {fmt}: images={stats['images']} annotations={stats['annotations']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
