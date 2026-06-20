"""Create RefCOCO success/failure examples and contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.refcoco_utils import resolve_image_path  # noqa: E402


COLORS = {
    "gt": (35, 170, 75),
    "prediction": (220, 55, 55),
    "text": (25, 25, 25),
    "background": (250, 250, 250),
}

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def load_rows(results_dir: Path) -> List[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"summary.json", "sample_manifest.json"} or path.name.endswith(
            "_errors.json"
        ):
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            for row in payload:
                row["_source_split"] = path.stem
            rows.extend(payload)
    return [row for row in rows if row.get("status") == "ok"]


def select_diverse(rows: Iterable[dict], count: int, success: bool) -> List[dict]:
    candidates = [
        row for row in rows
        if bool(row.get("correct")) is success and row.get("pred_box_xyxy")
    ]
    candidates.sort(
        key=lambda row: float(row["iou"]), reverse=success
    )
    buckets = defaultdict(list)
    for row in candidates:
        for tag in row.get("tags", ["other"]):
            buckets[tag].append(row)

    selected = []
    seen = set()
    preferred_tags = [
        "attribute", "position", "relation", "occlusion", "small_target", "person", "other"
    ]
    while len(selected) < count:
        progressed = False
        for tag in preferred_tags:
            while buckets[tag]:
                row = buckets[tag].pop(0)
                key = (row["image_id"], row["sent_id"])
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def draw_box(draw: ImageDraw.ImageDraw, box: List[float], color, width: int = 4) -> None:
    draw.rectangle(tuple(box), outline=color, width=width)


def render_example(row: dict, image_dir: Path, output_path: Path,
                   canvas_width: int = 1200) -> Path:
    image = Image.open(resolve_image_path(image_dir, row["image_id"])).convert("RGB")
    max_image_height = 850
    scale = min(canvas_width / image.width, max_image_height / image.height, 1.0)
    resized = image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    text_height = 220
    canvas = Image.new("RGB", (canvas_width, resized.height + text_height), COLORS["background"])
    canvas.paste(resized, ((canvas_width - resized.width) // 2, text_height))
    offset_x = (canvas_width - resized.width) // 2
    draw = ImageDraw.Draw(canvas)

    def scaled_box(box):
        return [
            box[0] * scale + offset_x,
            box[1] * scale + text_height,
            box[2] * scale + offset_x,
            box[3] * scale + text_height,
        ]

    draw_box(draw, scaled_box(row["gt_box_xyxy"]), COLORS["gt"])
    draw_box(draw, scaled_box(row["pred_box_xyxy"]), COLORS["prediction"])
    expression_font = load_font(32, bold=True)
    metadata_font = load_font(24)
    legend_font = load_font(24, bold=True)
    expression = "\n".join(textwrap.wrap(row["expression"], width=62))
    header = (
        f"{row['_source_split']} | IoU={row['iou']:.3f} | "
        f"score={row['score']:.3f} | tags={','.join(row.get('tags', []))}"
    )
    draw.text((24, 18), expression, fill=COLORS["text"], font=expression_font)
    draw.text((24, 124), header, fill=COLORS["text"], font=metadata_font)
    draw.text((24, 172), "GT", fill=COLORS["gt"], font=legend_font)
    draw.text((92, 172), "Prediction", fill=COLORS["prediction"], font=legend_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=96, subsampling=0)
    return output_path


def make_contact_sheet(
    paths: List[Path],
    output_path: Path,
    columns: int = 3,
    thumb_width: int = 900,
) -> None:
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    thumbs = []
    for image in images:
        ratio = thumb_width / image.width
        thumbs.append(
            image.resize(
                (thumb_width, int(image.height * ratio)), Image.Resampling.LANCZOS
            )
        )
    row_heights = []
    for start in range(0, len(thumbs), columns):
        row_heights.append(max(image.height for image in thumbs[start:start + columns]))
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, sum(row_heights)),
        COLORS["background"],
    )
    y = 0
    for row_index, start in enumerate(range(0, len(thumbs), columns)):
        for column, image in enumerate(thumbs[start:start + columns]):
            sheet.paste(image, (column * thumb_width, y))
        y += row_heights[row_index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=96, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/refcoco"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/coco/train2014"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/refcoco/visualizations"))
    parser.add_argument("--success-count", type=int, default=6)
    parser.add_argument("--failure-count", type=int, default=6)
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    groups = {
        "success": select_diverse(rows, args.success_count, True),
        "failure": select_diverse(rows, args.failure_count, False),
    }
    selection = {}
    for group, selected in groups.items():
        paths = []
        for index, row in enumerate(selected, start=1):
            path = args.output_dir / group / f"{index:02d}_{row['_source_split']}.jpg"
            paths.append(render_example(row, args.image_dir, path))
        make_contact_sheet(paths, args.output_dir / f"{group}_contact_sheet.jpg")
        selection[group] = [
            {
                "split": row["_source_split"],
                "image_id": row["image_id"],
                "sent_id": row["sent_id"],
                "expression": row["expression"],
                "iou": row["iou"],
                "tags": row.get("tags", []),
            }
            for row in selected
        ]
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
