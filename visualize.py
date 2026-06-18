"""Run Grounding DINO V2 on one image and save predicted boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from grounding_dino_v2 import GroundingDINOV2
from tools.load_checkpoint_v2 import load_checkpoint_v2
from tools.refcoco_utils import (
    cxcywh_normalized_to_xyxy,
    normalize_expression,
    preprocess_image,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="pretrained_weights/groundingdino_swinb_cogcoor.pth",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=Path("output_vis.jpg"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    model = GroundingDINOV2().to(device)
    load_checkpoint_v2(model, args.checkpoint, verbose=True)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    tensor = preprocess_image(image).unsqueeze(0).to(device)
    prompt = normalize_expression(args.text, append_period=True)
    output = model.predict(tensor, [prompt], confidence_threshold=args.threshold)[0]
    boxes = cxcywh_normalized_to_xyxy(output["boxes"], width, height).cpu().tolist()
    scores = output["scores"].cpu().tolist()
    labels = output["labels"].cpu().tolist()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, score, token_index in zip(boxes, scores, labels):
        token = model.tokenizer.convert_ids_to_tokens(
            model.tokenizer(prompt, return_tensors="pt")["input_ids"][0, token_index].item()
        )
        draw.rectangle(tuple(box), outline=(230, 50, 50), width=3)
        label = f"{token} {score:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=(230, 50, 50))
        draw.text((box[0], box[1]), label, fill="white", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(f"Saved {len(boxes)} predictions to {args.output}")


if __name__ == "__main__":
    main()
