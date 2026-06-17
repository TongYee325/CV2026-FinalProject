"""Test GroundingDINO V2 on a real COCO image."""
import sys
sys.path.insert(0, ".")

import os
import glob
import torch
from PIL import Image
from torchvision import transforms

from grounding_dino_v2 import GroundingDINOV2
from tools.load_checkpoint_v2 import load_checkpoint_v2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Create model
    model = GroundingDINOV2().to(device)
    ckpt_path = "pretrained_weights/groundingdino_swinb_cogcoor.pth"
    load_checkpoint_v2(model, ckpt_path, verbose=True)
    model.eval()

    # 2. Find a COCO image
    image_dir = "data/coco/val2017"
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    image_path = image_paths[0]
    print(f"\nUsing image: {image_path}")

    # 3. Load image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Original size: {orig_w}x{orig_h}")

    # Official GroundingDINO uses ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    # 4. Run inference with COCO-style caption
    caption = "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush"

    print(f"Caption: {caption[:80]}...")
    results = model.predict(img_tensor, [caption], confidence_threshold=0.25)

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    print(f"\nDetected {len(boxes)} boxes:")
    for i in range(min(len(boxes), 10)):
        # Convert normalized coords back to image size
        cx, cy, w, h = boxes[i].tolist()
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        print(f"  Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]  score={scores[i].item():.3f}  label_token={labels[i].item()}")

    if len(boxes) > 10:
        print(f"  ... and {len(boxes) - 10} more")


if __name__ == "__main__":
    main()
