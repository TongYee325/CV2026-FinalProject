"""Print a concise reproducibility and data readiness report."""

import json
from pathlib import Path

import cv2
import timm
import tokenizers
import torch
import transformers


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.is_dir() else 0


def main() -> None:
    checkpoint = Path("pretrained_weights/groundingdino_swinb_cogcoor.pth")
    expected_checkpoint_bytes = 938057991
    checkpoint_keys = None
    if checkpoint.is_file() and checkpoint.stat().st_size == expected_checkpoint_bytes:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        checkpoint_keys = len(state_dict)
        del payload, state_dict

    bert_root = Path("hf_models/bert-base-uncased")
    bert_files = ["config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json"]
    manifest_path = Path("manifests/refcoco_subset_seed2026.json")
    manifest_samples = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest_samples = sum(len(samples) for samples in manifest["splits"].values())

    report = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "timm": timm.__version__,
        "opencv": cv2.__version__,
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "checkpoint_size_ok": (
            checkpoint.is_file() and checkpoint.stat().st_size == expected_checkpoint_bytes
        ),
        "checkpoint_state_dict_keys": checkpoint_keys,
        "local_bert_files_ok": all((bert_root / name).is_file() for name in bert_files),
        "refcoco_manifest_samples": manifest_samples,
        "coco_val2017_images": count_files(Path("data/coco/val2017"), "*.jpg"),
        "coco_train2014_images": count_files(Path("data/coco/train2014"), "*.jpg"),
    }
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
