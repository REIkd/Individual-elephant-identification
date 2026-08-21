"""训练包内模型评测（compare_training_runs 用）。"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torchvision import transforms

from elephant_net import build_model


class ElephantClassifier:
    def __init__(
        self,
        model_path: str = "best_elephant_model.pth",
        class_names_path: str = "class_names.json",
        cuda_device: int = 0,
    ):
        if int(cuda_device) < 0 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            dev = max(0, min(int(cuda_device), torch.cuda.device_count() - 1))
            self.device = torch.device(f"cuda:{dev}")
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
        else:
            with open(class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

        arch = "resnet50"
        image_size = 224
        if isinstance(checkpoint, dict):
            arch = checkpoint.get("arch", arch)
            image_size = int(checkpoint.get("image_size", image_size))

        self.arch = arch
        self.image_size = image_size
        self.model = build_model(arch, len(self.class_names), pretrained=False)
        state = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if self.device.type == "cpu":
            try:
                default_nt = max(1, min(8, (os.cpu_count() or 4)))
                _nt = int(os.environ.get("ELEPHANT_TORCH_THREADS", str(default_nt)))
                torch.set_num_threads(max(1, _nt))
                torch.set_num_interop_threads(1)
            except Exception:
                pass

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            probs = torch.softmax(logits.float(), dim=1)
        all_probs = {
            self.class_names[i]: probs[0, i].item() * 100
            for i in range(len(self.class_names))
        }
        best = max(all_probs, key=all_probs.get)
        return best, all_probs[best], all_probs
