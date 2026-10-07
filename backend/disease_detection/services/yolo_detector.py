# disease_detection/services/yolo_detector.py
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from disease_detection.services.knowledge_base import KnowledgeBase


from pathlib import Path

# Base directory = folder where this script lives
BASE_DIR = Path(__file__).resolve().parent.parent

class EfficientNetDetector:
    def __init__(
        self,
        model_path: str = str(BASE_DIR / "models" / "trained" / "V2.1_EfficientNetV2-M_best.pth"),
        class_names_path: str = str(BASE_DIR / "models" / "trained" / "class_names.txt"),
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kb = KnowledgeBase()

        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")

        with open(class_names_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(self.class_names)} classes")

        # ---------- Load model ----------
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
            print("Loaded full model object")
        else:
            print("Detected state_dict / checkpoint dict → rebuilding model")

            self.model = models.efficientnet_v2_m(weights=None)
            self.model.classifier[1] = torch.nn.Linear(
                self.model.classifier[1].in_features,
                len(self.class_names),
            )

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

        # Must match training preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _describe_class(self, class_id: int) -> Dict:
        """Turn a class index into crop / disease / treatment info."""
        class_name = self.class_names[class_id]

        parts = class_name.split("___")  # classes use triple underscore
        if len(parts) == 2:
            crop = parts[0]
            disease = parts[1].replace("_", " ").title()
        else:
            crop = ""
            disease = class_name.replace("_", " ").title()

        is_healthy = "healthy" in class_name.lower()
        treatment = None if is_healthy else self.kb.get_treatment(class_name)

        return {
            "class_name": class_name,
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
            "treatment": treatment,
        }

    @torch.inference_mode()
    def predict_probs(self, images: List[Image.Image], batch_size: int = 8) -> np.ndarray:
        """
        Run the model on many images.
        Returns softmax probabilities, shape [N, num_classes].
        Lower batch_size if you run out of GPU memory.
        """
        all_probs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            batch = torch.stack(
                [self.transform(img.convert("RGB")) for img in chunk]
            ).to(self.device)
            probs = F.softmax(self.model(batch), dim=1)
            all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    # ------------------------------------------------------------------
    # Single image (same output as before)
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image) -> Dict:
        width, height = image.size

        start = time.perf_counter()
        probs = self.predict_probs([image])[0]
        inference_time = round((time.perf_counter() - start) * 1000, 2)

        class_id = int(probs.argmax())
        info = self._describe_class(class_id)

        return {
            "model": "Ashwin's_AI_Model",
            "image": {"width": width, "height": height},
            "class_id": class_id,
            "class_name": info["class_name"],
            "crop": info["crop"],
            "disease": info["disease"],
            "confidence": round(float(probs[class_id]), 4),
            "is_healthy": info["is_healthy"],
            "inference_time_ms": inference_time,
            "treatment": info["treatment"],
        }