from ultralytics import YOLO
from PIL import Image
from typing import List, Dict
import time


class YOLODetector:
    def __init__(
        self,
        model_path: str = "disease_detection/models/trained/best_v6.pt",
        confidence: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = self.model.names

    def detect(self, image: Image.Image) -> Dict:

        width, height = image.size

        start = time.perf_counter()

        results = self.model(
            image,
            conf=self.confidence,
            verbose=False,
        )

        inference_time = round((time.perf_counter() - start) * 1000, 2)

        detections = []

        for result in results:

            boxes = result.boxes

            masks = result.masks

            polygons = []

            if masks is not None:
                polygons = masks.xy

            for idx, box in enumerate(boxes):

                class_id = int(box.cls.item())

                confidence = float(box.conf.item())

                class_name = self.class_names[class_id]

                parts = class_name.split("_")

                crop = parts[0].capitalize()

                disease = " ".join(parts[1:]).replace("_", " ").title()

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                polygon = []

                if idx < len(polygons):

                    polygon = [
                        {
                            "x": round(float(x), 2),
                            "y": round(float(y), 2),
                        }
                        for x, y in polygons[idx]
                    ]

                detections.append(
                    {
                        "class_id": class_id,
                        "crop": crop,
                        "disease": disease,
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                        "polygon": polygon,
                    }
                )

        return {
            "model": "YOLO11 Segmentation",
            "image": {
                "width": width,
                "height": height,
            },
            "total_detections": len(detections),
            "inference_time_ms": inference_time,
            "detections": detections,
        }
