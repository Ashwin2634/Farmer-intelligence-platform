import json
import os
from typing import Dict, Optional, Any
import re


class KnowledgeBase:
    def __init__(self, kb_path: str = "disease_detection/data/Knoledge_base.json"):
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")

        with open(kb_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Build fast lookup: (normalized_species, normalized_disease) → entry
        self.lookup = {}
        for entry in self.data:
            species = self._normalize(entry.get("species", ""))
            disease = self._normalize(entry.get("disease", ""))
            self.lookup[(species, disease)] = entry

        # Mapping from model crop names → knowledge base species
        self.crop_map = {
            "bellpepper": "capsicum",
            "bell_pepper": "capsicum",
            "dragonfruit": "dragon_fruit",
            "dragon_fruit": "dragon_fruit",
            "eggplant": "brinjal",
            "blueberryfruit": "blueberry",
            "blueberryleaf": "blueberry",
            "blueberry": "blueberry",
            "squash": "zucchini",
            "chilli": "chilli",
            "chili": "chilli",
            "muskmelon": "muskmelon",
            "papaya": "papaya",
            "spinach": "spinach",
            "strawberry": "strawberry",
            "basil": "basil",
            "broccoli": "broccoli",
            "cauliflower": "cauliflower",
            "cucumber": "cucumber",
            "lettuce": "lettuce",
            "zucchini": "zucchini",
            "tomato": "tomato",
            "watermelon": "watermelon",
            # Add more if needed
        }

    def _normalize(self, text: str) -> str:
        """Lowercase + replace spaces/underscores/hyphens with single underscore"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[\s\-_]+", "_", text)
        return text

    def get_treatment(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Given model class_name (e.g. 'Muskmelon___Powdery_Mildew' or 'DragonFruit___Stem_Canker')
        return the matching knowledge base entry or None.
        """
        if not class_name or "healthy" in class_name.lower():
            return None

        # Split crop and disease
        if "___" in class_name:
            crop_raw, disease_raw = class_name.split("___", 1)
        elif "_" in class_name:
            parts = class_name.split("_")
            crop_raw = parts[0]
            disease_raw = "_".join(parts[1:])
        else:
            return None

        crop_norm = self._normalize(crop_raw)
        disease_norm = self._normalize(disease_raw)

        # Map model crop name → KB species
        species = self.crop_map.get(crop_norm, crop_norm)

        # 1. Exact match
        key = (species, disease_norm)
        if key in self.lookup:
            return self.lookup[key]

        # 2. Try common variations
        # e.g. "powdery_mildew_leaf" → "powdery_mildew"
        #      "anthracnose_fruit_rot" → "anthracnose"
        disease_variants = [
            disease_norm,
            disease_norm.replace("_fruit", "").replace("_leaf", "").replace("_rot", ""),
            re.sub(r"_fruit_rot|_leaf_spot|_leaf|_fruit|_rot|_blight$", "", disease_norm),
        ]

        for d in disease_variants:
            key = (species, d)
            if key in self.lookup:
                return self.lookup[key]

        # 3. Partial / contains match (last resort)
        for (sp, dis), entry in self.lookup.items():
            if sp == species and (dis in disease_norm or disease_norm in dis):
                return entry

        return None