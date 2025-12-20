from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict

from PIL import Image

@dataclass
class Garment:
    id: str
    name: str
    path: str

def rebuild_catalog(garments_dir: str, catalog_json: str) -> List[Garment]:
    os.makedirs(garments_dir, exist_ok=True)
    items = []
    for fn in sorted(os.listdir(garments_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            gid = os.path.splitext(fn)[0]
            items.append({"id": gid, "name": gid.replace("_", " ").title(), "path": os.path.join(garments_dir, fn)})
    os.makedirs(os.path.dirname(catalog_json), exist_ok=True)
    with open(catalog_json, "w", encoding="utf-8") as f:
        json.dump({"garments": items}, f, indent=2)
    return [Garment(**x) for x in items]

def load_catalog(catalog_json: str) -> List[Garment]:
    if not os.path.exists(catalog_json):
        return []
    with open(catalog_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Garment(**x) for x in data.get("garments", [])]

def load_garment_image(garment: Garment) -> Image.Image:
    return Image.open(garment.path).convert("RGB")
