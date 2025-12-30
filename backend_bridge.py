# backend_bridge.py
import os
from PIL import Image

from core.config import AppConfig, ModelConfig
from core.catalog import load_catalog, rebuild_catalog
from core.tryon_engine import TryOnEngine


class BackendBridge:
    """
    Thin wrapper around your existing backend so Tkinter can call it
    with minimal code changes.
    """
    def __init__(self):
        self.app_cfg = AppConfig()
        self.model_cfg = ModelConfig()

        # Ensure folders exist
        os.makedirs(self.app_cfg.garments_dir, exist_ok=True)
        os.makedirs(self.app_cfg.uploads_dir, exist_ok=True)
        os.makedirs(self.app_cfg.outputs_dir, exist_ok=True)

        # Build catalog if missing
        if not os.path.exists(self.app_cfg.catalog_json):
            rebuild_catalog(self.app_cfg.garments_dir, self.app_cfg.catalog_json)

        self.catalog = load_catalog(self.app_cfg.catalog_json)
        self.engine = TryOnEngine(self.model_cfg, self.app_cfg)

    def refresh_catalog(self):
        rebuild_catalog(self.app_cfg.garments_dir, self.app_cfg.catalog_json)
        self.catalog = load_catalog(self.app_cfg.catalog_json)
        return self.catalog

    def list_catalog_items(self):
        """
        Returns list of dict: {name, path, image(PIL RGBA), obj(original garment entry)}
        """
        items = []
        for g in self.catalog:
            try:
                im = Image.open(g.path).convert("RGBA")
            except Exception:
                continue

            # pretty name
            name = getattr(g, "id", None) or os.path.splitext(os.path.basename(g.path))[0]
            pretty = str(name).replace("_", " ").replace("-", " ").title()

            items.append({
                "name": pretty,
                "path": g.path,
                "image": im,
                "obj": g
            })
        return items

    def run_tryon(self, person_rgba, garment_obj, steps=35, guidance=2.5):
        """
        person_rgba: PIL RGBA or RGB
        garment_obj: one entry from catalog list
        returns: (out_img_rgba, overlay_rgb, mask_rgb, info_text)
        """
        if person_rgba is None:
            raise RuntimeError("Missing person image.")
        if garment_obj is None:
            raise RuntimeError("Missing garment selection.")

        person_rgb = person_rgba.convert("RGB")
        result, out_img, overlay, mask = self.engine.run(
            person_rgb, garment_obj, steps=steps, guidance=guidance
        )

        info = (
            f"Session: {result.session_id}\n"
            f"Saved to: {os.path.join(self.app_cfg.outputs_dir, result.session_id)}\n"
            f"Files: result.png, person.png, garment.png, auto_mask.png, mask_overlay.png"
        )

        # normalize formats for Tk
        out_img_rgba = out_img.convert("RGBA")
        overlay_rgb = overlay.convert("RGB") if overlay is not None else None
        mask_rgb = mask.convert("RGB") if mask is not None else None

        return out_img_rgba, overlay_rgb, mask_rgb, info
