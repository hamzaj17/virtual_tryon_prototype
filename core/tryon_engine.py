from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from PIL import Image

from .config import ModelConfig, AppConfig
from .catvton.pipeline import CatVTONPipeline
from .masking import AutoMasker, MaskResult
from .catalog import Garment, load_garment_image

@dataclass
class TryOnResult:
    session_id: str
    output_path: str
    mask_path: str
    overlay_path: str
    person_path: str
    garment_path: str

class TryOnEngine:
    def __init__(self, model_cfg: ModelConfig, app_cfg: AppConfig):
        self.model_cfg = model_cfg
        self.app_cfg = app_cfg

        os.makedirs(app_cfg.uploads_dir, exist_ok=True)
        os.makedirs(app_cfg.outputs_dir, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.pipeline = CatVTONPipeline(
            base_ckpt=model_cfg.base_ckpt,
            attn_ckpt=model_cfg.attn_ckpt,
            attn_ckpt_version=model_cfg.attn_ckpt_version,
            device=device,
            weight_dtype=dtype,
            compile=False,
            use_tf32=True,
        )
        self.masker = AutoMasker()

    def run(self, person_img: Image.Image, garment: Garment, steps: int = 35, guidance: float = 2.5) -> Tuple[TryOnResult, Image.Image, Image.Image, Image.Image]:
        session_id = uuid.uuid4().hex[:12]
        out_dir = os.path.join(self.app_cfg.outputs_dir, session_id)
        os.makedirs(out_dir, exist_ok=True)

        # Load garment image
        garment_img = load_garment_image(garment)

        # Resize person to app cfg size now so mask matches
        person_resized = person_img.convert("RGB").resize((self.app_cfg.width, self.app_cfg.height), Image.BICUBIC)

        mask_res: MaskResult = self.masker.make_upper_body_mask(person_resized)
        mask = mask_res.mask

        # Save inputs
        person_path = os.path.join(out_dir, "person.png")
        garment_path = os.path.join(out_dir, "garment.png")
        mask_path = os.path.join(out_dir, "auto_mask.png")
        overlay_path = os.path.join(out_dir, "mask_overlay.png")
        out_path = os.path.join(out_dir, "result.png")

        person_resized.save(person_path)
        garment_img.save(garment_path)
        mask.save(mask_path)
        mask_res.overlay.save(overlay_path)

        # Run pipeline
        # NOTE: pipeline does its own resize/crop/pad to width/height.
        output = self.pipeline(
            image=person_resized,
            condition_image=garment_img,
            mask=mask,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=self.app_cfg.height,
            width=self.app_cfg.width,
        )

        output.save(out_path)

        return (
            TryOnResult(
                session_id=session_id,
                output_path=out_path,
                mask_path=mask_path,
                overlay_path=overlay_path,
                person_path=person_path,
                garment_path=garment_path,
            ),
            output,
            mask_res.overlay,
            mask,
        )
