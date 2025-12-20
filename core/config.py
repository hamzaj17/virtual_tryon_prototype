from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    # NOTE: runwayml/stable-diffusion-inpainting was deprecated; this mirror is commonly used.
    base_ckpt: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    # CatVTON attention checkpoints (HF repo)
    attn_ckpt: str = "zhengchong/CatVTON"
    attn_ckpt_version: str = "mix"  # 'mix' is the general checkpoint

@dataclass(frozen=True)
class AppConfig:
    # CatVTON default resolution
    width: int = 768
    height: int = 1024

    uploads_dir: str = "data/uploads/people"
    outputs_dir: str = "data/outputs"
    garments_dir: str = "assets/garments"
    catalog_json: str = "assets/catalog.json"
