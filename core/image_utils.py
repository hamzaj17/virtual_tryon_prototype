import math
from typing import Tuple, Union, List

import numpy as np
import torch
from PIL import Image

def _to_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    raise TypeError("Unsupported image type")

def resize_and_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize keeping aspect ratio, then center-crop to exact (width, height)."""
    target_w, target_h = size
    img = _to_pil(img).convert("RGB")
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))

def resize_and_padding(img: Image.Image, size: Tuple[int, int], pad_color=(255, 255, 255)) -> Image.Image:
    """Resize keeping aspect ratio to fit within (width, height), pad to exact size."""
    target_w, target_h = size
    img = _to_pil(img).convert("RGB")
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_r = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(img_r, (left, top))
    return canvas

def prepare_image(image: Image.Image) -> torch.Tensor:
    """PIL -> torch float tensor in [-1, 1], shape (1,3,H,W)."""
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr)[None, ...]
    return tensor

def prepare_mask_image(mask: Image.Image) -> torch.Tensor:
    """PIL mask -> torch float tensor in [0,1], shape (1,1,H,W)."""
    mask = mask.convert("L")
    arr = np.array(mask).astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr)[None, None, ...]
    return tensor

def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    """(B,H,W,C) in [0,1] -> list of PIL"""
    if images.ndim == 3:
        images = images[None, ...]
    pil_images = []
    for img in images:
        img = (img * 255).round().astype("uint8")
        pil_images.append(Image.fromarray(img))
    return pil_images

@torch.no_grad()
def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input
