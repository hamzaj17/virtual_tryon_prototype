import inspect
import os

import numpy as np
import torch
import tqdm

from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download

from .attn_processor import SkipAttnProcessor
from .model_utils import get_trainable_module, init_adapter
from ..image_utils import (
    compute_vae_encodings,
    numpy_to_pil,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)

def crop_garment_foreground(img, bg_thresh=245, pad=12):
    """
    Crops garment to foreground so condition_image isn't mostly white.
    Works for:
      - PNG with alpha (uses alpha mask)
      - JPG/PNG on near-white background (uses threshold)
    """
    im = img.convert("RGBA")
    arr = np.array(im)  # (H,W,4)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # If alpha has transparency, use it
    if alpha.min() < 250:
        mask = alpha > 10
    else:
        # Otherwise treat near-white as background
        mask = np.any(rgb < bg_thresh, axis=2)

    if not mask.any():
        return img

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    h, w = mask.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)

    return im.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGB")


class CatVTONPipeline:
    def __init__(
        self,
        base_ckpt: str,
        attn_ckpt: str,
        attn_ckpt_version: str = "mix",
        weight_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        compile: bool = False,
        use_tf32: bool = True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            ).to(device, dtype=weight_dtype)
            
        self.unet = UNet2DConditionModel.from_pretrained(
            base_ckpt,
            subfolder="unet",
            low_cpu_mem_usage=True,
            use_safetensors=False,  # base_ckpt may fall back to .bin anyway
            ).to(device, dtype=weight_dtype)

        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")

        # ✅ downloads ONLY the attention file now
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)

        if compile and hasattr(torch, "compile"):
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        if use_tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt: str, version: str):
        """
        Loads ONLY the CatVTON attention checkpoint:
          <sub_folder>/attention/model.safetensors

        attn_ckpt can be:
          - HF repo id like "zhengchong/CatVTON"
          - Local folder containing subfolders
          - Direct local file path to model.safetensors
        """
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]

        # ✅ HF filenames must use forward slashes (POSIX), NOT os.path.join() on Windows
        rel_file_hf = f"{sub_folder}/attention/model.safetensors"

        # Local filesystem can use os.path.join()
        rel_file_local = os.path.join(sub_folder, "attention", "model.safetensors")

        if os.path.exists(attn_ckpt):
            # local file or local folder
            if os.path.isfile(attn_ckpt):
                ckpt_file = attn_ckpt
            else:
                ckpt_file = os.path.join(attn_ckpt, rel_file_local)

            if not os.path.exists(ckpt_file):
                raise FileNotFoundError(
                    f"CatVTON attention checkpoint not found.\nExpected: {ckpt_file}"
                )
        else:
            # ✅ downloads only the single file we need
            ckpt_file = hf_hub_download(
                repo_id=attn_ckpt,
                filename=rel_file_hf,
            )

        # Load weights into attention modules only
        load_checkpoint_in_model(self.attn_modules, ckpt_file)

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image,
        condition_image,
        mask,
        num_inference_steps: int = 35,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta: float = 1.0,
    ):
        concat_dim = -2

        # Your pipeline expects mask and image to be same size before resize/crop
        assert image.size == mask.size, "Image and mask must have the same size"

        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = crop_garment_foreground(condition_image)
        condition_image = resize_and_padding(condition_image, (width, height))

        image_t = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        cond_t = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask_t = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)

        masked_image = image_t * (mask_t < 0.5)

        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(cond_t, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask_t, size=masked_latent.shape[-2:], mode="nearest")

        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for _, t in enumerate(timesteps):
                latent_in = torch.cat([latents] * 2) if do_cfg else latents
                latent_in = self.noise_scheduler.scale_model_input(latent_in, t)

                inpaint_in = torch.cat([latent_in, mask_latent_concat, masked_latent_concat], dim=1)
                noise_pred = self.unet(
                    inpaint_in,
                    t.to(self.device),
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]

                if do_cfg:
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + guidance_scale * (cond - uncond)

                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                progress_bar.update(1)

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image_out = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image_out = (image_out / 2 + 0.5).clamp(0, 1)
        image_out = image_out.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(image_out)[0]
