import inspect
import os
from typing import Union

import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download

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
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)

        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)

        if compile and hasattr(torch, "compile"):
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        if use_tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt: str, version: str):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]

        if os.path.exists(attn_ckpt):
            ckpt_path = os.path.join(attn_ckpt, sub_folder, "attention")
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            ckpt_path = os.path.join(repo_path, sub_folder, "attention")

        load_checkpoint_in_model(self.attn_modules, ckpt_path)

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
        # CatVTON concatenates along Y axis (height) in latent space
        concat_dim = -2

        # Ensure PIL inputs are resized consistently
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))

        # To tensor
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
            for i, t in enumerate(timesteps):
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

        # Split back the person half
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image_out = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image_out = (image_out / 2 + 0.5).clamp(0, 1)
        image_out = image_out.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(image_out)[0]
