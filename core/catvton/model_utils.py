import os, json, torch
from torch import nn

from .attn_processor import AttnProcessor2_0

def init_adapter(unet, cross_attn_cls, self_attn_cls=None, cross_attn_dim=None, **kwargs):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim

    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
            else:
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
        else:
            attn_procs[name] = cross_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)

    unet.set_attn_processor(attn_procs)
    adapter_modules = nn.ModuleList(unet.attn_processors.values())
    return adapter_modules

def get_trainable_module(unet, trainable_module_name):
    if trainable_module_name == "attention":
        attn_blocks = torch.nn.ModuleList()
        for name, module in unet.named_modules():
            if "attn1" in name:
                attn_blocks.append(module)
        return attn_blocks
    if trainable_module_name == "unet":
        return unet
    raise ValueError(f"Unknown trainable_module_name: {trainable_module_name}")
