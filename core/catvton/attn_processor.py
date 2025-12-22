import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal versions of the attention processors used by CatVTON.
# Reference implementation originates from the CatVTON HF Space.

class AttnProcessor2_0(nn.Module):
    def __init__(self, hidden_size=None, cross_attention_dim=None, **kwargs):
        super().__init__()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class SkipAttnProcessor(AttnProcessor2_0):
    """
    CatVTON runs without text conditioning, but UNet cross-attn layers still
    expect encoder_hidden_states with dim=768. If encoder_hidden_states is None,
    diffusers will treat it as self-attn and crash (320 vs 768).

    Fix: supply a 'null' encoder_hidden_states of the correct shape/dim so
    cross-attn executes safely.
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):

        if encoder_hidden_states is None:
            # batch size is always first dim (3D or 4D)
            batch_size = hidden_states.shape[0]

            # typical CLIP seq length
            seq_len = 77

            # cross-attn expects 768 input features (usually attn.to_k.in_features)
            cross_dim = attn.to_k.in_features

            encoder_hidden_states = torch.zeros(
                (batch_size, seq_len, cross_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **kwargs
        )

