import torch
import math

from unet_blocks import CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D


def get_down_block(down_block_type, **kwargs):
    if down_block_type == "DownBlock2D":
        return DownBlock2D(**kwargs)
    elif down_block_type == "CrossAttnDownBlock2D":
        return CrossAttnDownBlock2D(**kwargs)
    else:
        raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(up_block_type, **kwargs):
    if up_block_type == "UpBlock2D":
        return UpBlock2D(**kwargs)
    elif up_block_type == "CrossAttnUpBlock2D":
        return CrossAttnUpBlock2D(**kwargs)
    else:
        raise ValueError(f"{get_up_block} does not exist.")

def get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift, scale, max_period=10000):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb