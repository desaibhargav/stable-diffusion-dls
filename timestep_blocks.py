import torch

from unet_utils import get_timestep_embedding


class TimestepEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int = None):
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim)
        self.act = torch.nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb