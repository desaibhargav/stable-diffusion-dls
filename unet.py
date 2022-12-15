import torch

from resnet_block import ResnetBlock2D
from transformer_blocks import Transformer2DModel
from timestep_blocks import Timesteps, TimestepEmbedding
from unet_utils import get_down_block, get_up_block


class UNet2DConditionModel(torch.nn.Module):
    def __init__(self, sample_size=None, in_channels=4, out_channels=4, freq_shift=0, layers_per_block=2):
        super().__init__()
        self.down_block_types = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
        self.up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
        self.block_out_channels = (320, 640, 1280)

        self.sample_size = sample_size
        time_embed_dim = self.block_out_channels[0] * 3

        # input
        self.conv_in = torch.nn.Conv2d(in_channels, self.block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(self.block_out_channels[0], freq_shift)
        timestep_input_dim = self.block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = torch.nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = torch.nn.ModuleList([])

        attention_head_dim = (8,) * len(self.down_block_types)

        # down
        output_channel = self.block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = self.block_out_channels[i]
            is_final_block = i == len(self.block_out_channels) - 1
            down_block = get_down_block(down_block_type, 
                                        num_layers=layers_per_block, 
                                        in_channels=input_channel, 
                                        out_channels=output_channel, 
                                        temb_channels=time_embed_dim, 
                                        add_downsample=not is_final_block)
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(in_channels=self.block_out_channels[-1], temb_channels=time_embed_dim)

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.up_block_types):
            is_final_block = i == len(self.block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(self.block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(up_block_type, 
                                    num_layers=layers_per_block + 1, 
                                    in_channels=input_channel, 
                                    out_channels=output_channel, 
                                    prev_output_channel=prev_output_channel, 
                                    temb_channels=time_embed_dim, 
                                    add_upsample=add_upsample)

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=self.block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(self.block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            if isinstance(timestep, float):
                dtype = torch.float32
            else:
                dtype = torch.int32
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        # t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class UNetMidBlock2DCrossAttn(torch.nn.Module):
    def __init__(self, in_channels: int, temb_channels: int):
        super().__init__()
        self.has_cross_attention = True
        self.attn_num_head_channels = 8
        self.resnet_1 = ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels)
        self.resnet_2 = ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels)
        self.attention = Transformer2DModel(self.attn_num_head_channels, in_channels // self.attn_num_head_channels, in_channels=in_channels)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, encoder_hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states