import torch

from resnet_block import ResnetBlock2D
from transformer_blocks import Transformer2DModel
from sampling_blocks import Downsample2D, Upsample2D

class CrossAttnDownBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers=1, add_downsample=True):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = 8

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels))
            attentions.append(Transformer2DModel(num_attention_heads=self.attn_num_head_channels, 
                                                 attention_head_dim=(out_channels // self.attn_num_head_channels), 
                                                 in_channels=out_channels))
           
        self.attentions = torch.nn.ModuleList(attentions)
        self.resnets = torch.nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler = Downsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.downsampler = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()
        for (resnet, attn) in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states += (hidden_states,)


        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)

            output_states += (hidden_states,)
        
        return hidden_states, output_states


class CrossAttnUpBlock2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, prev_output_channel: int, temb_channels: int, num_layers: int = 1, add_upsample=True):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = 8

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels))
            attentions.append(Transformer2DModel(self.attn_num_head_channels, out_channels // self.attn_num_head_channels, in_channels=out_channels))
            
        self.attentions = torch.nn.ModuleList(attentions)
        self.resnets = torch.nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsampler = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, upsample_size=None):
        for (resnet, attn) in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # residual
            hidden_states = resnet(hidden_states, temb)

            # attention
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, upsample_size)

        return hidden_states


class DownBlock2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, num_layers: int = 1, add_downsample=True):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels))

        self.resnets = torch.nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler = Downsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.downsampler = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsampler:
            hidden_states = self.downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class UpBlock2D(torch.nn.Module):
    def __init__(self, in_channels: int, prev_output_channel: int, out_channels: int, temb_channels: int, num_layers: int = 1, add_upsample=True):
        super().__init__()
        self.add_upsample = add_upsample
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels))

        self.resnets = torch.nn.ModuleList(resnets)

        if self.add_upsample:
            self.upsampler = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.add_upsample:
            hidden_states = self.upsampler(hidden_states, upsample_size)

        return hidden_states