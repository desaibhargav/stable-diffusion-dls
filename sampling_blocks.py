import torch
from residual_block import ResnetBlock2D

class Upsample2D(torch.nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name
        
        if use_conv:
            self.conv = torch.nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size):
        if output_size is None:
            hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = torch.nn.functional.interpolate(hidden_states, size=output_size, mode="nearest")
            
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states

class Downsample2D(torch.nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name

        if use_conv:
            self.conv = torch.nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.conv = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states