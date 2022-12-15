import torch

class ResnetBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.nonlinearity = torch.nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-5, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = torch.nn.Linear(temb_channels, self.out_channels)

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-5, affine=True)
        self.conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.use_conv_shortcut = self.in_channels != self.out_channels
        if self.use_conv_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        # first stack
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        # second stack
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # skip connections
        if self.use_conv_shortcut:
            input_tensor = self.conv_shortcut(input_tensor)

        # concat and return
        output_tensor = (input_tensor + hidden_states)
        return output_tensor