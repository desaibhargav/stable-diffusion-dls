import torch

from attention_blocks import CrossAttention, FeedForward


class Transformer2DModel(torch.nn.Module):
    def __init__(self, num_attention_heads=16, attention_head_dim=88, in_channels=None):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = torch.nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # 3. Define transformers blocks
        self.transformer_block = BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim)

        # 4. Define output layers
        self.proj_out = torch.nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, hidden_states, encoder_hidden_states=None):
        # 1. Input
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)

        # 2. Blocks
        hidden_states = self.transformer_block(hidden_states, context=encoder_hidden_states)

        # 3. Output
        hidden_states = (hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous())
        hidden_states = self.proj_out(hidden_states)    
        output = hidden_states + residual
        return output


class BasicTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()
        # 1. Self-Attn
        self.attn1 = CrossAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim)
        self.norm1 = torch.nn.LayerNorm(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, cross_attention_dim=768)  
        self.norm2 = torch.nn.LayerNorm(dim)

        # 3. Feed-forward
        self.ff = FeedForward(dim)
        self.norm3 = torch.nn.LayerNorm(dim)

    def forward(self, hidden_states, context=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states