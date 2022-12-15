import torch

from activations import GEGLU


class CrossAttention(torch.nn.Module):
    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, cross_attention_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Linear(inner_dim, query_dim)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None):
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # attention
        hidden_states = self._attention(query, key, value)
        
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out= None, mult= 4):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.act_fn = GEGLU(dim, inner_dim)
        self.linear = torch.nn.Linear(inner_dim, dim_out)

    def forward(self, hidden_states):
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear(hidden_states)
        return hidden_states