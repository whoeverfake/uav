import math
import torch
import torch.nn as nn


class LocalSpatialSelfAttention(nn.Module):
    """
    Local Spatial Self-Attention for multi-UAV coordination.

    Each UAV attends only to neighbors within a spatial radius,
    replacing global (Bahdanau-style) attention with spatially-aware local attention.

    Args:
        n_agents      : number of UAVs (M)
        obs_per_agent : observation dims per UAV (default 4: x, y, n_users, throughput)
        hidden_size   : output feature dimension
        n_heads       : number of attention heads
        radius        : spatial attention radius (same unit as UAV coordinates)
        use_orthogonal: weight init method
    Input:
        cent_obs : [batch, n_agents * obs_per_agent]  (flattened global state)
    Output:
        [batch, hidden_size]  (mean-pooled attended features)
    """

    def __init__(self, n_agents, obs_per_agent, hidden_size,
                 n_heads=1, radius=200.0, use_orthogonal=True):
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.n_agents = n_agents
        self.obs_per_agent = obs_per_agent
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.radius = radius

        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            init_method(m.weight)
            nn.init.constant_(m.bias, 0)
            return m

        # normalize per-agent features before projection: raw coords are O(500)
        # while throughput is O(1), which would otherwise dominate attention logits
        self.feature_norm = nn.LayerNorm(obs_per_agent)

        self.q = init_(nn.Linear(obs_per_agent, hidden_size))
        self.k = init_(nn.Linear(obs_per_agent, hidden_size))
        self.v = init_(nn.Linear(obs_per_agent, hidden_size))
        self.out_proj = init_(nn.Linear(hidden_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, cent_obs):
        # cent_obs: [B, n_agents * obs_per_agent]
        B = cent_obs.size(0)
        N = self.n_agents
        x = cent_obs.view(B, N, self.obs_per_agent)  # [B, N, D]

        # --- spatial mask built from RAW coordinates (before normalization) ---
        pos = x[:, :, :2]                              # [B, N, 2]  (x, y)
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)  # [B, N, N]
        local_mask = dist <= self.radius               # [B, N, N]
        # self-attention is always allowed, so no row can be fully masked
        eye = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        local_mask = local_mask | eye

        # --- normalize features, then multi-head attention ---
        h = self.feature_norm(x)                       # [B, N, D]
        Q = self.q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, N, N]

        mask = local_mask.unsqueeze(1).expand_as(attn)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)        # handle isolated agents

        out = torch.matmul(attn, V)                   # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.norm(self.out_proj(out))            # [B, N, hidden_size]

        return out.mean(dim=1)                         # [B, hidden_size]
