import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_embed(obs_per_agent, hidden_size, use_orthogonal):
    """Shared input stage for both attention variants.

    Rationale: the raw per-agent feature vector mixes coordinates O(100s) with
    n_users O(1-10) and throughput O(1). Applying LayerNorm directly on these 4
    heterogeneous dims lets the coordinates dominate mean/std and squashes the
    throughput/user signals to noise. Instead we first PROJECT to hidden space,
    then LayerNorm there, so every input feature keeps its own learnable scale.
    Both variants use the identical stage so the comparison isolates the
    aggregation mechanism (local self-attention vs. global additive attention).
    """
    init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
    embed = nn.Linear(obs_per_agent, hidden_size)
    init_method(embed.weight)
    nn.init.constant_(embed.bias, 0)
    return embed, nn.LayerNorm(hidden_size)


class LocalSpatialSelfAttention(nn.Module):
    """
    Local Spatial Self-Attention for the centralized critic.

    Each UAV attends to every other UAV, but attention logits are biased by
    spatial distance so nearby UAVs dominate. Crucially the bias is SOFT (a
    learnable distance decay) rather than a hard -inf mask: a centralized critic
    must still be able to see the global joint state to estimate the true return,
    so we emphasize locality without cutting off the global information flow that
    a hard radius mask destroys.

    Input:  cent_obs [B, n_agents * obs_per_agent]  (flattened global state)
    Output: [B, hidden_size]  (mean-pooled attended features)
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

        # project raw features into hidden space, then normalize there
        self.embed, self.embed_norm = _make_embed(obs_per_agent, hidden_size, use_orthogonal)

        self.q = init_(nn.Linear(hidden_size, hidden_size))
        self.k = init_(nn.Linear(hidden_size, hidden_size))
        self.v = init_(nn.Linear(hidden_size, hidden_size))
        self.out_proj = init_(nn.Linear(hidden_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

        # learnable locality strength (>=0 via softplus). Starts near 1 so the
        # module is local at init but can relax toward global if that helps the
        # value estimate, letting the reward curve recover instead of being
        # permanently starved of global context.
        self.log_gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, cent_obs):
        B = cent_obs.size(0)
        N = self.n_agents
        x = cent_obs.view(B, N, self.obs_per_agent)          # [B, N, D]

        # spatial distance from RAW coordinates (first two dims)
        pos = x[:, :, :2]                                     # [B, N, 2]
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)  # [B, N, N]

        # embed + normalize in hidden space
        h = self.embed_norm(self.embed(x))                   # [B, N, H]

        Q = self.q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        logits = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, N, N]

        # soft locality bias: 0 at self, smoothly negative with distance.
        # far agents are down-weighted, never hard-zeroed, so global info flows.
        gamma = F.softplus(self.log_gamma)
        bias = -gamma * (dist / self.radius) ** 2            # [B, N, N]
        logits = logits + bias.unsqueeze(1)                  # broadcast over heads

        attn = torch.softmax(logits, dim=-1)                 # no -inf -> no NaNs
        out = torch.matmul(attn, V)                          # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        # residual around the attention block, then pool over agents
        out = self.norm(h + self.out_proj(out))              # [B, N, H]
        return out.mean(dim=1)                               # [B, H]


class BahdanauCriticAttention(nn.Module):
    """
    Global additive (Bahdanau-style) attention baseline for the critic.

    A single learnable query attends over ALL agents (no spatial locality).
    Uses the identical input embedding stage as LocalSpatialSelfAttention so the
    only difference between the two runs is the aggregation mechanism.

    Input:  cent_obs [B, n_agents * obs_per_agent]
    Output: [B, hidden_size]
    """

    def __init__(self, n_agents, obs_per_agent, hidden_size, use_orthogonal=True):
        super().__init__()
        self.n_agents = n_agents
        self.obs_per_agent = obs_per_agent
        self.hidden_size = hidden_size

        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            init_method(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            return m

        # same embedding stage as the local module for a fair comparison
        self.embed, self.embed_norm = _make_embed(obs_per_agent, hidden_size, use_orthogonal)

        self.value = init_(nn.Linear(hidden_size, hidden_size))
        self.W_h = init_(nn.Linear(hidden_size, hidden_size))
        self.W_q = init_(nn.Linear(hidden_size, hidden_size, bias=False))
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.score = init_(nn.Linear(hidden_size, 1, bias=False))
        self.out_proj = init_(nn.Linear(hidden_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, cent_obs):
        B = cent_obs.size(0)
        N = self.n_agents
        x = cent_obs.view(B, N, self.obs_per_agent)          # [B, N, D]

        h = self.embed_norm(self.embed(x))                   # [B, N, H]
        # additive score: v^T tanh(W_h h_i + W_q q) -- global, no spatial bias
        q = self.W_q(self.query).view(1, 1, self.hidden_size)
        energy = torch.tanh(self.W_h(h) + q)                 # [B, N, H]
        scores = self.score(energy).squeeze(-1)              # [B, N]
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)   # [B, N, 1]

        V = self.value(h)                                    # [B, N, H]
        context = (alpha * V).sum(dim=1)                     # [B, H]
        out = self.norm(self.out_proj(context))              # [B, H]
        return out
