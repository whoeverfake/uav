import torch
import torch.nn as nn
from .attention import LocalSpatialSelfAttention


class LocalAttentionRNNCritic(nn.Module):
    """
    Local Spatial Self-Attention + RNN Hybrid Critic.
    
    Combines:
    - Spatial dimension: Local Self-Attention with learnable gamma (spatial structure)
    - Temporal dimension: GRU models dynamics across timesteps
    
    This is the original intended innovation: replacing Bahdanau with Local
    Self-Attention in the RNN-based critic.
    
    Input:  cent_obs [B, n_agents * obs_per_agent]
            rnn_states [B, num_layers, hidden_size] or None
    Output: features [B, hidden_size]
            new_rnn_states [B, num_layers, hidden_size]
    """

    def __init__(self, n_agents, obs_per_agent, hidden_size,
                 n_heads=1, radius=200.0, num_layers=1, use_orthogonal=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stage 1: Local Spatial Self-Attention for spatial aggregation
        self.attention = LocalSpatialSelfAttention(
            n_agents=n_agents,
            obs_per_agent=obs_per_agent,
            hidden_size=hidden_size,
            n_heads=n_heads,
            radius=radius,
            use_orthogonal=use_orthogonal
        )

        # Stage 2: GRU for temporal modeling
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, cent_obs, rnn_states=None):
        """
        Args:
            cent_obs: [B, n_agents * obs_per_agent]
            rnn_states: [B, num_layers, hidden_size] (MAPPO format) or None
        Returns:
            output: [B, hidden_size]
            new_rnn_states: [B, num_layers, hidden_size]
        """
        B = cent_obs.size(0)

        # [1] Spatial: Local Self-Attention with spatial structure
        attn_out = self.attention(cent_obs)  # [B, hidden_size]

        # [2] Temporal: RNN models dynamics
        attn_out = attn_out.unsqueeze(1)  # [B, 1, hidden_size]
        
        if rnn_states is None or rnn_states.shape[0] != B:
            rnn_states_gru = torch.zeros(
                self.num_layers, B, self.hidden_size,
                device=cent_obs.device, dtype=cent_obs.dtype
            )
        else:
            rnn_states_gru = rnn_states.transpose(0, 1).contiguous()
        
        rnn_out, new_rnn_states_gru = self.rnn(attn_out, rnn_states_gru)
        rnn_out = rnn_out.squeeze(1)
        
        new_rnn_states = new_rnn_states_gru.transpose(0, 1).contiguous()

        return rnn_out, new_rnn_states
