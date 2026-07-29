import torch
import torch.nn as nn
from .attention import BahdanauCriticAttention


class AttentionRNNCritic(nn.Module):
    """
    Attention-RNN Hybrid Critic: Bahdanau for spatial + GRU for temporal.
    
    Motivation: Pure Bahdanau (976.65) lacks temporal memory. This hybrid
    combines:
    - Spatial dimension: Bahdanau attention aggregates multi-agent interactions 
      at current timestep
    - Temporal dimension: GRU models dynamics across timesteps (UAV trajectories,
      channel switching patterns, interference accumulation, etc.)
    
    Theoretical advantage over pure Bahdanau: captures both spatial (multi-agent)
    and temporal (sequential) dependencies, similar to Spatial-Temporal Attention
    in video understanding.
    
    Input:  cent_obs [B, n_agents * obs_per_agent]
            rnn_states [num_layers, B, hidden_size] or None
    Output: features [B, hidden_size]
            new_rnn_states [num_layers, B, hidden_size]
    """

    def __init__(self, n_agents, obs_per_agent, hidden_size,
                 num_layers=1, use_orthogonal=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stage 1: Bahdanau attention for spatial aggregation
        self.attention = BahdanauCriticAttention(
            n_agents, obs_per_agent, hidden_size, use_orthogonal
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
            rnn_states: [num_layers, B, hidden_size] or None
        Returns:
            output: [B, hidden_size]
            new_rnn_states: [num_layers, B, hidden_size]
        """
        B = cent_obs.size(0)

        # [1] Spatial: Attention aggregates multi-agent info at current timestep
        attn_out = self.attention(cent_obs)  # [B, hidden_size]

        # [2] Temporal: RNN models dynamics across timesteps
        attn_out = attn_out.unsqueeze(1)  # [B, 1, hidden_size] for RNN input
        
        if rnn_states is None:
            rnn_states = torch.zeros(
                self.num_layers, B, self.hidden_size,
                device=cent_obs.device, dtype=cent_obs.dtype
            )
        
        rnn_out, new_rnn_states = self.rnn(attn_out, rnn_states)
        rnn_out = rnn_out.squeeze(1)  # [B, hidden_size]

        return rnn_out, new_rnn_states
