"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_actor_critic.py
"""

import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt
from algorithms.utils.attention import LocalSpatialSelfAttention, BahdanauCriticAttention
from algorithms.utils.attention_rnn import AttentionRNNCritic

from utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase  # 三维的用卷积，否则用全连接
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)  # 先处理一下obs

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # 将处理好的actor_features送入ACTlayer
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO with Local Spatial Self-Attention.
    Replaces MLP+RNN with a spatially-aware attention module over all agents'
    centralized observations.
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        obs_per_agent = getattr(args, 'obs_per_agent', 4)
        attn_radius = getattr(args, 'attn_radius', 200.0)
        attn_heads = getattr(args, 'attn_heads', 1)

        # infer agent count from the centralized obs so that runs with a
        # different number of UAVs work without touching the config
        assert cent_obs_shape[0] % obs_per_agent == 0, \
            f"cent_obs dim {cent_obs_shape[0]} not divisible by obs_per_agent {obs_per_agent}"
        n_agents = cent_obs_shape[0] // obs_per_agent

        # select critic attention mechanism:
        # 'local' = Local Spatial Self-Attention
        # 'bahdanau' = global additive attention
        # 'attention_rnn' = Bahdanau + GRU (spatial + temporal)
        critic_attn = getattr(args, 'critic_attn', 'local')
        self.use_rnn = (critic_attn == 'attention_rnn')
        
        if critic_attn == 'bahdanau':
            self.attention = BahdanauCriticAttention(
                n_agents=n_agents,
                obs_per_agent=obs_per_agent,
                hidden_size=self.hidden_size,
                use_orthogonal=self._use_orthogonal,
            )
        elif critic_attn == 'attention_rnn':
            self.attention = AttentionRNNCritic(
                n_agents=n_agents,
                obs_per_agent=obs_per_agent,
                hidden_size=self.hidden_size,
                num_layers=1,
                use_orthogonal=self._use_orthogonal,
            )
        else:
            self.attention = LocalSpatialSelfAttention(
                n_agents=n_agents,
                obs_per_agent=obs_per_agent,
                hidden_size=self.hidden_size,
                n_heads=attn_heads,
                radius=attn_radius,
                use_orthogonal=self._use_orthogonal,
            )


        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        :param cent_obs:   [B, n_agents * obs_per_agent]
        :param rnn_states: [num_layers, B, hidden] for attention_rnn, else unused
        :param masks:      unused (kept for API compatibility)
        :return values:    [B, 1]
        :return rnn_states: updated if attention_rnn, else unchanged
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.use_rnn:
            # AttentionRNNCritic returns (features, new_rnn_states)
            critic_features, rnn_states = self.attention(cent_obs, rnn_states)
        else:
            # Other attention modules only return features
            critic_features = self.attention(cent_obs)
        
        values = self.v_out(critic_features)

        return values, rnn_states
