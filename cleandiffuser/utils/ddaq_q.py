import torch
import torch.nn as nn


class DDAQCritic(nn.Module):
    """
    Args:
        obs_dim: int,
            The dimension of the observation space.
        act_dim: int,
            The dimension of the action space.
        hidden_dim: int,
            The dimension of the hidden layers.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_model(x)

    def q_min(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)
