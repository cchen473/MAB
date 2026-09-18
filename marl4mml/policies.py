import torch
import torch.nn as nn


class GRUActor(nn.Module):
    """15 interpretable stats -> GRU(64) -> 5 action probabilities."""

    def __init__(self, obs_dim=15, hidden=64, n_actions=5):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(obs_dim, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs, h):
        h = self.cell(obs, h)
        return self.head(h), h

    def init_h(self, batch=1, device="cpu"):
        return torch.zeros(batch, self.hidden, device=device)


class CentralCritic(nn.Module):
    """Central history -> Q values for the full 5x5 joint action space."""

    def __init__(self, central_dim=32, hidden=128, n_joint=25):
        super().__init__()
        self.hidden = hidden
        self.n_joint = n_joint
        self.cell = nn.GRUCell(central_dim, hidden)
        self.head = nn.Linear(hidden, n_joint)

    def forward(self, central_obs, h):
        h = self.cell(central_obs, h)
        return self.head(h), h

    def init_h(self, batch=1, device="cpu"):
        return torch.zeros(batch, self.hidden, device=device)


def coma_advantages(q25, pi1, pi2, a1, a2):
    """q25: (25,) joint-action Q; pi1/pi2: (5,) action probs; a1/a2: taken actions.

    A_i = Q(H, k_i, k_j) - sum_a pi_i(a) Q(H, a, k_j)   (counterfactual baseline)
    """
    Q = q25.view(5, 5)
    q_taken = Q[a1, a2]
    b1 = (pi1 * Q[:, a2]).sum()
    b2 = (pi2 * Q[a1, :]).sum()
    return q_taken - b1, q_taken - b2
