import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Takes a pooled WSI state vector (size D_FEATURE)
    Returns:
        - logits over possible actions
        - value function V(s)
    """

    def __init__(self, input_dim=1024, hidden_dim=256, action_dim=8000):
        super().__init__()

        # Shared body
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch, input_dim)
        returns: (logits, value)
        """

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        logits = self.policy_head(h)
        value = self.value_head(h)

        return logits, value