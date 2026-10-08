import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    def __init__(
        self,
        policy_network,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coef=0.01,
        critic_coef=0.5,
    ):
        self.policy = policy_network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef

    # -------------------------------------------------------
    # ACTION SELECTION WITH MASKING
    # -------------------------------------------------------
    def choose_action(self, obs, action_mask=None):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, D)
        logits, value = self.policy(obs)

        # ✅ Apply action mask by setting invalid actions to very negative value
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32)
            # Use -1e10 instead of -inf to avoid NaN issues
            logits = logits + (mask_tensor - 1) * 1e10

        # Get probabilities and ensure they sum to 1
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)

        # IMPORTANT — detach so graph is not kept
        return (
            action.item(),
            logp.detach(),
            value.detach(),
        )

    # -------------------------------------------------------
    # RETURNS
    # -------------------------------------------------------
    def compute_returns(self, rewards):
        # terminal reward only → easy baseline
        final_R = rewards[-1]
        return torch.tensor(
            [final_R] * len(rewards), dtype=torch.float32
        )

    # -------------------------------------------------------
    # UPDATE POLICY
    # -------------------------------------------------------
    def update(self, obs_list, act_list, old_logps, values, rewards):
        # Detach everything BEFORE using it
        old_logps = torch.stack(old_logps).detach()
        values = torch.stack(values).detach().squeeze()

        # compute returns and advantage
        returns = self.compute_returns(rewards)
        advantages = returns - values

        # convert observations to a clean tensor
        obs_tensor = torch.tensor(
            obs_list, dtype=torch.float32
        )  # shape (T, D)

        # PPO update loop
        for _ in range(4):  # number of PPO epochs
            logits, new_values = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            new_logps = dist.log_prob(torch.tensor(act_list))

            # PPO ratio
            ratios = torch.exp(new_logps - old_logps)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - new_values.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = (
                policy_loss
                + self.critic_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.optimizer.step()