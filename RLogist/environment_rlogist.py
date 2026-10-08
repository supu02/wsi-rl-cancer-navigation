import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, Box
from typing import Optional

# --------------------
# Constants
# --------------------
D_FEATURE = 384     # number of features per region (embedding size)
N_MAX = 8000         # maximum number of regions allowed in a slide
K_SUBPATCHES = 16    # number of high-magnification subpatches per region


# --------------------
# Simple placeholder models
# --------------------
def placeholder_f_local(subpatch_features: torch.Tensor) -> torch.Tensor:
    """
    f_local: aggregates K subpatch features -> 1 region feature.
    """
    return subpatch_features.mean(dim=0)


def placeholder_f_global(
    v_i: torch.Tensor,
    v_a: torch.Tensor,
    v_a_prime: torch.Tensor
) -> torch.Tensor:
    """
    f_global: updates an unobserved region feature.
    """
    return (v_i + v_a_prime) / 2.0


# --------------------
# Environment
# --------------------
class RLogistWSIEnv(gym.Env):
    """
    RLogist-style WSI navigation environment
    with configurable reward variants (Sparse, Dense, First Hit).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        pre_extracted_features: dict,
        max_steps: int = 10,
        reward_mode: str = "sparse",
    ):
        super().__init__()

        # ---- Reward mode ----
        self.reward_mode = reward_mode

        # ---- Unpack data ----
        self.N_regions = pre_extracted_features["N_regions"]
        self.initial_features = pre_extracted_features["initial_features"]    # (N, D)
        self.high_mag_features = pre_extracted_features["high_mag_features"]  # (N, K, D)
        self.slide_label = int(pre_extracted_features["slide_label"])         # 0 or 1
        self.region_labels = pre_extracted_features["region_labels"]          # (N,), 0/1

        self.max_steps = max_steps

        # ---- Observation & action spaces ----
        # State: 1D feature vector of the whole slide (pooled) or similar representation
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(D_FEATURE,),
            dtype=np.float32,
        )

        self.action_space = Discrete(self.N_regions)

        # ---- Internal state ----
        self.current_feature_map = None
        self.observed_mask = None
        self.step_count = 0

    # --------------------
    # Helper: build observation
    # --------------------
    def _get_obs(self):
        # Return the mean of the current feature map as the observation
        pooled = self.current_feature_map.mean(dim=0)
        return pooled.cpu().numpy()

    # --------------------
    # Gym: reset
    # --------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_feature_map = self.initial_features.clone()
        self.observed_mask = torch.zeros(self.N_regions, dtype=torch.bool)
        self.step_count = 0

        return self._get_obs(), {}

    # --------------------
    # Reward computation (ALL VARIANTS)
    # --------------------
    def _compute_reward(self) -> float:
        visited_cancer = (
            (self.observed_mask) & (self.region_labels == 1)
        ).any().item()

        # -------------------------
        # Original sparse reward
        # -------------------------
        if self.reward_mode == "sparse":
            if self.slide_label == 1:
                return 1.0 if visited_cancer else -1.0
            else:
                return 1.0 if not visited_cancer else -1.0

        # -------------------------
        # First-hit reward
        # -------------------------
        elif self.reward_mode == "first_hit":
            # Reward 1.0 immediately if we found cancer, else 0.0
            return 1.0 if visited_cancer else 0.0

        # -------------------------
        # Dense reward
        # -------------------------
        elif self.reward_mode == "dense":
            # Small reward for every action taken (exploration)
            # You can adjust this factor (0.05) or logic as needed
            reward = 0.05

            # Big bonus if we have currently found cancer
            if visited_cancer:
                reward += 1.0
            return reward

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    # --------------------
    # Gym: step
    # --------------------
    def step(self, action: int):
        # 1. Handle Invalid action
        if action >= self.N_regions:
            return self._get_obs(), -0.1, False, False, {"invalid_action": True}

        self.step_count += 1

        # 2. Apply f_local (Update chosen region)
        v_a_prime = placeholder_f_local(self.high_mag_features[action])
        v_a_old = self.current_feature_map[action].clone()

        self.current_feature_map[action] = v_a_prime
        self.observed_mask[action] = True

        # 3. Apply f_global (Update unobserved regions)
        for i in torch.where(~self.observed_mask)[0]:
            self.current_feature_map[i] = placeholder_f_global(
                self.current_feature_map[i],
                v_a_old,
                v_a_prime,
            )

        # 4. Check for Cancer (Critical for termination logic)
        visited_cancer = (
            (self.observed_mask) & (self.region_labels == 1)
        ).any().item()

        # 5. Handle Termination
        terminated = False

        # Condition A: Max steps reached
        if self.step_count >= self.max_steps:
            terminated = True

        # Condition B: Early Stopping for "first_hit" mode
        # If we are on a positive slide and found cancer, stop immediately.
        if self.reward_mode == "first_hit" and self.slide_label == 1 and visited_cancer:
            terminated = True

        truncated = False

        # 6. Handle Reward
        if self.reward_mode == "dense":
            # Always calculate reward (gets small step reward + big cancer bonus)
            reward = self._compute_reward()

        elif self.reward_mode == "first_hit":
            # Give reward immediately if found (matches termination logic)
            reward = self._compute_reward()

        else: # "sparse" (Default)
            # Only give reward at the very end. 0.0 otherwise.
            reward = self._compute_reward() if terminated else 0.0

        # 7. Info for Metrics
        info = {
            "regions_observed": int(self.observed_mask.sum().item()),
            "visited_cancer": bool(visited_cancer),
            "slide_label": self.slide_label,
            # Helper for sensitivity/specificity calculation:
            "success": (visited_cancer if self.slide_label == 1 else not visited_cancer)
        }

        return self._get_obs(), reward, terminated, truncated, info


# --------------------
# Synthetic data generator (UNCHANGED)
# --------------------
def generate_demo_features(
    N_regions: int,
    D: int,
    K: int,
    slide_label: Optional[int] = None,
    n_cancer_regions: int = 5,
):
    torch.manual_seed(0)

    normal_center = torch.zeros(D)
    cancer_center = torch.zeros(D)
    cancer_center[0] = 3.0

    if slide_label is None:
        slide_label = int(torch.rand(()) > 0.5)

    region_labels = torch.zeros(N_regions, dtype=torch.long)

    if slide_label == 1:
        cancer_indices = torch.randperm(N_regions)[:n_cancer_regions]
        region_labels[cancer_indices] = 1

    initial = torch.zeros(N_regions, D)
    highmag = torch.zeros(N_regions, K, D)

    for i in range(N_regions):
        center = cancer_center if region_labels[i] == 1 else normal_center
        initial[i] = center + 0.5 * torch.randn(D)
        highmag[i] = center + 0.5 * torch.randn(K, D)

    return {
        "N_regions": N_regions,
        "initial_features": initial,
        "high_mag_features": highmag,
        "slide_label": slide_label,
        "region_labels": region_labels,
    }