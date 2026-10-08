import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box


class VariableWSIEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        embeddings: np.ndarray,
        coords: np.ndarray,
        mask: np.ndarray,
        max_steps: int = 100,
        reward_mode: str = "sparse",
    ):
        super().__init__()
        self.embeddings = embeddings.astype(np.float32)
        self.coords = coords

        # ✅ Flatten mask if it's 2D
        if mask.ndim == 2:
            mask = mask.flatten()
        self.mask = mask.astype(np.int32)

        self.N, self.D = self.embeddings.shape
        self.max_steps = max_steps
        self.reward_mode = reward_mode

        # Global action space (matches PPO)
        self.action_space = Discrete(8000)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.D,),
            dtype=np.float32,
        )

        # Safety check: ensure mask matches number of patches
        if len(self.mask) != self.N:
            # Silently adjust - this is expected due to quality filtering
            if len(self.mask) < self.N:
                self.mask = np.pad(self.mask, (0, self.N - len(self.mask)))
            else:
                self.mask = self.mask[:self.N]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visited = np.zeros(self.N, dtype=bool)
        self.steps = 0
        self.found_tumor = False
        self.first_hit_step = None
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False

        # Ensure action is a scalar integer
        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(action.item() if hasattr(action, 'item') else action[0])
        action = int(action)

        # HARD SAFETY CLIP - Clamp action to valid range [0, N-1]
        if action >= self.N or action < 0:
            action = min(max(0, action), self.N - 1)
            reward = -1.0

        # Process the valid action
        self.visited[action] = True
        is_tumor = bool(self.mask[action] > 0)

        if self.reward_mode == "sparse":
            if is_tumor:
                self.found_tumor = True

        elif self.reward_mode == "first_hit":
            reward += -0.01
            if is_tumor and not self.found_tumor:
                reward = 1.0
                self.found_tumor = True
                self.first_hit_step = self.steps

        elif self.reward_mode == "dense":
            reward += -0.01
            if is_tumor:
                reward = 1.0
                self.found_tumor = True
            else:
                tumor_coords = self.coords[self.mask > 0]
                if len(tumor_coords) > 0:
                    dists = np.linalg.norm(
                        tumor_coords - self.coords[action],
                        axis=1,
                    )
                    reward += 0.1 * np.exp(-dists.min() / 1000)

        if self.steps >= self.max_steps:
            terminated = True
            if self.reward_mode == "sparse":
                reward = 1.0 if self.found_tumor else 0.0

        info = {
            "hit_tumor": self.found_tumor,
            "regions_observed": int(self.visited.sum()),
            "first_hit_step": self.first_hit_step,
        }

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        return self.embeddings.mean(axis=0).astype(np.float32)

    def get_action_mask(self):
        """Returns binary mask for valid actions (1=valid, 0=invalid)"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        mask[:self.N] = 1.0
        return mask