import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box


class VariableWSIEnvDense(gym.Env):
    """
    Variant 2: Dense reward with tumor proximity shaping.
    """

    def __init__(self, embeddings, coords, mask, max_steps=100):
        super().__init__()
        self.embeddings = embeddings.astype(np.float32)
        self.coords = coords

        # ✅ Flatten mask if it's 2D
        if mask.ndim == 2:
            mask = mask.flatten()
        self.mask = mask.astype(np.int32)

        self.max_steps = max_steps
        self.N, self.D = embeddings.shape

        # ✅ Global action space (matches PPO - same as other env)
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
        obs = self.embeddings.mean(axis=0).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        reward = -0.01  # small step penalty

        # Ensure action is a scalar integer
        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(action.item() if hasattr(action, 'item') else action[0])
        action = int(action)

        # ✅ HARD SAFETY CLIP - Clamp action to valid range [0, N-1]
        if action >= self.N or action < 0:
            action = min(max(0, action), self.N - 1)
            reward = -1.0

        if self.visited[action]:
            reward -= 0.02  # revisiting penalty

        self.visited[action] = True

        # tumor logic
        is_tumor = bool(self.mask[action] == 1)
        if is_tumor and not self.found_tumor:
            reward += 1.0
            self.found_tumor = True

        # proximity reward (distance-based)
        tumor_indices = np.where(self.mask == 1)[0]
        if len(tumor_indices) > 0:
            dists = np.linalg.norm(
                self.coords[tumor_indices] - self.coords[action],
                axis=1,
            )
            min_dist = dists.min()
            reward += np.exp(-min_dist / 200.0) * 0.1

        terminated = self.steps >= self.max_steps
        if terminated and not self.found_tumor:
            reward -= 0.5

        info = {
            "hit_tumor": self.found_tumor,
            "regions_observed": int(self.visited.sum()),
        }

        return self._obs(), reward, terminated, False, info

    def _obs(self):
        return self.embeddings.mean(axis=0).astype(np.float32)

    def get_action_mask(self):
        """Returns binary mask for valid actions (1=valid, 0=invalid)"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        mask[:self.N] = 1.0
        return mask