# RLogist/train_camelyon.py
import os
import argparse
import torch
import numpy as np
from RLogist.ppo_agent import PPOAgent
from RLogist.policy_network import PolicyNetwork
from RLogist.rl.env_variable_wsi import VariableWSIEnv
from RLogist.rl.env_variable_wsi_dense import VariableWSIEnvDense


def load_embeddings(slide):
    emb = np.load(f"embeddings/{slide}.npy")
    coords = np.load(f"embeddings/{slide}_coords.npy")
    mask = np.load(f"embeddings/{slide}_mask.npy")
    return emb, coords, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["first_hit", "dense"], required=True)
    args = parser.parse_args()

    EnvClass = VariableWSIEnv if args.reward == "first_hit" else VariableWSIEnvDense

    policy = PolicyNetwork(input_dim=384, hidden_dim=256, action_dim=8000)
    agent = PPOAgent(policy)

    slides = ["tumor_005.tif", "tumor_012.tif"]  # small subset for now

    for episode in range(200):
        slide = np.random.choice(slides)
        emb, coords, mask = load_embeddings(slide)

        env = EnvClass(embeddings=emb, coords=coords, mask=mask, max_steps=100)
        obs, _ = env.reset()

        obs_list, act_list, logp_list, value_list, rewards = [], [], [], [], []
        done = False

        while not done:
            # ✅ GET ACTION MASK FROM ENVIRONMENT
            action_mask = env.get_action_mask()

            # ✅ PASS MASK TO AGENT
            action, logp, value = agent.choose_action(obs, action_mask=action_mask)

            # 🔍 DEBUG: Check if action is valid
            if action >= env.N:
                print(f"⚠️  WARNING: Invalid action {action} (N={env.N})")
                print(f"   Action mask sum: {action_mask.sum()}")
                print(f"   Valid actions: 0-{env.N-1}")

            obs2, reward, done, _, _ = env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp)
            value_list.append(value)
            rewards.append(reward)

            obs = obs2

        agent.update(obs_list, act_list, logp_list, value_list, rewards)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/200 - Slide: {slide} - Reward: {sum(rewards):.3f}")

    os.makedirs("RLogist/checkpoints", exist_ok=True)
    torch.save(
        policy.state_dict(),
        f"RLogist/checkpoints/ppo_camelyon_{args.reward}.pt"
    )
    print(f"✅ Training complete! Model saved to RLogist/checkpoints/ppo_camelyon_{args.reward}.pt")


if __name__ == "__main__":
    main()