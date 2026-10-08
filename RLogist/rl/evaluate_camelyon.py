import os
import json
import argparse
import torch
import numpy as np

from RLogist.ppo_agent import PPOAgent
from RLogist.policy_network import PolicyNetwork
from RLogist.rl.env_variable_wsi import VariableWSIEnv
from RLogist.rl.env_variable_wsi_dense import VariableWSIEnvDense

print("[evaluate_camelyon] Script started")


# Fixed evaluation slides
EVAL_SLIDES = [
    "tumor_005.tif",
    "tumor_012.tif",
    "tumor_024.tif",
    "tumor_031.tif",
    "tumor_042.tif",
    "normal_014.tif",
    "normal_037.tif",
    "normal_058.tif",
    "normal_074.tif",
    "normal_093.tif",
]


def load_embeddings(slide):
    """Load pre-extracted embeddings for a slide"""
    emb = np.load(f"embeddings/{slide}.npy")
    coords = np.load(f"embeddings/{slide}_coords.npy")
    mask = np.load(f"embeddings/{slide}_mask.npy")
    return emb, coords, mask


def main():
    print("[evaluate_camelyon] Entered main()")

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward",
        type=str,
        default="first_hit",
        choices=["first_hit", "dense"],
        help="Reward variant checkpoint to evaluate",
    )
    parser.add_argument(
        "--episodes-per-slide",
        type=int,
        default=20,
        help="Number of evaluation episodes per slide",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per episode",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint path
    ckpt_path = f"RLogist/checkpoints/ppo_camelyon_{args.reward}.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # Output directory
    out_dir = f"results/camelyon_{args.reward}"

    # Build policy with correct input dimension (384 for Camelyon)
    policy = PolicyNetwork(
        input_dim=384,      # Camelyon embeddings dimension
        hidden_dim=256,
        action_dim=8000,
    ).to(device)

    agent = PPOAgent(policy_network=policy)

    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"[evaluate_camelyon] Loaded checkpoint: {ckpt_path}")

    # Select environment class
    EnvClass = VariableWSIEnv if args.reward == "first_hit" else VariableWSIEnvDense

    # Evaluation loop
    logs = []

    for slide in EVAL_SLIDES:
        # Check if embeddings exist
        emb_path = f"embeddings/{slide}.npy"
        if not os.path.exists(emb_path):
            print(f"[WARNING] Skipping {slide} - embeddings not found")
            continue

        for ep in range(args.episodes_per_slide):
            # Load slide embeddings
            emb, coords, mask = load_embeddings(slide)

            # Debug info
            tumor_patches = np.sum(mask > 0) if mask.ndim == 1 else np.sum(mask.flatten() > 0)
            if ep == 0:
                print(f"  [INFO] Slide has {emb.shape[0]} patches, {tumor_patches} tumor patches")

            # Create environment
            env = EnvClass(
                embeddings=emb,
                coords=coords,
                mask=mask,
                max_steps=args.max_steps,
            )

            obs, _ = env.reset()

            terminated = False
            truncated = False
            steps = 0
            total_reward = 0.0
            success = False
            steps_to_hit = None

            episode_rewards = []  # Track rewards per step

            while not (terminated or truncated):
                # Get action mask and choose action
                action_mask = env.get_action_mask()
                action, _, _ = agent.choose_action(obs, action_mask=action_mask)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                episode_rewards.append(float(reward))
                steps += 1

                # Track first tumor hit
                if info is not None:
                    hit = bool(info.get("hit_tumor", False))
                    if hit and not success:
                        success = True
                        steps_to_hit = steps

            # Debug: print first episode details
            if ep == 0:
                print(f"  [DEBUG] First episode: steps={steps}, total_reward={total_reward:.3f}, "
                      f"success={success}, rewards_range=[{min(episode_rewards):.3f}, {max(episode_rewards):.3f}]")

            logs.append({
                "slide": slide,
                "episode": ep,
                "reward_variant": args.reward,
                "return": float(total_reward),
                "steps": int(steps),
                "success": bool(success),
                "steps_to_hit": steps_to_hit,
            })

        # Progress per slide
        slide_returns = [x["return"] for x in logs if x["slide"] == slide]
        slide_success = [x["success"] for x in logs if x["slide"] == slide]
        print(
            f"[eval] slide={slide} "
            f"episodes={len(slide_returns)} "
            f"mean_return={np.mean(slide_returns):.3f} "
            f"success_rate={np.mean(slide_success):.3f}"
        )

    # Compute metrics
    successes = [x["success"] for x in logs]
    returns = [x["return"] for x in logs]
    lengths = [x["steps"] for x in logs]
    hit_steps = [x["steps_to_hit"] for x in logs if x["steps_to_hit"] is not None]

    metrics = {
        "reward_variant": args.reward,
        "checkpoint": ckpt_path,
        "num_episodes": int(len(logs)),
        "episodes_per_slide": int(args.episodes_per_slide),
        "max_steps": int(args.max_steps),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_steps": float(np.mean(lengths)) if lengths else 0.0,
        "mean_steps_to_hit": float(np.mean(hit_steps)) if hit_steps else None,
    }

    # Save results
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "eval_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[evaluate_camelyon] Evaluation complete")
    print("=" * 70)
    print(f"Results saved to: {out_dir}/")
    print(f"Total episodes evaluated: {len(logs)}")
    print(f"Success rate: {metrics['success_rate']:.3f}")
    print(f"Mean return: {metrics['mean_return']:.3f} ± {metrics['std_return']:.3f}")
    print(f"Mean steps: {metrics['mean_steps']:.1f}")
    if metrics['mean_steps_to_hit']:
        print(f"Mean steps to hit (when successful): {metrics['mean_steps_to_hit']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()