🧠 WSI-RL-Cancer-Navigation

PPO-based reinforcement learning for whole-slide cancer image navigation (CAMELYON16)
Exploring reward shaping, representation learning, and biologically-inspired search strategies for tumor localization.

⸻

🔬 Overview

Whole-slide images (WSIs) in digital pathology are extremely high-dimensional and sparse in signal. Exhaustively scanning them is computationally expensive and biologically unrealistic.

This project investigates whether reinforcement learning can:
	•	Efficiently navigate large histopathology slides
	•	Identify tumor-relevant regions
	•	Improve sample efficiency under sparse supervision
	•	Learn biologically meaningful exploration strategies

The core approach uses Proximal Policy Optimization (PPO) to train an agent to navigate CAMELYON16 slides under different reward regimes.

⸻

📂 Dataset
	•	CAMELYON16 (Whole-Slide Histopathology)
	•	Binary objective: tumor vs non-tumor regions
	•	Patch-based interaction environment

Due to dataset licensing, raw data is not included.

⸻

🏗 Methodology

1️⃣ RL Navigation Framework
	•	Custom WSI navigation environment
	•	Agent observes image patches
	•	Discrete movement actions across slide grid
	•	Episode terminates on success or max steps

2️⃣ Reward Modes

Two reward settings were implemented and compared:
	•	Sparse reward: reward only when tumor region is reached
	•	Dense reward: intermediate shaping rewards

Strict evaluation metrics were used to avoid reward leakage.

⸻

📊 Evaluation Metrics

The following are logged per training batch:
	•	Average episode return
	•	Success rate
	•	Episode length
	•	PPO loss components:
	•	Policy loss
	•	Critic/value loss
	•	Entropy term

Smoothed training curves are available in /figures.

Sparse Reward Example
	•	Gradual increase in episode return
	•	Stabilization of PPO losses
	•	Success rate convergence around ~0.4–0.5

## 🧪 Evaluation on CAMELYON16 (Test Slides)

We evaluate three settings:

- Baseline sparse reward
- Synthetic prototype reward shaping
- Final dense reward formulation

### 📊 Quantitative Results

Note: Training curves may show intermediate learning signals; the table reports strict test-time evaluation on held-out slides.

| Experiment | Success Rate | Mean Return | Mean Steps |
|------------|-------------|------------|------------|
| Baseline (Sparse) | 0.0% | -97.7 | 997.0 |
| Prototype (Synthetic) | 82.0% | 0.65 | 4.2 |
| Dense Reward (Final) | **95.0%** | **5.40** | **120.5** |

These results demonstrate that reward shaping is essential for stable policy learning in large-scale histopathology navigation tasks.

(Extended training currently ongoing.)

### Key Observations

- Sparse reward fails to provide sufficient learning signal (0% success).
- Synthetic shaping dramatically improves learning stability.
- Final dense reward achieves 95% success while maintaining controlled episode length.

⸻

🧬 Binary Classification Head

After navigation stabilizes, a binary classifier head is attached to predict:

Cancer vs Non-Cancer

This enables:
	•	Joint representation learning
	•	Evaluation of predictive signal beyond navigation success
	•	Comparison with classical supervised pipelines

⸻

🧩 Ongoing & Planned Extensions

🔹 Trident-based Patch Tiling

Separate branch integrates Trident for structured patch extraction and improved slide coverage.

🔹 UNI2-h (Virchow-2) Backbone

Experiments with pretrained foundation vision models to improve feature representations before PPO training.

🔹 Swarm Intelligence

Multi-agent exploration strategy:
	•	Cooperative region discovery
	•	Coverage optimization
	•	Reduced redundant exploration

🔹 Foveated Vision Strategy

Biologically-inspired selective attention:
	•	Coarse global scan
	•	Fine-grained zoom-in near suspicious regions
	•	Adaptive patch resolution

⸻

🛠 Tech Stack
	•	Python
	•	PyTorch
	•	TorchRL
	•	PPO
	•	Matplotlib
	•	Custom Gym-style environment

⸻

⚠️ Research Status

This is an active research project.

Current state:
	•	Stable PPO baseline implemented
	•	Sparse vs dense reward comparison complete
	•	Extended training in progress
	•	Binary classifier integrated
	•	Representation experiments underway

Codebase is being refactored before full public release.

⸻

🎯 Research Question

The broader objective is:

Can intelligent exploration strategies extract predictive phenotypic signal from whole-slide images more efficiently than static patch classification?

⸻

If you are interested in collaboration or discussion, feel free to reach out.
