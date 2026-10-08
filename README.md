# WSI-RL Cancer Navigation

PPO-based reinforcement learning for whole-slide image (WSI) patch navigation on CAMELYON16. This research project explores whether an agent can inspect a limited number of regions, find tumor-containing tissue, and make navigation more efficient than uniform patch processing.

The repository contains a compact 384-dimensional PPO prototype and a separate UNI2-h feature extraction script. The included UNI2 extraction stage is not connected to the PPO policy: it produces raw 1,536-dimensional features, while the current PPO environment expects 384-dimensional features. The source is explicit about that boundary; the external SASHA training stages required to bridge it are not included.

## Project Goals

- Formulate patch selection as a sequential decision problem.
- Compare sparse terminal feedback with denser reward shaping.
- Measure tumor-hit success, episode return, and navigation steps.
- Explore stronger pathology representations through a separate UNI2-h feature-extraction path.
- Keep data and model artifacts external so the repository remains lightweight and respects access constraints.

## Project Status

The repository includes the PPO prototype, a small CAMELYON16 trainer/evaluator, and UNI2-h feature extraction for a 16-slide subset. It does not include CAMELYON16 slides, patch files, embeddings, trained weights, the downstream SASHA stages, or their checkpoints. The report's historical results are included below for context, but their raw logs are not available here for independent reproduction.

## WSI Data Pipeline

![WSI data pipeline](figures/data_pipeline.png)

*Figure 1. Slide tiling, tissue filtering, and frozen feature extraction used in the project workflow.*

The WSI workflow is organized into these stages:

1. **Slide input.** Load CAMELYON16 whole-slide images. Slides are not distributed in this repository; users must obtain them under the dataset's access terms.
2. **Patch tiling.** Convert each large slide into smaller, spatially located image patches. The UNI2 extraction script consumes coordinates stored in SASHA-style HDF5 patch files rather than generating those patch files itself.
3. **Tissue filtering.** Retain tissue-containing regions and omit background where the upstream tiling pipeline provides filtered coordinates.
4. **Feature extraction.** The historical DINOv2 workflow represented regions with 384-dimensional embeddings. The added UNI2-h extractor is a distinct experiment and writes raw 1,536-dimensional features. It requires access to the gated `MahmoodLab/UNI2-h` model.
5. **Feature organization.** Store patch features together with coordinates and tumor masks for the PPO prototype, or as SASHA-format files for later external SASHA stages. These generated files are intentionally ignored by Git.

## Reinforcement Learning Formulation

The PPO prototype models navigation as an episodic Markov decision process:

- **State:** a pooled 384-dimensional slide representation, updated as regions are visited.
- **Action:** choose a discrete patch or region index.
- **Transition:** inspect the selected region and update its feature representation; the environment tracks visited regions and tumor hits.
- **Reward:** choose a reward mode. Sparse reward is delivered at the end of an episode; first-hit reward rewards detecting tumor; dense reward adds a small per-step signal and a tumor-hit bonus.
- **Termination:** an episode ends at the configured step horizon; first-hit mode can end early on a positive slide after finding tumor.
- **Evaluation:** aggregate tumor-hit success, return, steps, and time-to-hit across evaluated slides/episodes.

The implementation is in [`RLogist/environment_rlogist.py`](RLogist/environment_rlogist.py), [`RLogist/rl/env_variable_wsi.py`](RLogist/rl/env_variable_wsi.py), and [`RLogist/rl/env_variable_wsi_dense.py`](RLogist/rl/env_variable_wsi_dense.py). These are research prototypes and should not be interpreted as clinical decision systems.

## PPO Agent

![PPO architecture](figures/ppo_architecture.png)

*Figure 2. Actor-critic policy for discrete patch selection.*

The prototype's `PolicyNetwork` maps the pooled input through two fully connected layers, then produces policy logits over patch actions and a scalar value estimate. `PPOAgent` samples actions from a categorical distribution and updates the policy using clipped probability ratios, value loss, and an entropy term. During evaluation, the environment's action mask restricts selection to valid patches.

## Defense Report Results

The following figures and metrics are taken from the six-page defense report dated April 2026. They document the locked CAMELYON16/SASHA experiments, rather than the compact PPO prototype included in this repository. The report states that the main comparison used all available slides, `split_4`, seed 4, 50 epochs, and best-checkpoint selection, except the UNI2-h run, which used seed 4004.

### Pipeline Comparison

![Defense report diagram comparing the baseline SASHA and Virchow2 feature-extractor variants](figures/defense_pipeline_comparison.png)

Only Step 2, the feature extractor, changed between these two pipeline variants; the report states the remaining stages, hyperparameters, seeds, and data splits were held fixed. Virchow2 features were projected from 2,560 to the 384 dimensions expected by SASHA.

### Dataset Split and Evaluation Protocol

![CAMELYON16 split composition and evaluation notes from the defense report](figures/defense_split_composition.png)

The report records 111 training slides, 27 balanced validation slides, and 129 test slides. It explicitly notes that the test partition is all-normal, making test AUC undefined and accuracy uninformative for tumor detection; the report therefore uses validation metrics as its primary evidence. Three slides were missing from intermediate data, so 267 of 270 are accounted for in the shown split summary.

### Locked Full-270 Comparison

![Locked comparison diagram and validation metrics table from the defense report](figures/defense_locked_comparison.png)

Reported metrics are validation-set results, not test-set results:

| Model | Validation AUC | Validation F1 | Validation accuracy |
| --- | ---: | ---: | ---: |
| SASHA baseline RL | 1.000 | 1.000 | 100.0% |
| SASHA + Virchow2 RL | 0.969 | 0.783 | 81.5% |
| SASHA + Virchow2 RL, last-checkpoint sanity check | 0.963 | 0.783 | 81.5% |
| SASHA + UNI2-h RL | 1.000 | 0.941 | 96.3% |
| ABMIL baseline | 1.000 | 0.941 | Not reported |
| ABMIL + Virchow2 | 1.000 | 1.000 | Not reported |

The locked comparison used `full270`, seed 4, and `split_4` for the listed baseline and Virchow2 rows; the UNI2-h RL row used seed 4004. These are transcribed from the defense report and have not been rerun from the files in this repository.

### Validation F1 During Training

![Validation F1 over training epochs for the SASHA baseline and Virchow2 RL](figures/defense_validation_f1_training.png)

The report notes that the Virchow2 agent's hard predictions remained at F1 0.783 from epoch 1, although its AUC changed as confidence scores shifted. It reports the Virchow2 AUC peaking at 0.969 around epoch 25 before declining, motivating careful checkpoint selection.

### Backbone and Dataset-Size Comparison

![Validation F1 by backbone and exploratory test AUC across dataset sizes](figures/defense_backbone_dataset_ablation.png)

In the locked full-dataset comparison, the report shows validation F1 of 1.000 for the SASHA baseline, 0.783 for SASHA + Virchow2 RL, and 0.941 for SASHA + UNI2-h RL. The dataset-size panel is explicitly exploratory: small subsets have very few test slides and high variance, so those bars should not be treated as a robust scale comparison.

### Qualitative Navigation

![Patch-selection trajectories for the SASHA baseline and SASHA plus Virchow2 on tumor slide 082](figures/defense_navigation_trajectories.png)

![Crop-level comparison of visited regions for the baseline and Virchow2 agents](figures/defense_tumor_region_comparison.png)

For the same `tumor_082` slide, the report says both agents made 121 visits; the baseline visited 54 annotated tumor patches, while Virchow2 visited 2. Its interpretation is that the Virchow2 policy found a discriminative signal early and stopped receiving useful incentive to continue targeted tumor search. This illustrates why slide classification scores and tumor-localization behavior are different measures.

### Project Timeline

![Six-month project timeline from the defense report](figures/defense_project_timeline.png)

## Earlier PPO Prototype Results

The following curves and metrics are from the earlier PPO prototype documented in the previous repository README. They are **not the locked SASHA/Virchow2/UNI2-h defense experiments above**. Their source run logs and checkpoints are not included, so these figures are retained as historical context rather than independently verified results.

| Sparse reward: episode return | Sparse reward: success rate |
| --- | --- |
| ![Sparse reward average episode return over training batches](figures/sparse-episode_return.png) | ![Sparse reward success rate over training batches](figures/training_success-sparse.png) |

| Sparse vs. dense: episode return | Sparse vs. dense: success rate |
| --- | --- |
| ![Average episode return comparison for sparse and dense rewards](figures/average_episode-comparison.png) | ![Success rate comparison for sparse and dense rewards](figures/success_rate-comparison.png) |

| PPO losses: sparse reward | Reward-shaping summary |
| --- | --- |
| ![Smoothed PPO objective, critic, entropy, and total losses](figures/ppo_losses-sparse.png) | ![Average reward comparison for sparse and dense shaping](figures/reward_shaping_comparison.png) |

The previous README reported the following earlier prototype values:

| Experiment | Success rate | Mean return | Mean steps |
| --- | ---: | ---: | ---: |
| Synthetic prototype | 82.0% | 0.65 | 4.2 |
| Sparse CAMELYON16 prototype | 0.0% | -97.7 | 997.0 |
| Dense reward prototype | 95.0% | 5.40 | 120.5 |

It also listed a sparse-reward training summary of average episode return `-2.3`, success rate `0.43`, and 200 training batches. The included `RLogist/train_camelyon.py` is only a small prototype using two hard-coded slide IDs, and its evaluator uses a fixed list of ten IDs; these scripts should not be treated as the implementation of the locked defense protocol.

## UNI2-h Feature Extraction Track

The included [`step2_extract_uni2_features.py`](sasha_adapted/preprocessing/step2_extract_uni2_features.py) loads UNI2-h through `timm`, reads patch coordinates and WSI regions, and extracts low- and high-resolution features. It is intended for the CAMELYON16 16-slide subset and expects external SASHA patch HDF5 files, slide images, a slide CSV, and Hugging Face access to the gated model.

This is an extraction step only. It is not a drop-in backbone for the PPO prototype: feature width is 1,536 for UNI2-h versus 384 for the included policy/environment. Downstream feature adaptation and SASHA training/checkpoints are outside this repository. Example commands and expected file inputs are documented in [`docs/experiment-scope.md`](docs/experiment-scope.md).

## Setup

Use Python 3.10 or newer. Install PyTorch and TorchVision using the command recommended for your operating system and accelerator at [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For WSI reading, install the OpenSlide system library (for example, `brew install openslide` on macOS). Practical full-slide UNI2 extraction requires a CUDA-enabled Linux environment and approved access to `MahmoodLab/UNI2-h` on Hugging Face.

## Running the PPO Prototype

Provide 384-dimensional patch embeddings, coordinates, and aligned tumor masks under `embeddings/`. See [experiment scope](docs/experiment-scope.md) for file naming and array shapes.

```bash
python -m RLogist.train_camelyon --reward first_hit
python -m RLogist.rl.evaluate_camelyon --reward first_hit --episodes-per-slide 20 --max-steps 100
```

The trainer writes model checkpoints under `RLogist/checkpoints/`. The evaluator writes per-episode logs and summary metrics under `results/`. These are runtime artifacts and are excluded from Git.

## Repository Layout

```text
RLogist/                                      PPO prototype, environments, trainer, evaluator
sasha_adapted/preprocessing/                  UNI2-h feature extraction stage
configs/                                      Notes on configuration and omitted cluster paths
docs/                                         Experiment scope and literature context
figures/                                      Report and historical training visualizations
results/                                      Runtime results location; no raw results committed
```

## Reproducibility and Data

CAMELYON16 slides, patient-level data, embeddings, HDF5 patch files, checkpoints, and generated run logs are not included. Obtain data and model access through their approved channels and follow their respective terms. Never commit private data or model credentials. The historical tables and figures above are retained to preserve report context, not as a claim that this repository can reproduce them as-is.

## Research Context

This work draws on reinforcement learning for WSI navigation, multiple-instance learning, and adaptive patch selection. The literature review also considered MuRCL, adaptive policy-driven MIL, ABMIL, and CLAM; these approaches informed project context but are not implemented in the included code. See [`docs/literature_review`](docs/literature_review).

## Intended Use

For research and education only. This code is not validated for clinical diagnosis or treatment decisions.