# WSI Cancer Navigation with Reinforcement Learning

Research code for patch-level navigation in CAMELYON16 whole-slide images. This repository is a curated, report-oriented subset of the larger research workspace: source code and reproducibility notes are included, while datasets, embeddings, checkpoints, and run logs are not.

## Included Work

- `RLogist/`: 384-dimensional PPO navigation prototype, Gymnasium environments, and CAMELYON evaluation scripts.
- `sasha_adapted/preprocessing/`: UNI2-h feature extraction for the CAMELYON16 16-slide subset. This is a separate extraction track, not an integrated UNI2 PPO trainer.
- `figures/reward_shaping_comparison.png`: existing sparse-versus-dense reward comparison figure from the report workspace.
- `docs/experiment-scope.md`: data expectations, commands, and known limitations.

## Setup

Use Python 3.10 or newer. Install PyTorch and TorchVision using the command recommended for your operating system and accelerator at [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

UNI2 feature extraction also requires the OpenSlide system library (for example, `brew install openslide` on macOS). Use a CUDA-enabled Linux environment for practical extraction of the full slide subset.

The UNI2-h extraction track additionally requires access to the gated `MahmoodLab/UNI2-h` model on Hugging Face and a valid Hugging Face login.

## CAMELYON16 PPO Prototype

The prototype expects precomputed 384-dimensional embeddings and matching coordinates and tumor masks under `embeddings/`. It can train and evaluate the included first-hit or dense reward variants:

```bash
python -m RLogist.train_camelyon --reward first_hit
python -m RLogist.rl.evaluate_camelyon --reward first_hit
```

Use `--help` on the evaluation command to see its episode and step options. This prototype is separate from the UNI2-h extractor; its policy and environments are configured for 384-dimensional inputs.

## UNI2-h Feature Extraction

The extractor requires CAMELYON16 slide files, SASHA patch-coordinate HDF5 files, a CSV listing the slides, and access to the gated UNI2-h weights. Those assets must be provided locally and are deliberately not committed. See [experiment scope](docs/experiment-scope.md) for a command template and the expected feature dimensions.

## Data and Intended Use

CAMELYON16 and pretrained model files are not included. Follow their respective access terms. This repository is for research and education only; it is not a clinical diagnostic system.