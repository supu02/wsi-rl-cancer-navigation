# Source Code

Runnable source is organized at the repository root:

- `RLogist/` contains the 384-dimensional PPO agent, Gymnasium environments, CAMELYON16 prototype trainer, and evaluator.
- `sasha_adapted/preprocessing/step2_extract_uni2_features.py` extracts raw 1,536-dimensional UNI2-h features and is a separate pipeline stage.

The UNI2-h extractor is not connected to the PPO prototype. The external SASHA pipeline, downstream UNI2 training stages, datasets, and trained checkpoints are not included. See `docs/experiment-scope.md` before attempting reproduction.
