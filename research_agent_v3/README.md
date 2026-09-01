# `research_agent_v3`

This directory contains the GPU-enabled V3 research agent, model implementations, experiment runner, checkpoint handling, and submission tools.

For project setup, KuaiRand-Pure download instructions, Colab GPU steps, verified results, and troubleshooting, see the [repository README](../README.md).

## Module map

| File | Responsibility |
|---|---|
| `agent.py` | Runs the bounded autonomous experiment loop and selects the best validation checkpoint. |
| `data.py` | Loads KuaiRand-Pure and builds leakage-safe train/validation/test features. |
| `cache.py` | Persists deterministic encoded features for reuse between experiments. |
| `models.py` | Implements the CUDA-aware DeepFM, DCN, DIN, and auxiliary-task models. |
| `operations.py` | Defines the allowed model and training operations available to the agent. |
| `recipe_compiler.py` | Validates and compiles bounded experiment recipes. |
| `runner.py` | Trains candidates, evaluates validation metrics, and records resource usage. |
| `checkpoints.py` | Saves and restores model configurations, weights, and recovery state. |
| `make_submission.py` | Generates label-free test predictions from one checkpoint. |
| `final_ensemble.py` | Reproduces the validated multi-seed ensemble and final submission. |

All commands below must be run from the repository root so the unchanged official `evaluate.py` and `submit.py` are available.

## Autonomous experiment

```bash
python -m research_agent_v3.agent \
  --data-dir ./KuaiRand-Pure/data \
  --output-dir ./v3_runs/colab_run \
  --cache-dir ./v3_runs/cache \
  --device cuda \
  --epochs 12
```

Arguments:

- `--data-dir`: required KuaiRand-Pure `data/` directory.
- `--output-dir`: required directory for logs and checkpoints.
- `--cache-dir`: encoded-feature cache; default is `.cache/v3`.
- `--device`: `cuda`, `cpu`, or `auto`; default is `auto`.
- `--epochs`: candidate training epochs; default is `12`.

## Single-checkpoint submission

```bash
python -m research_agent_v3.make_submission \
  --data-dir ./KuaiRand-Pure/data \
  --checkpoint ./v3_runs/colab_run/validation_best.pt \
  --output ./submission_v3.csv \
  --device cuda
```

This command performs test inference without reading test labels. Validate the generated file with the official checker:

```bash
python submit.py --check --split test ./submission_v3.csv
```

## Final ensemble

```bash
python -m research_agent_v3.final_ensemble \
  --data-dir ./KuaiRand-Pure/data \
  --din-checkpoint ./v3_runs/colab_run/validation_best.pt \
  --output-dir ./v3_runs/final \
  --cache-dir ./v3_runs/cache \
  --device cuda \
  --epochs 18
```

The output directory contains the ensemble manifest, member checkpoints, validation records, and `submission_v3.csv`.

## Development constraints

- Use the fixed date-based training and validation split.
- Select models and ensemble weights using validation data only.
- Never inspect or use test labels during training, tuning, or selection.
- Keep experiment recipes within the operations allowed by `recipe_compiler.py`.
- Do not commit datasets, `.pt` checkpoints, caches, secrets, or API tokens.

