# RecResearcher V3

GPU-enabled autonomous ML research agent for the TikTok TechJam 2026 KuaiRand-Pure recommender benchmark.

## Verified validation result

| System | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM | 0.6674 | 0.5357 | 0.6016 |
| V2 best ensemble | 0.671813 | 0.538292 | 0.605052 |
| **V3 final ensemble** | **0.672020** | **0.538426** | **0.605223** |

V3 improves over V2 by `+0.000171` and over the official FM by `+0.003623`. These are validation results only; no hidden-test metric is claimed.

## What V3 adds

- Tesla T4/CUDA training with cached leakage-safe date splits.
- DIN causal positive-history attention and multi-behavior auxiliary learning.
- Three-seed DeepFM/DCN stability ensemble selected only on validation data.
- Bounded experiment logs, checkpoints, automatic recovery, and resource reporting.
- Label-free test inference and official submission-format checking.

See [research_agent_v3/README.md](research_agent_v3/README.md) for exact Colab commands. The official Starter Kit supplies the unchanged `evaluate.py` and `submit.py`.

## Data and task

- Dataset: KuaiRand-Pure from <https://kuairand.com/>
- Label: `long_view`
- Metrics: GAUC and nDCG@5; Primary is their mean.
- Train: 2022-04-08 to 2022-04-21; validation: 2022-04-22 to 2022-04-28.
- Test labels are not used during development.

## Limitations

The measured V3-over-V2 delta is positive but small and should be confirmed across additional independent runs. The final ensemble uses seven members, trading inference simplicity for stability. Pairwise fine-tuning and extra context features were tested and rejected because they did not improve the official validation metric.

## Team contributions

V2 was supplied by the team repository. V3 GPU adaptation, DIN/multi-task modeling, robustness fixes, Colab experiments, multi-seed selection, final submission generation, and documentation were completed in this iteration.
