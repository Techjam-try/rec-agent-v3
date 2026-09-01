# RecResearcher V3

GPU-enabled autonomous machine-learning research agent for the TikTok TechJam 2026 KuaiRand-Pure recommender benchmark. V3 preserves V2 and adds CUDA training, DIN, multi-behavior auxiliary learning, cached features, bounded experiment search, reproducible multi-seed ensembling, and label-free submission generation.

## Quick start: Google Colab GPU

The recommended route is the included [`colab/RecResearcher_V3_GPU.ipynb`](colab/RecResearcher_V3_GPU.ipynb). Open it in Colab, select **Runtime → Change runtime type → T4 GPU**, then run the cells in order.

To run manually in a new Colab notebook, use the following cells.

### 1. Clone the repository

```bash
!git clone https://github.com/Techjam-try/rec-agent-v3.git
%cd rec-agent-v3
```

### 2. Confirm that CUDA is available

```python
import torch
assert torch.cuda.is_available(), "Enable a GPU under Runtime → Change runtime type"
print(torch.cuda.get_device_name(0))
```

### 3. Install dependencies

```bash
!pip install -q -r requirements-v3.txt
```

### 4. Download KuaiRand-Pure

```bash
!wget -q --show-progress https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
!echo "0820331067a3784d9691136f772b35a7  KuaiRand-Pure.tar.gz" | md5sum -c -
!tar -xzf KuaiRand-Pure.tar.gz
!ls KuaiRand-Pure/data
```

The dataset source is <https://kuairand.com/>. Do not commit the downloaded dataset or model checkpoints to GitHub.

### 5. Run the V3 tests

```bash
!PYTHONPATH=. python -m pytest tests/v3 -q
```

### 6. Run the autonomous V3 experiment

```bash
!python -m research_agent_v3.agent \
  --data-dir ./KuaiRand-Pure/data \
  --output-dir ./v3_runs/colab_run \
  --cache-dir ./v3_runs/cache \
  --device cuda \
  --epochs 12
```

The agent evaluates bounded candidate recipes using the validation split and saves its best validation checkpoint under `v3_runs/colab_run/`.

### 7. Generate a submission from one checkpoint

```bash
!python -m research_agent_v3.make_submission \
  --data-dir ./KuaiRand-Pure/data \
  --checkpoint ./v3_runs/colab_run/validation_best.pt \
  --output ./submission_v3.csv \
  --device cuda

!python submit.py --check --split test ./submission_v3.csv
```

This inference path does not read test labels. The last command checks the official submission format and row alignment.

## Reproduce the final ensemble

After the DIN/autonomous run has produced `validation_best.pt`, run the reproducible multi-seed DeepFM/DCN ensemble:

```bash
!python -m research_agent_v3.final_ensemble \
  --data-dir ./KuaiRand-Pure/data \
  --din-checkpoint ./v3_runs/colab_run/validation_best.pt \
  --output-dir ./v3_runs/final \
  --cache-dir ./v3_runs/cache \
  --device cuda \
  --epochs 18

!python submit.py --check --split test ./v3_runs/final/submission_v3.csv
```

The final outputs are:

- `v3_runs/final/submission_v3.csv`: label-free test predictions.
- `v3_runs/final/ensemble_manifest.json`: ensemble members, weights, and validation metrics.
- `v3_runs/final/models/`: trained checkpoints; store these in Google Drive rather than GitHub.
- `v3_runs/colab_run/`: autonomous experiment logs and its best validation checkpoint.

## Verified validation result

| System | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM | 0.6674 | 0.5357 | 0.6016 |
| V2 best ensemble | 0.671813 | 0.538292 | 0.605052 |
| **V3 final ensemble** | **0.672020** | **0.538426** | **0.605223** |

`Primary = (GAUC + nDCG@5) / 2`. V3 improves over V2 by `+0.000171` and over the official FM baseline by `+0.003623`. These are validation results only; no hidden-test score is claimed. The generated submission contains 170,588 rows and passed the official alignment checker.

Machine-readable records and the generated submission are available in [`results/`](results/).

## What V3 changes

- Uses Tesla T4/CUDA training while retaining CPU fallback through `--device auto`.
- Caches leakage-safe encoded features and fixed date-based splits.
- Adds DIN causal positive-history attention and multi-behavior auxiliary learning.
- Searches bounded, auditable experiment recipes instead of making unrestricted code changes.
- Uses three-seed DeepFM/DCN stability ensembling selected only on validation data.
- Persists experiment logs, checkpoints, recovery state, and resource usage.
- Produces test predictions without accessing test labels.

## Dataset and evaluation protocol

- Dataset: KuaiRand-Pure.
- Target label: `long_view`.
- Train dates: 2022-04-08 through 2022-04-21.
- Validation dates: 2022-04-22 through 2022-04-28.
- Metrics: GAUC and nDCG@5; Primary is their arithmetic mean.
- Model selection and ensemble weighting use validation data only.
- Test labels must never be inspected or used for training, tuning, or model selection.
- The official `evaluate.py` and `submit.py` remain unchanged.

## Repository layout

```text
rec-agent-v3/
├── colab/                         # Ready-to-run GPU notebook
├── research_agent_v2/             # Original V2 implementation
├── research_agent_v3/             # GPU agent, models and submission tools
├── results/                        # Validated metrics and final CSV
├── tests/v3/                       # V3 unit and smoke tests
├── evaluate.py                     # Official validation evaluator
├── submit.py                       # Official submission checker
├── requirements-v3.txt
└── README.md
```

For module-level details, see [`research_agent_v3/README.md`](research_agent_v3/README.md).

## Collaboration workflow

Contributors should create a branch instead of developing directly on `main`:

```bash
git checkout -b feature/my-change
git add .
git commit -m "Describe the change"
git push -u origin feature/my-change
```

Then open a pull request into `main`. Do not commit `KuaiRand-Pure/`, `.pt` checkpoints, API tokens, or Colab secrets.

## Troubleshooting

- **`torch.cuda.is_available()` is false:** change the Colab runtime to a T4 GPU and reconnect.
- **CUDA out of memory:** restart the runtime, close unused notebooks, or run fewer candidates before reproducing the full ensemble.
- **Dataset file not found:** verify that `KuaiRand-Pure/data/` exists directly under the repository root.
- **`evaluate.py` or `submit.py` cannot be imported:** run commands from the repository root.
- **Colab runtime disconnects:** persist important checkpoints and run artifacts to Google Drive; `/content` is temporary.

## Limitations

The measured V3-over-V2 improvement is positive but small and should be confirmed with additional independent runs. The final ensemble contains seven members, trading inference simplicity for stability. Pairwise fine-tuning and extra context features were tested but excluded because they did not improve the official validation metric.

