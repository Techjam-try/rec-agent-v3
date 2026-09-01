# RecResearcher V3

V3 keeps the V2 project unchanged and adds CUDA-aware DeepFM, DCN, DIN, multi-behavior auxiliary learning, deterministic feature caching, bounded autonomous recipes, auditable checkpoints, and label-free submission export.

## Colab GPU

Open `colab/RecResearcher_V3_GPU.ipynb`, choose **Runtime → Change runtime type → T4 GPU**, and run the cells in order. The notebook downloads KuaiRand-Pure, checks its MD5, runs the test suite, and launches:

```bash
python -m research_agent_v3.agent \
  --data-dir ./KuaiRand-Pure/data \
  --output-dir ./v3_runs/colab_run \
  --device cuda \
  --epochs 12
```

Generate predictions without reading test labels:

```bash
python -m research_agent_v3.make_submission \
  --data-dir ./KuaiRand-Pure/data \
  --checkpoint ./v3_runs/colab_run/validation_best.pt \
  --output ./submission_v3.csv \
  --device cuda
```

The full run must be launched from the Starter Kit root so the unchanged official `evaluate.py` is importable. Report only persisted validation metrics; do not describe them as hidden-test results.

## Final verified ensemble

After the DIN run completes, freeze the reproducible multi-seed ensemble and generate the final submission:

```bash
python -m research_agent_v3.final_ensemble \
  --data-dir ./KuaiRand-Pure/data \
  --din-checkpoint ./v3_runs/colab_run/validation_best.pt \
  --output-dir ./v3_runs/final \
  --cache-dir ./v3_runs/cache \
  --device cuda \
  --epochs 18

python submit.py --check --split test ./v3_runs/final/submission_v3.csv
```

The executed Tesla T4 run produced `GAUC=0.672020`, `nDCG@5=0.538426`, and `primary=0.605223`. This is `+0.000171` over the V2 validation result (`0.605052`) and `+0.003623` over the official FM validation baseline (`0.6016`). The generated submission contained 170,588 rows and passed the official alignment checker.
