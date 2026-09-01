"""Export the frozen V2 two-model ensemble without reading test labels.

Example:
  python research_agent_v2/make_submission.py --data-dir ./KuaiRand-Pure/data \
      --run-dir ./v2_runs/run_008 --output submission_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_agent_v2.data import encode_base_inference, load_test_features, load_train_valid
from research_agent_v2.ensemble_agent import _percentile_by_user
from research_agent_v2.torch_models import make_model


def predict(checkpoint, dim, X):
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recipe = saved["recipe"]
    if recipe.get("model_family") not in {"deepfm", "dcn"}:
        raise ValueError(f"{checkpoint} is not a supported frozen torch model")
    if any(recipe.get(flag, False) for flag in ("use_author_history", "use_history_count", "use_hour_weekday", "watch_bucket")):
        raise ValueError("submission exporter currently supports the frozen 5-field models only")
    model = make_model(recipe["model_family"], dim, X.shape[1])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).long()).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Export the frozen V2 validation-selected ensemble")
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--run-dir", required=True, help="Completed V2 run containing ensemble_result.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    ensemble = json.loads((run_dir / "ensemble_result.json").read_text(encoding="utf-8"))
    if not ensemble.get("accepted"):
        raise ValueError("The selected ensemble was not accepted on validation; refusing to export it.")
    model_a, model_b = ensemble["models"]
    checkpoint_a = run_dir / "models" / f"{model_a}.pt"
    checkpoint_b = run_dir / "models" / f"{model_b}.pt"
    if not checkpoint_a.exists() or not checkpoint_b.exists():
        raise FileNotFoundError("Selected model checkpoint is missing from the completed run.")

    # The two calls below read train labels for vocabulary fitting and test
    # exposure features only. No test labels or post-exposure columns are read.
    train = load_train_valid(args.data_dir)["train"]
    test_rows = load_test_features(args.data_dir)
    X, dim = encode_base_inference(train, test_rows)
    raw_a, raw_b = predict(checkpoint_a, dim, X), predict(checkpoint_b, dim, X)
    users = [row["user"] for row in test_rows]
    scores = ensemble["weight_a"] * _percentile_by_user(users, raw_a) + ensemble["weight_b"] * _percentile_by_user(users, raw_b)
    if not np.isfinite(scores).all():
        raise ValueError("Refusing to write NaN/Inf scores.")

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["row_id", "user_id", "video_id", "score"])
        for index, (row, score) in enumerate(zip(test_rows, scores)):
            writer.writerow([index, row["user"], row["video"], f"{float(score):.12g}"])
    if len(test_rows) != len(scores) or any(not math.isfinite(float(x)) for x in scores):
        raise RuntimeError("Internal submission validation failed.")
    print(f"Wrote {output}: {len(scores):,} test rows; frozen models={model_a}+{model_b}; "
          f"weights={ensemble['weight_a']}/{ensemble['weight_b']}. No test labels were read.")


if __name__ == "__main__":
    main()
