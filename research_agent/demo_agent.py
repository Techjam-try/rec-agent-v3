"""The smallest inspect -> train -> evaluate -> reflect agent demo.

This file is intentionally a teaching/demo version of ``agent.py``. It runs
one official-FM-style validation experiment, records each stage, and proposes
the next experiment without accessing the hidden-test period.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate  # supplied scorer; do not modify
from research_agent.data import encode, load_train_valid
from research_agent.models import FM


def log(path, stage, payload):
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, **payload}
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n")
    print(f"[{stage}] {payload.get('message', '')}")


def train_fm(train, valid, dim, epochs, seed):
    """A compact, validation-only reproduction of the starter FM training loop."""
    model = FM(dim, k=16, lr=.001, seed=seed)
    rng = np.random.default_rng(seed)
    best_score, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train["y"]))
        for start in range(0, len(order), 8192):
            idx = order[start:start + 8192]
            model.step_pointwise(train["X"][idx], train["y"][idx])
        metrics = evaluate(valid["users"], valid["y"], model.predict(valid["X"]))
        print(f"  epoch {epoch:02d}: primary={metrics['primary']:.4f}")
        if metrics["primary"] > best_score + 1e-5:
            best_score, best_epoch, bad = metrics["primary"], epoch, 0
            best_state = {key: value.copy() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    return model, evaluate(valid["users"], valid["y"], model.predict(valid["X"])), best_epoch


def main():
    parser = argparse.ArgumentParser(description="Simple validation-only research-agent demo")
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="./demo_runs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    run_log = output / "demo_log.jsonl"
    if run_log.exists():
        run_log.unlink()  # a demo is one self-contained, easy-to-read run

    log(run_log, "read_problem", {"message": "Target: rank long_view within user; use GAUC and nDCG@5.",
                                   "constraint": "development uses train + validation only; hidden test disabled"})
    splits = load_train_valid(args.data_dir)
    train_rate = float(np.mean([row["long_view"] for row in splits["train"]]))
    valid_rate = float(np.mean([row["long_view"] for row in splits["valid"]]))
    log(run_log, "inspect_data", {"message": "Loaded development data only.",
                                   "train_rows": len(splits["train"]), "valid_rows": len(splits["valid"]),
                                   "train_long_view_rate": train_rate, "valid_long_view_rate": valid_rate})
    encoded, dim, fields = encode(splits, use_history=False)
    log(run_log, "engineer_features", {"message": "Use official five categorical FM fields as the control.",
                                        "fields": fields, "feature_dimension": dim,
                                        "code_diff": "None: baseline-control experiment."})
    started = time.time()
    model, metrics, epoch = train_fm(encoded["train"], encoded["valid"], dim, args.epochs, args.seed)
    np.savez_compressed(output / "fm_validation_best.npz", **model.state_dict())
    log(run_log, "evaluate", {"message": "Evaluated with the unmodified official evaluator.",
                                "metrics": metrics, "best_epoch": epoch, "elapsed_seconds": round(time.time()-started, 2)})
    delta = float(metrics["primary"] - .6016)
    next_idea = ("Baseline reproduced. Next iteration: add dense engagement and watch-ratio auxiliary tasks."
                 if abs(delta) < .01 else "Baseline mismatch detected. Retry the control before changing the model.")
    log(run_log, "reflect_and_revise", {"message": next_idea, "baseline_validation_primary": .6016,
                                         "delta_from_published_validation_baseline": delta,
                                         "manual_interventions": 0,
                                         "next_hypothesis": "Multi-task auxiliary labels can regularize shared embeddings."})
    print(f"\nDone. Validation primary: {metrics['primary']:.4f}. Audit log: {run_log}")


if __name__ == "__main__":
    main()
