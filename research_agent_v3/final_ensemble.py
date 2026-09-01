"""Train, select, freeze, and export the final validation-only V3 ensemble."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from research_agent_v2.data import encode_base_inference, load_train_valid as load_v2
from research_agent_v2.torch_models import DCN, DeepFM
from research_agent_v3.cache import load_or_encode
from research_agent_v3.checkpoints import load_checkpoint
from research_agent_v3.data import EncoderConfig, EncoderState, load_test_features, transform_inference
from research_agent_v3.runner import _predict


def percentile_by_user(users, scores):
    scores = np.asarray(scores); result = np.empty(len(scores), np.float32); groups = {}
    for index, user in enumerate(users): groups.setdefault(str(user), []).append(index)
    for indices in groups.values():
        order = np.argsort(scores[indices], kind="mergesort"); ranks = np.empty(len(indices), np.float32)
        ranks[order] = np.arange(len(indices), dtype=np.float32); result[indices] = ranks / max(len(indices) - 1, 1)
    return result


def _plain(value):
    if isinstance(value, dict): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value.item() if hasattr(value, "item") else value


def _predict_base(model, X, device, batch_size=65536):
    model.eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], device=device, dtype=torch.long)
            chunks.append(model(batch).float().cpu().numpy())
    return np.concatenate(chunks)


def _train_base(cls, name, seed, encoded, dim, fields, evaluator, device, epochs, output):
    torch.manual_seed(seed); np.random.seed(seed); model = cls(dim, fields).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); rng = np.random.default_rng(seed)
    train, valid = encoded["train"], encoded["valid"]
    best, best_state, best_scores, stale, logs = -np.inf, None, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(train["y"]))
        for start in range(0, len(order), 8192):
            indices = order[start:start + 8192]
            X = torch.as_tensor(train["X"][indices], device=device, dtype=torch.long)
            y = torch.as_tensor(train["y"][indices], device=device)
            loss = F.binary_cross_entropy_with_logits(model(X), y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        scores = _predict_base(model, valid["X"], device)
        metrics = _plain(evaluator(valid["users"], valid["y"], scores)); logs.append({"epoch": epoch, **metrics})
        if metrics["primary"] > best + 1e-5:
            best, stale, best_scores = metrics["primary"], 0, scores.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3: break
    member = f"{name}_{seed}"
    torch.save({"family": name, "seed": seed, "dim": dim, "fields": fields,
                "state_dict": best_state, "validation": max(logs, key=lambda row: row["primary"])}, output / f"{member}.pt")
    np.save(output / f"{member}_valid.npy", best_scores)
    return member, best_scores, logs


def _coordinate_weights(components, users, labels, evaluator):
    weights = np.asarray([0.95 / (len(components) - 1)] * (len(components) - 1) + [0.05])
    score = lambda w: _plain(evaluator(users, labels, sum(a * item for a, item in zip(w, components))))
    best = score(weights)
    for _ in range(2):
        changed = False
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                local = (best, weights.copy())
                for delta in np.arange(-0.08, 0.081, 0.01):
                    trial = weights.copy(); trial[i] += delta; trial[j] -= delta
                    if np.any(trial < 0): continue
                    metrics = score(trial)
                    if metrics["primary"] > local[0]["primary"]: local = (metrics, trial)
                if local[0]["primary"] > best["primary"] + 1e-9:
                    best, weights, changed = local[0], local[1], True
        if not changed: break
    return weights, best


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True)
    parser.add_argument("--din-checkpoint", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=".cache/v3"); parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--device", default="cuda"); args = parser.parse_args(); started = time.time()
    output = Path(args.output_dir); models_dir = output / "models"; models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    try: from evaluate import evaluate
    except ImportError as exc: raise SystemExit("Run from the official Starter Kit root") from exc
    encoded, state, _ = load_or_encode(args.data_dir, EncoderConfig(), args.cache_dir)
    members, validation_scores, logs = [], [], []
    for seed in (0, 1, 2):
        for cls, name in ((DeepFM, "deepfm"), (DCN, "dcn")):
            member, scores, member_logs = _train_base(cls, name, seed, encoded, state.categorical_vocab_size,
                                                       len(state.fields), evaluate, device, args.epochs, models_dir)
            members.append(member); validation_scores.append(scores); logs.append({"member": member, "epochs": member_logs})
    din, din_meta = load_checkpoint(args.din_checkpoint); din = din.to(device)
    validation_scores.append(_predict(din, encoded["valid"], device, 16384)); members.append("din")
    components = [percentile_by_user(encoded["valid"]["users"], item) for item in validation_scores]
    weights, metrics = _coordinate_weights(components, encoded["valid"]["users"], encoded["valid"]["y"], evaluate)
    manifest = {"members": members, "weights": weights.tolist(), "validation_metrics": metrics,
                "official_fm_primary": 0.6016, "v2_primary": 0.605052,
                "delta_vs_official_fm": metrics["primary"] - 0.6016, "delta_vs_v2": metrics["primary"] - 0.605052,
                "iterations": len(members), "manual_interventions": 0, "wall_clock_seconds": time.time() - started,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    (output / "ensemble_manifest.json").write_text(json.dumps(_plain(manifest), indent=2), encoding="utf-8")
    (output / "runs.json").write_text(json.dumps(_plain(logs), indent=2), encoding="utf-8")
    raw = load_v2(args.data_dir); test_rows = load_test_features(args.data_dir)
    Xtest, dim = encode_base_inference(raw["train"], test_rows)
    if dim != state.categorical_vocab_size: raise RuntimeError("base inference vocabulary does not match training")
    test_scores = []
    for member in members[:-1]:
        saved = torch.load(models_dir / f"{member}.pt", map_location="cpu", weights_only=False)
        cls = DeepFM if saved["family"] == "deepfm" else DCN
        model = cls(saved["dim"], saved["fields"]).to(device); model.load_state_dict(saved["state_dict"])
        test_scores.append(_predict_base(model, Xtest, device))
    din_state = EncoderState.from_dict(din_meta["encoder_state"])
    test_scores.append(_predict(din, transform_inference([], test_rows, din_state), device, 16384))
    test_users = [row["user"] for row in test_rows]
    ranked = [percentile_by_user(test_users, item) for item in test_scores]
    final_scores = sum(weight * item for weight, item in zip(weights, ranked)); submission = output / "submission_v3.csv"
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["row_id", "user_id", "video_id", "score"])
        writer.writerows((i, row["user"], row["video"], float(final_scores[i])) for i, row in enumerate(test_rows))
    print(json.dumps(_plain(manifest), indent=2)); print(submission)


if __name__ == "__main__": main()
