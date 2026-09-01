"""Device-aware batched neural candidate trainer."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from research_agent_v3.models import ModelConfig, make_model, multitask_loss


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    batch_size: int = 16384
    amp: bool = True
    epochs: int = 12
    patience: int = 3
    learning_rate: float = 1e-3
    aux_weight: float = 0.1
    min_batch_size: int = 1024
    seed: int = 0


@dataclass
class TrainingResult:
    state_dict: dict
    validation_scores: np.ndarray
    metrics: dict[str, float]
    best_epoch: int
    elapsed_seconds: float
    device_type: str
    device_name: str
    amp_enabled: bool
    effective_batch_size: int
    recovery_events: list[dict] = field(default_factory=list)


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}: raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def _batch(split, indices, device):
    return (
        torch.as_tensor(split["X"][indices], device=device, dtype=torch.long),
        torch.as_tensor(split["candidate_video_ids"][indices], device=device, dtype=torch.long),
        torch.as_tensor(split["history_video_ids"][indices], device=device, dtype=torch.long),
        torch.as_tensor(split["history_mask"][indices], device=device, dtype=torch.bool),
    )


def _predict(model, split, device, batch_size):
    model.eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(split["X"]), batch_size):
            indices = slice(start, start + batch_size)
            chunks.append(model(*_batch(split, indices, device))["long_view_logits"].float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty(0, np.float32)


def train_candidate(encoded, model_config: ModelConfig, training_config: TrainingConfig, evaluator: Callable):
    torch.manual_seed(training_config.seed); np.random.seed(training_config.seed)
    device = resolve_device(training_config.device); use_amp = training_config.amp and device.type == "cuda"
    model = make_model(model_config).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp); train, valid = encoded["train"], encoded["valid"]
    best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0; started = time.time()
    batch_size = training_config.batch_size; recoveries = []; oom_retried = False
    for epoch in range(1, training_config.epochs + 1):
        model.train(); order = np.random.permutation(len(train["y"])); start = 0
        while start < len(order):
            indices = order[start:start + batch_size]
            try:
                batch = _batch(train, indices, device); y = torch.as_tensor(train["y"][indices], device=device)
                aux = {name: torch.as_tensor(train["aux"][name][indices], device=device) for name in model_config.aux_tasks}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = multitask_loss(model(*batch), y, aux, training_config.aux_weight)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); start += len(indices)
            except torch.OutOfMemoryError:
                smaller = batch_size // 2
                if device.type != "cuda" or smaller < training_config.min_batch_size or oom_retried: raise
                recoveries.append({"event": "cuda_oom", "old_batch_size": batch_size, "new_batch_size": smaller})
                batch_size = smaller; oom_retried = True; torch.cuda.empty_cache()
        scores = _predict(model, valid, device, batch_size); metrics = evaluator(valid["users"], valid["y"], scores)
        if metrics["primary"] > best_score + 1e-8:
            best_score, best_epoch, stale = metrics["primary"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= training_config.patience: break
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state); scores = _predict(model, valid, device, batch_size); metrics = evaluator(valid["users"], valid["y"], scores)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return TrainingResult(best_state, scores, {k: float(v) for k,v in metrics.items()}, best_epoch, time.time()-started, device.type, name, use_amp, batch_size, recoveries)
