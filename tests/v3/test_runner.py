from __future__ import annotations

import numpy as np
import torch

from research_agent_v3.models import ModelConfig
from research_agent_v3.runner import TrainingConfig, resolve_device, train_candidate


def encoded_fixture():
    X = np.array([[1, 5, 9, 13, 17], [2, 6, 10, 14, 18], [3, 7, 11, 15, 19], [4, 8, 12, 16, 20]], np.int64)
    base = {"X": X, "candidate_video_ids": np.array([1, 2, 3, 4]), "history_video_ids": np.array([[0,0],[0,1],[1,2],[2,3]]), "history_mask": np.array([[0,0],[0,1],[1,1],[1,1]], bool), "y": np.array([0,1,0,1], np.float32), "aux": {"click": np.array([0,1,1,1], np.float32)}, "users": ["u1","u1","u2","u2"]}
    return {"train": base, "valid": {k: (v.copy() if hasattr(v,"copy") else list(v)) for k,v in base.items()}}


def config():
    return ModelConfig("deepfm", 32, 5, 8, embedding_dim=4, mlp_dims=(8,4))


def evaluator(users, labels, scores):
    value = float(np.mean((scores - labels) ** 2)); return {"gauc": 1-value, "ndcg": 1-value, "primary": 1-value}


def test_resolve_device_honors_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_cpu_smoke_training_returns_metrics_and_restored_best_state():
    result = train_candidate(encoded_fixture(), config(), TrainingConfig(device="cpu", epochs=2, batch_size=2, amp=False), evaluator=evaluator)
    assert set(result.metrics) == {"gauc", "ndcg", "primary"}
    assert result.device_type == "cpu" and result.best_epoch >= 1
    assert result.validation_scores.shape == (4,)
    assert result.elapsed_seconds >= 0


def test_cuda_request_fails_clearly_without_cuda():
    if torch.cuda.is_available(): return
    try: resolve_device("cuda")
    except RuntimeError as exc: assert "CUDA" in str(exc)
    else: raise AssertionError("explicit CUDA request must not silently use CPU")
