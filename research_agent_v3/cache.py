"""Fingerprint and cache deterministic V3 encoded arrays."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from research_agent_v3.data import (
    LATER_LOG,
    TRAIN_LOG,
    EncoderConfig,
    EncoderState,
    fit_transform,
    load_train_valid,
)


CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = (TRAIN_LOG, LATER_LOG, "video_features_basic_pure.csv")


def cache_fingerprint(data_dir: os.PathLike[str] | str, config: EncoderConfig) -> str:
    root = Path(data_dir).resolve()
    sources = []
    for name in SOURCE_FILES:
        stat = (root / name).stat()
        sources.append({"name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    value = {
        "schema": CACHE_SCHEMA_VERSION,
        "config": asdict(config),
        "sources": sources,
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flatten(encoded: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split, values in encoded.items():
        for key in ("X", "candidate_video_ids", "history_video_ids", "history_mask", "y"):
            arrays[f"{split}__{key}"] = np.asarray(values[key])
        for name, target in values["aux"].items():
            arrays[f"{split}__aux__{name}"] = np.asarray(target)
    return arrays


def _inflate(arrays: Any, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    encoded: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        encoded[split] = {
            key: arrays[f"{split}__{key}"]
            for key in ("X", "candidate_video_ids", "history_video_ids", "history_mask", "y")
        }
        encoded[split]["users"] = list(manifest["users"][split])
        prefix = f"{split}__aux__"
        encoded[split]["aux"] = {
            key[len(prefix) :]: arrays[key]
            for key in arrays.files
            if key.startswith(prefix)
        }
    return encoded


def load_or_encode(
    data_dir: os.PathLike[str] | str,
    config: EncoderConfig,
    cache_dir: os.PathLike[str] | str,
) -> tuple[dict[str, dict[str, Any]], EncoderState, bool]:
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = cache_fingerprint(data_dir, config)
    arrays_path = directory / f"{fingerprint}.npz"
    manifest_path = directory / f"{fingerprint}.manifest.json"
    if arrays_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as arrays:
            encoded = _inflate(arrays, manifest)
        return encoded, EncoderState.from_dict(manifest["encoder_state"]), True

    encoded, state = fit_transform(load_train_valid(data_dir), config)
    manifest = {
        "schema": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "encoder_state": state.to_dict(),
        "users": {split: values["users"] for split, values in encoded.items()},
    }
    arrays = _flatten(encoded)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as handle:
        temporary_arrays = Path(handle.name)
    temporary_manifest = directory / f".{fingerprint}.manifest.tmp"
    try:
        np.savez_compressed(temporary_arrays, **arrays)
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        temporary_arrays.replace(arrays_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_arrays.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return encoded, state, False
