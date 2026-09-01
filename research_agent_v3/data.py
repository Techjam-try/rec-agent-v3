"""Leakage-safe KuaiRand-Pure loading and causal V3 feature encoding."""
from __future__ import annotations

import csv
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
TRAIN_RANGE = (20220408, 20220421)
VALID_RANGE = (20220422, 20220428)
TEST_RANGE = (20220429, 20220508)
BINARY_AUX = ("click", "like", "follow", "comment", "forward", "hate")
BASE_FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")


@dataclass(frozen=True)
class EncoderConfig:
    max_history: int = 20
    duration_buckets: int = 10
    use_hour_weekday: bool = False

    def __post_init__(self) -> None:
        if self.max_history < 1:
            raise ValueError("max_history must be positive")
        if self.duration_buckets < 2:
            raise ValueError("duration_buckets must be at least 2")


@dataclass
class EncoderState:
    config: EncoderConfig
    fields: tuple[str, ...]
    vocabs: list[dict[str, int]]
    offsets: list[int]
    duration_edges: list[float]
    video_vocab: dict[str, int]
    padding_video_id: int
    unknown_video_id: int
    train_positive_history: dict[str, list[str]]

    @property
    def categorical_vocab_size(self) -> int:
        return sum(len(vocab) + 1 for vocab in self.vocabs)

    @property
    def video_vocab_size(self) -> int:
        return self.unknown_video_id + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "fields": list(self.fields),
            "vocabs": self.vocabs,
            "offsets": self.offsets,
            "duration_edges": self.duration_edges,
            "video_vocab": self.video_vocab,
            "padding_video_id": self.padding_video_id,
            "unknown_video_id": self.unknown_video_id,
            "train_positive_history": self.train_positive_history,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EncoderState":
        return cls(
            config=EncoderConfig(**value["config"]),
            fields=tuple(value["fields"]),
            vocabs=[{str(k): int(v) for k, v in vocab.items()} for vocab in value["vocabs"]],
            offsets=[int(item) for item in value["offsets"]],
            duration_edges=[float(item) for item in value["duration_edges"]],
            video_vocab={str(k): int(v) for k, v in value["video_vocab"].items()},
            padding_video_id=int(value["padding_video_id"]),
            unknown_video_id=int(value["unknown_video_id"]),
            train_positive_history={str(k): list(v) for k, v in value["train_positive_history"].items()},
        )


def _authors(data_dir: os.PathLike[str] | str) -> dict[str, str]:
    path = Path(data_dir) / "video_features_basic_pure.csv"
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle); header = next(reader); video = header.index("video_id"); author = header.index("author_id")
        return {row[video]: row[author] for row in reader}


def _development_row(raw: dict[str, str], authors: dict[str, str]) -> dict[str, Any]:
    duration = max(float(raw["duration_ms"]), 1.0)
    play_time = max(float(raw["play_time_ms"]), 0.0)
    return {
        "date": int(raw["date"]),
        "hour": int(raw["hourmin"]) // 100,
        "user": raw["user_id"],
        "video": raw["video_id"],
        "author": authors.get(raw["video_id"], "UNK"),
        "tab": raw["tab"],
        "duration": duration,
        "long_view": float(raw["long_view"] != "0"),
        "click": float(raw["is_click"] != "0"),
        "like": float(raw["is_like"] != "0"),
        "follow": float(raw["is_follow"] != "0"),
        "comment": float(raw["is_comment"] != "0"),
        "forward": float(raw["is_forward"] != "0"),
        "hate": float(raw["is_hate"] != "0"),
        "watch_ratio": min(play_time / duration, 1.0),
    }


def load_train_valid(data_dir: os.PathLike[str] | str) -> dict[str, list[dict[str, Any]]]:
    authors = _authors(data_dir)
    result: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    specs = ((TRAIN_LOG, "train", TRAIN_RANGE), (LATER_LOG, "valid", VALID_RANGE))
    for filename, split, dates in specs:
        with (Path(data_dir) / filename).open(encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                date = int(raw["date"])
                if dates[0] <= date <= dates[1]:
                    result[split].append(_development_row(raw, authors))
    return result


def load_test_features(data_dir: os.PathLike[str] | str) -> list[dict[str, Any]]:
    """Read only fields available at impression time from the test window."""
    authors = _authors(data_dir)
    rows: list[dict[str, Any]] = []
    with (Path(data_dir) / LATER_LOG).open(encoding="utf-8") as handle:
        reader = csv.reader(handle); header = next(reader)
        safe = {name: header.index(name) for name in ("date","hourmin","user_id","video_id","tab","duration_ms")}
        for values in reader:
            date = int(values[safe["date"]])
            if TEST_RANGE[0] <= date <= TEST_RANGE[1]:
                rows.append({
                    "date": date,
                    "hour": int(values[safe["hourmin"]]) // 100,
                    "user": values[safe["user_id"]],
                    "video": values[safe["video_id"]],
                    "author": authors.get(values[safe["video_id"]], "UNK"),
                    "tab": values[safe["tab"]],
                    "duration": max(float(values[safe["duration_ms"]]), 1.0),
                })
    return rows


def _field_values(row: dict[str, Any], edges: np.ndarray, config: EncoderConfig) -> list[str]:
    values = [
        str(row["user"]),
        str(row["video"]),
        str(row["author"]),
        str(row["tab"]),
        str(int(np.searchsorted(edges, row["duration"]))),
    ]
    if config.use_hour_weekday:
        weekday = datetime.strptime(str(row["date"]), "%Y%m%d").weekday()
        values.append(f"{weekday}_{row['hour']}")
    return values


def _padded_history(videos: Iterable[str], state: EncoderState) -> tuple[np.ndarray, np.ndarray]:
    encoded = [state.video_vocab.get(video, state.unknown_video_id) for video in videos]
    encoded = encoded[-state.config.max_history :]
    padding = state.config.max_history - len(encoded)
    ids = np.asarray([state.padding_video_id] * padding + encoded, dtype=np.int64)
    mask = np.asarray([False] * padding + [True] * len(encoded), dtype=np.bool_)
    return ids, mask


def _encode_rows(
    rows: list[dict[str, Any]],
    state: EncoderState,
    initial_histories: dict[str, list[str]],
    update_history: bool,
    include_targets: bool,
) -> dict[str, Any]:
    size = len(rows)
    fields = len(state.fields)
    X = np.empty((size, fields), dtype=np.int64)
    histories = np.empty((size, state.config.max_history), dtype=np.int64)
    masks = np.empty((size, state.config.max_history), dtype=np.bool_)
    candidate_video_ids = np.empty(size, dtype=np.int64)
    edges = np.asarray(state.duration_edges)
    queues = defaultdict(lambda: deque(maxlen=state.config.max_history))
    for user, videos in initial_histories.items():
        queues[user].extend(videos[-state.config.max_history :])
    users: list[str] = []
    for index, row in enumerate(rows):
        values = _field_values(row, edges, state.config)
        for column, value in enumerate(values):
            unknown = len(state.vocabs[column])
            X[index, column] = state.vocabs[column].get(value, unknown) + state.offsets[column]
        histories[index], masks[index] = _padded_history(queues[row["user"]], state)
        candidate_video_ids[index] = state.video_vocab.get(row["video"], state.unknown_video_id)
        users.append(row["user"])
        if update_history and row.get("long_view", 0.0) > 0:
            queues[row["user"]].append(row["video"])
    output: dict[str, Any] = {
        "X": X,
        "history_video_ids": histories,
        "history_mask": masks,
        "candidate_video_ids": candidate_video_ids,
        "users": users,
    }
    if include_targets:
        output["y"] = np.asarray([row["long_view"] for row in rows], dtype=np.float32)
        output["aux"] = {
            name: np.asarray([row[name] for row in rows], dtype=np.float32)
            for name in BINARY_AUX
        }
    return output


def fit_transform(
    splits: dict[str, list[dict[str, Any]]], config: EncoderConfig | None = None
) -> tuple[dict[str, dict[str, Any]], EncoderState]:
    config = config or EncoderConfig()
    train = splits["train"]
    if not train:
        raise ValueError("training split is empty")
    duration_edges = np.quantile(
        [row["duration"] for row in train],
        np.linspace(0, 1, config.duration_buckets + 1)[1:-1],
    )
    fields = list(BASE_FIELDS)
    if config.use_hour_weekday:
        fields.append("hour_weekday")
    vocabs: list[dict[str, int]] = [dict() for _ in fields]
    for row in train:
        for index, value in enumerate(_field_values(row, duration_edges, config)):
            vocabs[index].setdefault(value, len(vocabs[index]))
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int64).tolist()
    video_vocab = {video: index + 1 for index, video in enumerate(dict.fromkeys(row["video"] for row in train))}
    padding_video_id = 0
    unknown_video_id = len(video_vocab) + 1
    history_queues: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=config.max_history))
    for row in train:
        if row["long_view"] > 0:
            history_queues[row["user"]].append(row["video"])
    frozen_histories = {user: list(queue) for user, queue in history_queues.items()}
    state = EncoderState(
        config=config,
        fields=tuple(fields),
        vocabs=vocabs,
        offsets=offsets,
        duration_edges=duration_edges.astype(float).tolist(),
        video_vocab=video_vocab,
        padding_video_id=padding_video_id,
        unknown_video_id=unknown_video_id,
        train_positive_history=frozen_histories,
    )
    encoded = {
        "train": _encode_rows(train, state, {}, update_history=True, include_targets=True),
        "valid": _encode_rows(
            splits["valid"], state, frozen_histories, update_history=False, include_targets=True
        ),
    }
    return encoded, state


def transform_inference(
    train_rows: list[dict[str, Any]], inference_rows: list[dict[str, Any]], state: EncoderState
) -> dict[str, Any]:
    """Encode label-free rows using only the frozen positive training history."""
    del train_rows  # State already contains the audited, frozen training history.
    return _encode_rows(
        inference_rows,
        state,
        state.train_positive_history,
        update_history=False,
        include_targets=False,
    )
