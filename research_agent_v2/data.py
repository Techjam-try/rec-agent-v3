"""V2 train/validation-only loader and recipe-driven feature encoder."""
from __future__ import annotations
import csv, os
from collections import defaultdict
from datetime import datetime
import numpy as np

TRAIN_LOG, LATER_LOG = "log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"
TRAIN_RANGE, VALID_RANGE, TEST_RANGE = (20220408, 20220421), (20220422, 20220428), (20220429, 20220508)
BINARY_AUX = ("click", "like", "follow", "comment", "forward", "hate")

def _authors(data_dir):
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8") as f:
        return {r["video_id"]: r["author_id"] for r in csv.DictReader(f)}

def _row(r, authors):
    duration, play = max(float(r["duration_ms"]), 1.), max(float(r["play_time_ms"]), 0.)
    return {"date": int(r["date"]), "hour": int(r["hourmin"]) // 100, "user": r["user_id"],
            "video": r["video_id"], "author": authors.get(r["video_id"], "UNK"), "tab": r["tab"],
            "duration": duration, "long_view": float(r["long_view"] != "0"),
            "click": float(r["is_click"] != "0"), "like": float(r["is_like"] != "0"),
            "follow": float(r["is_follow"] != "0"), "comment": float(r["is_comment"] != "0"),
            "forward": float(r["is_forward"] != "0"), "hate": float(r["is_hate"] != "0"),
            "watch_ratio": min(play / duration, 1.)}

def load_train_valid(data_dir):
    authors, train, valid = _authors(data_dir), [], []
    for filename, target, dates in ((TRAIN_LOG, train, TRAIN_RANGE), (LATER_LOG, valid, VALID_RANGE)):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                if dates[0] <= int(raw["date"]) <= dates[1]: target.append(_row(raw, authors))
    return {"train": train, "valid": valid}


def load_test_features(data_dir):
    """Load only inference-time test fields, deliberately excluding every label.

    This is intentionally separate from ``load_train_valid``: a submission
    export must not even parse ``long_view`` or the post-exposure actions from
    the test CSV.
    """
    authors, rows = _authors(data_dir), []
    with open(os.path.join(data_dir, LATER_LOG), encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if TEST_RANGE[0] <= int(raw["date"]) <= TEST_RANGE[1]:
                duration = max(float(raw["duration_ms"]), 1.)
                rows.append({"date": int(raw["date"]), "hour": int(raw["hourmin"]) // 100,
                             "user": raw["user_id"], "video": raw["video_id"],
                             "author": authors.get(raw["video_id"], "UNK"), "tab": raw["tab"],
                             "duration": duration})
    return rows

def _bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2-4" if n < 5 else "5-9" if n < 10 else "10-19" if n < 20 else "20+"

def encode(splits, use_author_history=False, use_history_count=False, use_hour_weekday=False, watch_bucket=False):
    train = splits["train"]; edges = np.quantile([r["duration"] for r in train], np.linspace(0, 1, 11)[1:-1])
    fields = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
    if use_author_history: fields.append("last_positive_author")
    if use_history_count: fields.append("positive_history_bucket")
    if use_hour_weekday: fields.append("hour_weekday")
    aux_names = BINARY_AUX + (("watch_high",) if watch_bucket else ("watch_ratio",))
    def values(r, previous_author="UNK", count=0):
        x = [r["user"], r["video"], r["author"], r["tab"], str(int(np.searchsorted(edges, r["duration"])))]
        if use_author_history: x.append(previous_author)
        if use_history_count: x.append(_bucket(count))
        if use_hour_weekday: x.append(f"{datetime.strptime(str(r['date']), '%Y%m%d').weekday()}_{r['hour']}")
        return x
    vocabs = [dict() for _ in fields]
    for r in train:
        for i, value in enumerate(values(r, r["author"], 20)): vocabs[i].setdefault(value, len(vocabs[i]))
    unks, dims = [len(v) for v in vocabs], [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32); previous_author, count, result = defaultdict(lambda: "UNK"), defaultdict(int), {}
    for split in ("train", "valid"):
        rows = splits[split]; X = np.empty((len(rows), len(fields)), np.int32); y = np.empty(len(rows), np.float32)
        aux = {n: np.empty(len(rows), np.float32) for n in aux_names}; users = []
        for i, r in enumerate(rows):
            for j, value in enumerate(values(r, previous_author[r["user"]], count[r["user"]])): X[i, j] = vocabs[j].get(value, unks[j]) + offsets[j]
            y[i] = r["long_view"]; users.append(r["user"])
            for n in BINARY_AUX: aux[n][i] = r[n]
            aux[aux_names[-1]][i] = float(r["watch_ratio"] >= .6) if watch_bucket else r["watch_ratio"]
            if split == "train" and r["long_view"]: previous_author[r["user"]] = r["author"]; count[r["user"]] += 1
        result[split] = {"X": X, "y": y, "users": users, "aux": aux, "aux_names": aux_names}
    return result, int(sum(dims)), fields


def encode_base_inference(train, rows):
    """Recreate the 5-field training vocabulary and encode label-free rows."""
    edges = np.quantile([r["duration"] for r in train], np.linspace(0, 1, 11)[1:-1])
    def base_values(r):
        return [r["user"], r["video"], r["author"], r["tab"],
                str(int(np.searchsorted(edges, r["duration"])))]
    vocabs = [dict() for _ in range(5)]
    for r in train:
        for i, value in enumerate(base_values(r)):
            vocabs[i].setdefault(value, len(vocabs[i]))
    dims = [len(vocab) + 1 for vocab in vocabs]
    unks = [len(vocab) for vocab in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    X = np.empty((len(rows), 5), np.int32)
    for i, row in enumerate(rows):
        for j, value in enumerate(base_values(row)):
            X[i, j] = vocabs[j].get(value, unks[j]) + offsets[j]
    return X, int(sum(dims))
