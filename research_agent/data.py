"""Leakage-safe train/validation loader for the research agent.

Unlike the starter kit's general loader, this module never creates a test split
and stops reading the second log before the test period begins.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
TRAIN_RANGE = (20220408, 20220421)
VALID_RANGE = (20220422, 20220428)
BASE_FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")


def _authors(data_dir):
    out = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["video_id"]] = row["author_id"]
    return out


def _row(row, vid2author):
    duration = max(float(row["duration_ms"]), 1.0)
    play_time = max(float(row["play_time_ms"]), 0.0)
    return {
        "date": int(row["date"]), "user": row["user_id"], "video": row["video_id"],
        "author": vid2author.get(row["video_id"], "UNK"), "tab": row["tab"],
        "duration": duration, "long_view": float(row["long_view"] != "0"),
        "click": float(row["is_click"] != "0"), "like": float(row["is_like"] != "0"),
        "follow": float(row["is_follow"] != "0"), "comment": float(row["is_comment"] != "0"),
        "forward": float(row["is_forward"] != "0"),
        "watch_ratio": min(play_time / duration, 1.0),
    }


def load_train_valid(data_dir):
    """Load only development data. Test labels are neither returned nor parsed."""
    vid2author = _authors(data_dir)
    train, valid = [], []
    with open(os.path.join(data_dir, TRAIN_LOG), encoding="utf-8") as fh:
        for source in csv.DictReader(fh):
            if TRAIN_RANGE[0] <= int(source["date"]) <= TRAIN_RANGE[1]:
                train.append(_row(source, vid2author))
    # Select validation rows before constructing a dict. The later log is not
    # guaranteed to be globally ordered, so do not early-stop here. Test rows
    # are discarded as raw CSV values: their labels are never parsed or used.
    with open(os.path.join(data_dir, LATER_LOG), encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        date_i = header.index("date")
        for values in reader:
            date = int(values[date_i])
            if not (VALID_RANGE[0] <= date <= VALID_RANGE[1]):
                continue
            valid.append(_row(dict(zip(header, values)), vid2author))
    if not train or not valid:
        raise RuntimeError("Could not find the official train and validation periods.")
    return {"train": train, "valid": valid}


def _duration_edges(rows, n=10):
    return np.quantile([r["duration"] for r in rows], np.linspace(0, 1, n + 1)[1:-1])


def encode(splits, use_history=False):
    """Encode IDs from train only; validation-only values map to UNK.

    When enabled, ``last_positive_video`` is a causal, train-derived sequence
    feature. Validation histories are frozen at the end of training so no
    validation labels enter a prediction feature.
    """
    edges = _duration_edges(splits["train"])
    fields = list(BASE_FIELDS) + (["last_positive_video"] if use_history else [])

    def base(r):
        return [r["user"], r["video"], r["author"], r["tab"],
                str(int(np.searchsorted(edges, r["duration"])))]

    vocabs = [dict() for _ in fields]
    for r in splits["train"]:
        for i, value in enumerate(base(r)):
            vocabs[i].setdefault(value, len(vocabs[i]))
        if use_history:
            vocabs[-1].setdefault(r["video"], len(vocabs[-1]))
    unks = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    previous = defaultdict(lambda: "UNK")
    encoded = {}
    for name in ("train", "valid"):
        rows = splits[name]
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users, aux = [], {k: np.empty(len(rows), dtype=np.float32) for k in
                          ("click", "like", "follow", "comment", "forward", "watch_ratio")}
        for i, r in enumerate(rows):
            values = base(r)
            if use_history:
                values.append(previous[r["user"]])
            for j, value in enumerate(values):
                X[i, j] = vocabs[j].get(value, unks[j]) + offsets[j]
            y[i] = r["long_view"]
            users.append(r["user"])
            for key in aux:
                aux[key][i] = r[key]
            # Only train labels can update a later history. Validation is frozen.
            if name == "train" and r["long_view"] > 0:
                previous[r["user"]] = r["video"]
        encoded[name] = {"X": X, "y": y, "users": users, "aux": aux}
    return encoded, int(sum(dims)), fields
