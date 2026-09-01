"""Validation-safe exploratory analysis for RecResearcher.

This module deliberately receives only the train/valid splits returned by
``load_train_valid``.  It contains no test path or test-label access.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


AUXILIARY = ("click", "like", "follow", "comment", "forward")


def _rate(rows):
    return round(float(np.mean([row["long_view"] for row in rows])), 5) if rows else 0.0


def _per_user_summary(rows):
    counts = Counter(row["user"] for row in rows)
    values = np.asarray(list(counts.values()), dtype=np.float64)
    return {
        "users": len(counts),
        "p50_events": round(float(np.quantile(values, .50)), 2),
        "p90_events": round(float(np.quantile(values, .90)), 2),
        "p99_events": round(float(np.quantile(values, .99)), 2),
    }


def _distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.quantile(values, .50)), 2),
        "p90": round(float(np.quantile(values, .90)), 2),
        "p99": round(float(np.quantile(values, .99)), 2),
        "zero_fraction": round(float(np.mean(values == 0)), 5),
    }


def _tab_stats(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["tab"]].append(row["long_view"])
    return [{"tab": str(tab), "rows": len(labels),
             "long_view_rate": round(float(np.mean(labels)), 5)}
            for tab, labels in sorted(grouped.items(), key=lambda item: -len(item[1]))]


def _duration_stats(rows, edges):
    buckets = [[] for _ in range(len(edges) + 1)]
    for row in rows:
        buckets[int(np.searchsorted(edges, row["duration"], side="right"))].append(row["long_view"])
    return [{"bucket": index, "rows": len(labels), "long_view_rate": _rate_from_labels(labels)}
            for index, labels in enumerate(buckets)]


def _rate_from_labels(labels):
    return round(float(np.mean(labels)), 5) if labels else 0.0


def _auxiliary_signal(rows):
    output = {}
    base = _rate(rows)
    for label in AUXILIARY:
        positive = [row["long_view"] for row in rows if row[label]]
        output[label] = {
            "positive_rate": round(len(positive) / len(rows), 5),
            "long_view_given_positive": _rate_from_labels(positive),
            "lift_vs_base": round(_rate_from_labels(positive) - base, 5),
        }
    ratios = np.asarray([row["watch_ratio"] for row in rows], dtype=np.float64)
    return {"base_long_view_rate": base, "labels": output,
            "watch_ratio_p50": round(float(np.quantile(ratios, .50)), 4),
            "watch_ratio_p90": round(float(np.quantile(ratios, .90)), 4)}


def analyze(splits):
    """Return compact, JSON-serializable EDA evidence for train and validation."""
    train, valid = splits["train"], splits["valid"]
    train_users = {row["user"] for row in train}
    train_videos = {row["video"] for row in train}
    train_history = Counter(row["user"] for row in train)
    train_positive_history = Counter(row["user"] for row in train if row["long_view"])
    valid_users = sorted({row["user"] for row in valid})
    durations = np.asarray([row["duration"] for row in train], dtype=np.float64)
    edges = np.quantile(durations, [.2, .4, .6, .8])
    valid_unseen_users = sum(row["user"] not in train_users for row in valid) / len(valid)
    valid_unseen_videos = sum(row["video"] not in train_videos for row in valid) / len(valid)
    return {
        "scope": "train + validation only; test split is never loaded",
        "rows": {"train": len(train), "valid": len(valid)},
        "long_view_rate": {"train": _rate(train), "valid": _rate(valid),
                            "delta_valid_minus_train": round(_rate(valid) - _rate(train), 5)},
        "cold_start_in_validation": {"unseen_user_row_fraction": round(valid_unseen_users, 5),
                                       "unseen_video_row_fraction": round(valid_unseen_videos, 5)},
        "user_activity": {"train": _per_user_summary(train), "valid": _per_user_summary(valid)},
        "validation_user_train_history": {
            "meaning": "For each validation user, number of events/positive long views available before validation starts. This—not validation-window event count—measures sequence-feature coverage.",
            "all_train_events": _distribution_summary([train_history[user] for user in valid_users]),
            "train_positive_long_views": _distribution_summary([train_positive_history[user] for user in valid_users]),
            "users_with_at_least_5_train_positive_events": round(
                float(np.mean([train_positive_history[user] >= 5 for user in valid_users])), 5),
        },
        "tab_long_view": {"train": _tab_stats(train), "valid": _tab_stats(valid)},
        "duration_long_view": {"quantile_edges_ms": [round(float(x), 2) for x in edges],
                                 "train": _duration_stats(train, edges),
                                 "valid": _duration_stats(valid, edges)},
        "auxiliary_signal": {"train": _auxiliary_signal(train), "valid": _auxiliary_signal(valid)},
        "history_policy": "Features may use only prior train long_view events; validation history is frozen at the train boundary.",
    }
