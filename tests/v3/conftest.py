from __future__ import annotations

import csv
from pathlib import Path


LOG_FIELDS = [
    "date", "hourmin", "user_id", "video_id", "tab", "duration_ms",
    "play_time_ms", "long_view", "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_hate",
]


def _log_row(date: int, user: str, video: str, long_view: int) -> dict[str, object]:
    return {
        "date": date,
        "hourmin": 930,
        "user_id": user,
        "video_id": video,
        "tab": "1",
        "duration_ms": 1000,
        "play_time_ms": 800 if long_view else 100,
        "long_view": long_view,
        "is_click": long_view,
        "is_like": 0,
        "is_follow": 0,
        "is_comment": 0,
        "is_forward": 0,
        "is_hate": 0,
    }


def make_fixture_dataset(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    with (data_dir / "video_features_basic_pure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "author_id"])
        writer.writeheader()
        for index in range(1, 8):
            writer.writerow({"video_id": f"v{index}", "author_id": f"a{index}"})
    train_rows = [
        _log_row(20220407, "u1", "v7", 1),
        _log_row(20220408, "u1", "v1", 1),
        _log_row(20220421, "u1", "v2", 0),
        _log_row(20220422, "u1", "v3", 1),
    ]
    later_rows = [
        _log_row(20220421, "u1", "v7", 1),
        _log_row(20220422, "u1", "v3", 1),
        _log_row(20220428, "u1", "v4", 0),
        _log_row(20220429, "u1", "v5", 1),
        _log_row(20220508, "u2", "v6", 0),
    ]
    for name, rows in (
        ("log_standard_4_08_to_4_21_pure.csv", train_rows),
        ("log_standard_4_22_to_5_08_pure.csv", later_rows),
    ):
        with (data_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return data_dir

