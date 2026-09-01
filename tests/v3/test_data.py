from __future__ import annotations

import numpy as np

from tests.v3.conftest import make_fixture_dataset
from research_agent_v3.data import (
    EncoderConfig,
    fit_transform,
    load_test_features,
    load_train_valid,
    transform_inference,
)


def test_loaders_enforce_fixed_date_windows_and_hide_test_labels(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    splits = load_train_valid(data_dir)
    test_rows = load_test_features(data_dir)

    assert [row["date"] for row in splits["train"]] == [20220408, 20220421]
    assert [row["date"] for row in splits["valid"]] == [20220422, 20220428]
    assert [row["date"] for row in test_rows] == [20220429, 20220508]
    assert set(test_rows[0]) == {"date", "hour", "user", "video", "author", "tab", "duration"}


def test_test_loader_never_uses_dict_reader_for_label_bearing_rows(tmp_path, monkeypatch):
    data_dir = make_fixture_dataset(tmp_path)
    import research_agent_v3.data as module
    monkeypatch.setattr(module.csv, "DictReader", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("forbidden full-row parser")))
    rows = load_test_features(data_dir)
    assert len(rows) == 2


def test_history_is_causal_and_validation_outcomes_do_not_update_it(tmp_path):
    splits = load_train_valid(make_fixture_dataset(tmp_path))
    encoded, state = fit_transform(splits, EncoderConfig(max_history=3))

    padding = state.padding_video_id
    v1 = state.video_vocab["v1"]
    assert encoded["train"]["history_video_ids"][0].tolist() == [padding, padding, padding]
    assert encoded["train"]["history_video_ids"][1].tolist() == [padding, padding, v1]
    assert encoded["valid"]["history_video_ids"][0].tolist() == [padding, padding, v1]
    assert encoded["valid"]["history_video_ids"][1].tolist() == [padding, padding, v1]
    assert encoded["valid"]["history_mask"].tolist() == [[False, False, True], [False, False, True]]
    assert encoded["train"]["candidate_video_ids"].tolist() == [state.video_vocab["v1"], state.video_vocab["v2"]]


def test_padding_and_unknown_video_ids_are_distinct(tmp_path):
    splits = load_train_valid(make_fixture_dataset(tmp_path))
    _, state = fit_transform(splits, EncoderConfig(max_history=2))

    assert state.padding_video_id != state.unknown_video_id


def test_inference_reuses_frozen_training_history_and_vocabulary(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    splits = load_train_valid(data_dir)
    _, state = fit_transform(splits, EncoderConfig(max_history=2))
    inference = transform_inference(splits["train"], load_test_features(data_dir), state)

    expected = [state.padding_video_id, state.video_vocab["v1"]]
    assert inference["history_video_ids"][0].tolist() == expected
    assert inference["history_video_ids"][1].tolist() == [state.padding_video_id, state.padding_video_id]
    assert inference["X"].dtype == np.int64
    assert "y" not in inference and "aux" not in inference
