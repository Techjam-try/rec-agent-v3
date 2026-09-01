from __future__ import annotations

from research_agent_v3.cache import load_or_encode
from research_agent_v3.data import EncoderConfig
from tests.v3.conftest import make_fixture_dataset


def test_cache_reuses_identical_sources(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    cache_dir = tmp_path / "cache"

    first_encoded, first_state, first_hit = load_or_encode(
        data_dir, EncoderConfig(max_history=2), cache_dir
    )
    second_encoded, second_state, second_hit = load_or_encode(
        data_dir, EncoderConfig(max_history=2), cache_dir
    )

    assert first_hit is False
    assert second_hit is True
    assert first_state.to_dict() == second_state.to_dict()
    assert first_encoded["valid"]["X"].tolist() == second_encoded["valid"]["X"].tolist()


def test_cache_invalidates_when_encoder_config_changes(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    load_or_encode(data_dir, EncoderConfig(max_history=2), cache_dir)

    _, _, cache_hit = load_or_encode(data_dir, EncoderConfig(max_history=3), cache_dir)

    assert cache_hit is False
    assert len(list(cache_dir.glob("*.manifest.json"))) == 2


def test_cache_invalidates_when_source_file_changes(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    load_or_encode(data_dir, EncoderConfig(max_history=2), cache_dir)
    path = data_dir / "video_features_basic_pure.csv"
    path.write_text(path.read_text(encoding="utf-8") + "v99,a99\n", encoding="utf-8")

    _, _, cache_hit = load_or_encode(data_dir, EncoderConfig(max_history=2), cache_dir)

    assert cache_hit is False
    assert len(list(cache_dir.glob("*.manifest.json"))) == 2
