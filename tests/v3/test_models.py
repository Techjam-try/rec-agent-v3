from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from research_agent_v3.models import ModelConfig, make_model, multitask_loss


def tiny_config(family: str) -> ModelConfig:
    return ModelConfig(
        family=family,
        categorical_vocab_size=32,
        field_count=5,
        video_vocab_size=12,
        embedding_dim=4,
        mlp_dims=(8, 4),
        cross_layers=2,
        aux_tasks=(),
    )


def tiny_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[1, 7, 12, 18, 22], [2, 8, 13, 19, 23], [3, 9, 14, 20, 24], [4, 10, 15, 21, 25]],
        dtype=torch.long,
    )
    history = torch.tensor([[0, 0, 1], [0, 2, 3], [4, 5, 6], [0, 0, 0]], dtype=torch.long)
    mask = history.ne(0)
    candidates = torch.tensor([1, 3, 6, 8], dtype=torch.long)
    return X, candidates, history, mask


@pytest.mark.parametrize("family", ["deepfm", "dcn", "din"])
def test_each_model_family_returns_finite_primary_logits(family):
    model = make_model(tiny_config(family))

    outputs = model(*tiny_batch())

    assert outputs["long_view_logits"].shape == (4,)
    assert torch.isfinite(outputs["long_view_logits"]).all()


def test_din_padding_values_cannot_change_predictions():
    model = make_model(tiny_config("din"))
    model.eval()
    X, candidates, history, mask = tiny_batch()
    changed_padding = history.clone()
    changed_padding[~mask] = 7

    with torch.no_grad():
        original = model(X, candidates, history, mask)["long_view_logits"]
        changed = model(X, candidates, changed_padding, mask)["long_view_logits"]

    assert torch.allclose(original, changed, atol=1e-6)


def test_din_all_padding_history_stays_finite():
    model = make_model(tiny_config("din"))
    X, candidates, history, mask = tiny_batch()

    outputs = model(X, candidates, history, torch.zeros_like(mask))

    assert torch.isfinite(outputs["long_view_logits"]).all()


def test_din_half_precision_mask_does_not_overflow():
    model = make_model(tiny_config("din")).half()

    outputs = model(*tiny_batch())

    assert torch.isfinite(outputs["long_view_logits"]).all()


def test_auxiliary_heads_contribute_to_loss_and_receive_gradients():
    config = replace(tiny_config("din"), aux_tasks=("click", "like"))
    model = make_model(config)
    outputs = model(*tiny_batch())
    targets = {
        "click": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "like": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }

    loss = multitask_loss(
        outputs,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        targets,
        aux_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(outputs["aux_logits"]) == {"click", "like"}
    assert all(parameter.grad is not None for parameter in model.aux_heads.parameters())
