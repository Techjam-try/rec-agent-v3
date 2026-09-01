import numpy as np
import torch
from research_agent_v3.checkpoints import load_checkpoint, save_checkpoint
from research_agent_v3.models import make_model
from tests.v3.test_models import tiny_batch, tiny_config


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    config = tiny_config("din"); model = make_model(config); model.eval()
    with torch.no_grad(): expected = model(*tiny_batch())["long_view_logits"].numpy()
    save_checkpoint(tmp_path/"model.pt", model.state_dict(), config, {"recipe":{"name":"din"}})
    restored, metadata = load_checkpoint(tmp_path/"model.pt", "cpu"); restored.eval()
    with torch.no_grad(): actual = restored(*tiny_batch())["long_view_logits"].numpy()
    assert np.allclose(expected, actual)
    assert metadata["recipe"]["name"] == "din"
