"""Reconstructable V3 PyTorch checkpoints."""
from pathlib import Path
import torch
from research_agent_v3.models import ModelConfig, make_model

def save_checkpoint(path, state_dict, model_config: ModelConfig, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema":1, "state_dict":state_dict, "model_config":model_config.to_dict(), "metadata":metadata}, path)

def load_checkpoint(path, map_location="cpu"):
    saved = torch.load(path, map_location=map_location, weights_only=False)
    model = make_model(ModelConfig.from_dict(saved["model_config"])); model.load_state_dict(saved["state_dict"])
    return model, saved["metadata"]
