"""Configurable GPU-ready DeepFM, DCN, and DIN models."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    family: str
    categorical_vocab_size: int
    field_count: int
    video_vocab_size: int
    embedding_dim: int = 16
    mlp_dims: tuple[int, ...] = (128, 64, 32)
    cross_layers: int = 3
    aux_tasks: tuple[str, ...] = ()
    dropout: float = 0.1

    def to_dict(self):
        value = asdict(self); value["mlp_dims"] = list(self.mlp_dims); value["aux_tasks"] = list(self.aux_tasks); return value

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "mlp_dims": tuple(value["mlp_dims"]), "aux_tasks": tuple(value.get("aux_tasks", ()))})


def _mlp(input_dim: int, widths: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers = []; current = input_dim
    for width in widths:
        layers += [nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)]; current = width
    return nn.Sequential(*layers)


class _Base(nn.Module):
    def __init__(self, config: ModelConfig, representation_dim: int):
        super().__init__(); self.config = config
        self.primary_head = nn.Linear(representation_dim, 1)
        self.aux_heads = nn.ModuleDict({name: nn.Linear(representation_dim, 1) for name in config.aux_tasks})

    def heads(self, representation):
        return {"long_view_logits": self.primary_head(representation).squeeze(-1),
                "aux_logits": {name: head(representation).squeeze(-1) for name, head in self.aux_heads.items()}}


class DeepFMV3(_Base):
    def __init__(self, config):
        width = config.field_count * config.embedding_dim; final = config.mlp_dims[-1]
        super().__init__(config, final + 1); self.linear = nn.Embedding(config.categorical_vocab_size, 1)
        self.embedding = nn.Embedding(config.categorical_vocab_size, config.embedding_dim)
        self.deep = _mlp(width, config.mlp_dims, config.dropout)

    def forward(self, X, candidate_video_ids, history_video_ids, history_mask):
        del candidate_video_ids, history_video_ids, history_mask
        e = self.embedding(X); summed = e.sum(1); fm = .5 * ((summed * summed).sum(1) - (e * e).sum((1, 2)))
        linear = self.linear(X).sum(1).squeeze(-1); rep = torch.cat([(linear + fm).unsqueeze(1), self.deep(e.flatten(1))], 1)
        return self.heads(rep)


class DCNV3(_Base):
    def __init__(self, config):
        width = config.field_count * config.embedding_dim; deep_dim = config.mlp_dims[-1]
        super().__init__(config, width + deep_dim); self.embedding = nn.Embedding(config.categorical_vocab_size, config.embedding_dim)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(config.cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(config.cross_layers)])
        self.deep = _mlp(width, config.mlp_dims, config.dropout)
        for weight in self.cross_w: nn.init.normal_(weight, std=.01)

    def forward(self, X, candidate_video_ids, history_video_ids, history_mask):
        del candidate_video_ids, history_video_ids, history_mask
        x0 = self.embedding(X).flatten(1); cross = x0
        for weight, bias in zip(self.cross_w, self.cross_b): cross = x0 * (cross * weight).sum(1, keepdim=True) + bias + cross
        return self.heads(torch.cat([cross, self.deep(x0)], 1))


class DINV3(_Base):
    def __init__(self, config):
        cat_width = config.field_count * config.embedding_dim; input_dim = cat_width + 4 * config.embedding_dim
        super().__init__(config, config.mlp_dims[-1]); self.categorical = nn.Embedding(config.categorical_vocab_size, config.embedding_dim)
        self.video = nn.Embedding(config.video_vocab_size, config.embedding_dim, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(4 * config.embedding_dim, config.embedding_dim), nn.ReLU(), nn.Linear(config.embedding_dim, 1))
        self.deep = _mlp(input_dim, config.mlp_dims, config.dropout)

    def forward(self, X, candidate_video_ids, history_video_ids, history_mask):
        candidate = self.video(candidate_video_ids); history = self.video(history_video_ids)
        expanded = candidate.unsqueeze(1).expand_as(history)
        features = torch.cat([history, expanded, history - expanded, history * expanded], -1)
        scores = self.attention(features).squeeze(-1)
        scores = scores.masked_fill(~history_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, 1) * history_mask.to(scores.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        interest = (history * weights.unsqueeze(-1)).sum(1)
        interaction = torch.cat([candidate, interest, candidate - interest, candidate * interest], 1)
        rep = self.deep(torch.cat([self.categorical(X).flatten(1), interaction], 1))
        return self.heads(rep)


def make_model(config: ModelConfig) -> nn.Module:
    families = {"deepfm": DeepFMV3, "dcn": DCNV3, "din": DINV3}
    if config.family not in families: raise ValueError(f"unsupported model family: {config.family}")
    return families[config.family](config)


def multitask_loss(outputs, y, aux_targets, aux_weight: float):
    loss = F.binary_cross_entropy_with_logits(outputs["long_view_logits"], y)
    if outputs["aux_logits"] and aux_weight:
        aux = [F.binary_cross_entropy_with_logits(logits, aux_targets[name]) for name, logits in outputs["aux_logits"].items()]
        loss = loss + aux_weight * torch.stack(aux).mean()
    return loss
