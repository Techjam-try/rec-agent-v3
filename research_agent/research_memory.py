"""Small recommendation-research memory used to ground planning decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResearchCard:
    strategy: str
    bottleneck: str
    evidence: str
    expected_effect: str
    cost: str
    risk: str


CARDS = (
    ResearchCard("pairwise_bpr", "loss_alignment", "Ranking metrics differ from pointwise BCE.",
                 "Improve user-local ordering", "low", "Can be unstable with weak negative sampling"),
    ResearchCard("history_bpr", "sequence_interest", "Users have repeated train interactions.",
                 "Personalise item ranking from prior preference", "medium", "Carries BPR instability risk"),
    ResearchCard("multitask_engagement", "sparse_primary_signal", "Click/like/follow/comment/forward/watch ratio are available.",
                 "Regularise shared ID embeddings with denser labels", "low", "Auxiliary tasks can distract from long_view"),
    ResearchCard("history_multitask", "sequence_interest", "Both histories and dense feedback exist.",
                 "Combine sequence and engagement signals", "medium", "Higher variance and confounded gains"),
)


def retrieve(bottleneck, blocked=()):
    """Return small, explainable candidate memory; no external papers required."""
    blocked = set(blocked)
    preferred = [c for c in CARDS if c.strategy not in blocked and c.bottleneck == bottleneck]
    remaining = [c for c in CARDS if c.strategy not in blocked and c not in preferred]
    return [asdict(c) for c in preferred + remaining]
