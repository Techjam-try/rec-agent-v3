"""Candidate strategies the agent can propose. Add a class to extend its search."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str
    hypothesis: str
    kind: str
    use_history: bool = False
    use_aux: bool = False
    code_diff: str = ""


CATALOG = (
    Strategy("official_fm", "Reproduce the published 5-field pointwise FM before searching.", "pointwise",
             code_diff="Baseline control: 5-field FM + pointwise binary log loss."),
    Strategy("pairwise_bpr", "GAUC/nDCG are ranking metrics; BPR within each user should align the loss with evaluation.", "bpr",
             code_diff="models.FM.step_bpr: optimize positive-minus-negative user-local score gaps."),
    Strategy("history_bpr", "A user's prior long-view video is a compact causal interest signal; combine it with BPR.", "bpr", True,
             code_diff="data.encode(use_history=True): add causal last_positive_video field, frozen during validation."),
    Strategy("multitask_engagement", "Clicks and engagements provide denser auxiliary supervision for the shared FM embeddings.", "multitask", False, True,
             code_diff="models.FM(heads=7): shared FM representation with click/like/follow/comment/forward/watch auxiliary heads."),
    Strategy("history_multitask", "Sequence interest plus dense engagement supervision may improve cold/ambiguous rankings.", "multitask", True, True,
             code_diff="Compose causal history field with shared-representation multi-task learning."),
)
