"""Deterministic bottleneck detector: diagnose before changing a model."""
from __future__ import annotations

from research_agent.data_analyst import analyze


def profile(splits):
    """Compact train/validation EDA passed to the planner, never test data."""
    return analyze(splits)


def diagnose(history, epsilon=.002):
    """Map experiment evidence to one actionable bottleneck and its rationale."""
    completed = [e for e in history if e.get("status") == "completed" and e.get("metrics")]
    if not completed:
        return {"bottleneck": "baseline", "evidence": "No validation experiment exists yet.", "blocked": []}
    latest = completed[-1]
    delta = latest.get("improvement_over_validation_best")
    if latest["strategy"].get("kind") == "bpr" and delta is not None and delta <= -.02:
        return {"bottleneck": "sparse_primary_signal",
                "evidence": "BPR caused a severe validation regression; do not repeat the BPR family.",
                "blocked": ["pairwise_bpr", "history_bpr"]}
    if delta is not None and delta > epsilon:
        return {"bottleneck": "robustness_check",
                "evidence": "A material gain was observed; verify it before stacking complexity.", "blocked": []}
    return {"bottleneck": "sparse_primary_signal",
            "evidence": "No material gain; use dense auxiliary feedback before adding model capacity.", "blocked": []}
