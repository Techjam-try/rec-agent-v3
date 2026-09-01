"""Experiment-tree metadata for logs, UI, and the final hackathon report."""
from __future__ import annotations


def tree_record(iteration, history, diagnosis, candidates):
    best = max((e for e in history if e.get("status") == "completed" and e.get("metrics")),
               key=lambda e: float(e["metrics"]["primary"]), default=None)
    return {"node_id": iteration + 1, "parent_id": best.get("iteration") if best else None,
            "diagnosis": diagnosis, "candidate_memory": candidates}
