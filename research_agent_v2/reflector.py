"""Deterministic reflection keeps LLM prose separate from metric decisions."""
from __future__ import annotations


def reflect(events, epsilon=.002):
    completed = [event for event in events if event["status"] == "completed"]
    best = max(completed, key=lambda event: event["metrics"]["primary"])
    delta = best["metrics"]["primary"] - completed[0]["metrics"]["primary"]
    return {"最佳候选": best["recipe"]["name"], "best_primary": best["metrics"]["primary"],
            "相对官方控制组提升": delta,
            "结论": "达到实质提升，保留 checkpoint。" if delta > epsilon else "尚无超过 0.002 的实质提升；保留最佳 checkpoint 并研究下一 recipe。"}
