"""Allowlisted research operations. Qwen proposes; this module enforces."""
from __future__ import annotations


ALLOWED_OPERATIONS = {
    "recent_positive_author_history": {"窗口": [1, 3, 5, 10]},
    "positive_history_count_bucket": {"边界": "nonnegative integer list"},
    "hour_weekday_context": {},
    "auxiliary_weight_search": {"任务": ["click", "like", "follow", "comment", "forward", "hate", "profile_enter", "watch_ratio"]},
    "watch_time_auxiliary": {"形式": ["ratio_bucket", "censored_regression"]},
    "random_valid_audit": {},
}


def validate_operations(operations):
    """Drop unsafe or malformed LLM suggestions instead of executing arbitrary code."""
    accepted, rejected = [], []
    for item in operations or []:
        name = item.get("操作") or item.get("operation")
        if name not in ALLOWED_OPERATIONS:
            rejected.append({"建议": item, "原因": "不在安全操作白名单"})
            continue
        accepted.append({"操作": name, "参数": item.get("参数") or item.get("params", {}),
                         "假设": item.get("假设") or item.get("hypothesis", "")})
    return accepted, rejected
