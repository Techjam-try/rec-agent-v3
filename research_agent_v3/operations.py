"""Bounded V3 research operations."""
ALLOWED_OPERATIONS = {"gpu_capacity_search", "din_history_attention", "multibehavior_auxiliary", "auxiliary_weight_search"}

def validate_operations(items):
    accepted, rejected = [], []
    for item in items or []:
        name = item.get("操作") or item.get("operation")
        if name not in ALLOWED_OPERATIONS:
            rejected.append({"建议": item, "原因": "不在 V3 安全操作白名单"})
        else:
            accepted.append({"操作": name, "参数": item.get("参数", {}), "假设": item.get("假设", "")})
    return accepted, rejected
