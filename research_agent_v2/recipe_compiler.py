"""Compile safe LLM operations into concrete, bounded experiment recipes."""
from __future__ import annotations


def compile_recipes(operations):
    """Return a small, reproducible candidate queue; no arbitrary code is executed."""
    recipes = [{"name": "official_fm_control", "use_author_history": False, "use_aux": False,
                "aux_weight": 0.0, "model_family":"fm", "code_diff": "官方 5-field FM 对照组。"},
               {"name":"deepfm_primary", "use_author_history":False, "use_aux":False, "aux_weight":0.,
                "model_family":"deepfm", "epochs":6, "code_diff":"PyTorch DeepFM：FM 二阶交叉加非线性 DNN。"},
               {"name":"dcn_primary", "use_author_history":False, "use_aux":False, "aux_weight":0.,
                "model_family":"dcn", "epochs":6, "code_diff":"PyTorch DCN：Cross Network 加 DNN。"}]
    names = {item["操作"] for item in operations}
    if "auxiliary_weight_search" in names:
        for weight in (.03, .10, .20):
            recipes.append({"name": f"multitask_w{weight:.2f}", "use_author_history": False,
                            "use_aux": True, "aux_weight": weight,
                            "code_diff": f"多任务辅助损失权重={weight:.2f}。"})
    if "recent_positive_author_history" in names:
        recipes.append({"name": "author_history_multitask", "use_author_history": True,
                        "use_aux": "auxiliary_weight_search" in names,
                        "aux_weight": .10 if "auxiliary_weight_search" in names else 0.0,
                        "code_diff": "新增训练期因果 last_positive_author 字段；验证期历史冻结。"})
    if {"positive_history_count_bucket", "hour_weekday_context", "watch_time_auxiliary"} & names:
        recipes.append({"name": "context_history_multitask", "use_author_history": "recent_positive_author_history" in names,
                        "use_history_count": "positive_history_count_bucket" in names, "use_hour_weekday": "hour_weekday_context" in names,
                        "watch_bucket": "watch_time_auxiliary" in names, "use_aux": True, "aux_weight": .10,
                        "code_diff": "按 recipe 加入历史强度桶、小时星期上下文和高观看比率辅助目标。"})
    return recipes


def deferred_operations(operations):
    """Operations retained for future iterations but not falsely claimed as executed."""
    executable = {"recent_positive_author_history", "auxiliary_weight_search", "positive_history_count_bucket", "hour_weekday_context", "watch_time_auxiliary"}
    return [operation for operation in operations if operation["操作"] not in executable]
