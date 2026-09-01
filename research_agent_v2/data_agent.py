"""LLM-facing Data Agent: interprets audited EDA, never raw hidden-test data."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


SYSTEM = """你是 RecResearcher 的 Data Agent。你只能解释给定的、由 Python 从 train+validation 计算出的 EDA 和字段安全清单；绝不索取原始 hidden test 或其标签。用中文返回 JSON：
{"数据结论":["每条必须引用 EDA 数值"],"字段分类":[{"字段":"名称","决定":"输入|训练辅助标签|潜在泄漏暂停|随机曝光审计","原因":"中文"}],"聚合建议":[{"操作":"recent_positive_author_history|positive_history_count_bucket|hour_weekday_context|auxiliary_weight_search|watch_time_auxiliary|random_valid_audit","理由":"中文","风险":"中文"}],"泄漏警告":["中文"]}
规则：曝光后行为不得作为推理输入；视频全局统计未经时间范围审计不得使用；tab、author、duration 已在官方 FM 中，不得重复包装为新特征；不要用 AUC、p 值、recall 或 test 指标。"""


def ask_data_agent(context, model="qwen-plus", timeout=45):
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not key: raise RuntimeError("未设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY")
    payload = json.dumps({"model": model, "temperature": .1, "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]}).encode()
    request = urllib.request.Request("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", payload,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response: raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc: raise RuntimeError(f"Data Agent Qwen HTTP {exc.code}") from exc
    except urllib.error.URLError as exc: raise RuntimeError(f"Data Agent 网络错误：{exc.reason}") from exc
    text = raw["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    if not match: raise ValueError("Data Agent 未返回 JSON")
    return json.loads(match.group(0)), {"model": model, "response_text": text, "usage": raw.get("usage", {})}
