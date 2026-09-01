"""Chinese Qwen planner that produces an auditable operation plan."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


SYSTEM = """你是 RecResearcher V2 的推荐系统研究负责人。只使用提供的 train+validation 数据摘要，绝不请求或推断 hidden test 标签。请用中文返回 JSON：
{"中文结论":["带数值证据的结论"],"字段决策":[{"字段":"字段名","决定":"输入|辅助标签|暂不使用|无偏验证","原因":"中文"}],"操作计划":[{"操作":"recent_positive_author_history|positive_history_count_bucket|hour_weekday_context|auxiliary_weight_search|watch_time_auxiliary|random_valid_audit","参数":{},"假设":"可证伪的中文假设"}],"风险":["中文"],"下一轮目标":"中文"}
规则：曝光后行为只能作为训练辅助标签；全局视频统计在时间范围未审计前暂不使用；tab、author 和 duration 已属于官方 FM 输入，不能把再次加入它们当作新操作。所有成功/失败标准只能使用官方 GAUC、nDCG@5 或 primary，禁止使用 AUC、p 值、recall、precision、loss 或 test 指标。不要提出外部训练数据、测试集评估、静态特征堆叠、增大 embedding 或任意代码。每项操作必须来自允许列表。"""


def _extract(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Qwen 未返回 JSON 对象")
    return json.loads(match.group(0))


def ask(context, model="qwen-plus", timeout=45):
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY")
    body = json.dumps({"model": model, "temperature": .15, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]}).encode()
    request = urllib.request.Request("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Qwen HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen 网络错误：{exc.reason}") from exc
    text = raw["choices"][0]["message"]["content"]
    return _extract(text), {"model": model, "response_text": text, "usage": raw.get("usage", {})}
