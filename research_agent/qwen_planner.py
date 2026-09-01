"""Small, dependency-free Qwen planner for the autonomous research demo."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

SYSTEM_PROMPT = """You are the Research Manager of RecResearcher, an autonomous
recommendation ML engineer. Diagnose before optimisation. You never access a
hidden test set. Use the supplied data profile, bottleneck diagnosis, experiment
tree and research memory. Choose exactly one strategy from allowed_strategies and
return JSON only:
{"strategy":"pairwise_bpr|history_bpr|multitask_engagement|history_multitask|stop",
 "bottleneck":"loss_alignment|sequence_interest|sparse_primary_signal|robustness_check",
 "hypothesis":"one sentence", "evidence":"one sentence grounded in prior validation metrics",
 "why":"one sentence connecting evidence to research memory", "expected_metric":"GAUC|nDCG@5|primary",
 "risk":"one sentence", "expected_utility":"low|medium|high"}
Rules: never choose blocked strategies; do not propose static feature expansion,
larger FM embeddings, test evaluation, external data, arbitrary code changes or
new dependencies. Prefer an untried strategy when untried_strategies is nonempty.
Prefer stopping after three non-material improvements
(<= 0.002)."""

DATA_SYSTEM_PROMPT = """You are a recommendation-system data scientist. Analyse only
the supplied train/validation EDA; never request or infer hidden-test labels.
Return JSON only:
{"findings":[{"evidence":"numeric evidence from EDA","implication":"research implication"}],
 "bottleneck":"loss_alignment|sequence_interest|sparse_primary_signal|temporal_shift",
 "recommended_strategies":["pairwise_bpr|history_bpr|multitask_engagement|history_multitask"],
 "avoid":["one concise warning"], "next_hypothesis":"one falsifiable validation-only experiment"}
Do not recommend static feature expansion, larger embeddings, external data,
test evaluation, or arbitrary code changes. Treat validation-window activity as
evaluation exposure count, not as a user's available train history. If a prior
experiment reports a severe BPR regression, do not recommend a BPR strategy.
Recommendations must use the safe strategy allowlist and mention uncertainty
when evidence is insufficient."""


def _json_default(value):
    """Allow NumPy validation-metric scalars in the Qwen request context."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("Qwen response did not contain a JSON object")
    plan = json.loads(match.group(0))
    allowed = {"pairwise_bpr", "history_bpr", "multitask_engagement", "history_multitask", "stop"}
    if plan.get("strategy") not in allowed:
        raise ValueError("Qwen selected a strategy outside the safe allowlist")
    return plan


def ask_qwen_with_trace(context, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, timeout=45):
    """Return plan, usage and an audit-safe request/response trace (never the API key)."""
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not key:
        raise RuntimeError("Set DASHSCOPE_API_KEY (or QWEN_API_KEY) before using the Qwen planner.")
    payload = json.dumps({"model": model, "temperature": 0.2, "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": json.dumps(context, ensure_ascii=False,
                                                                                 default=_json_default)}]}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Qwen API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Qwen API: {exc.reason}") from exc
    content = raw["choices"][0]["message"]["content"]
    trace = {"model": model, "system_prompt": SYSTEM_PROMPT, "user_context": context,
             "response_text": content}
    return _extract_json(content), raw.get("usage", {}), trace


def ask_qwen(context, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, timeout=45):
    """Backward-compatible planner API returning only the plan and token usage."""
    plan, usage, _ = ask_qwen_with_trace(context, model, base_url, timeout)
    return plan, usage


def ask_qwen_data_report(eda, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, timeout=45):
    """Request an audit-safe EDA memo without relaxing experiment guardrails."""
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not key:
        raise RuntimeError("Set DASHSCOPE_API_KEY (or QWEN_API_KEY) before using Qwen data research.")
    payload = json.dumps({"model": model, "temperature": 0.1, "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": DATA_SYSTEM_PROMPT},
                                       {"role": "user", "content": json.dumps(eda, ensure_ascii=False,
                                                                                 default=_json_default)}]}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Qwen API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Qwen API: {exc.reason}") from exc
    content = raw["choices"][0]["message"]["content"]
    report = json.loads(re.search(r"\{.*\}", content, flags=re.S).group(0))
    allowed = {"pairwise_bpr", "history_bpr", "multitask_engagement", "history_multitask"}
    report["recommended_strategies"] = [item for item in report.get("recommended_strategies", []) if item in allowed]
    trace = {"model": model, "system_prompt": DATA_SYSTEM_PROMPT, "eda": eda, "response_text": content}
    return report, raw.get("usage", {}), trace
