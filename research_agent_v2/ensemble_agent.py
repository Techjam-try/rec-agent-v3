"""LLM-advised, validation-only two-model ensemble selection."""
from __future__ import annotations
import json, os, re, urllib.request
import numpy as np
from evaluate import evaluate

SYSTEM = """你是推荐系统 Ensemble Agent。根据给定的单模型 validation 指标和模型描述，用中文 JSON 推荐恰好两个互补模型：{"model_a":"名称","model_b":"名称","理由":"中文"}。只能从候选名称中选，不能引用 test、AUC、recall、p 值或不存在的模型。"""

def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item): return item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")

def suggest_pair(candidates, model="qwen-plus"):
    fallback = sorted(candidates, key=lambda x: x["metrics"]["primary"], reverse=True)[:2]
    if len(fallback) < 2: return None, {"mode":"insufficient_models"}
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not key: return (fallback[0]["name"], fallback[1]["name"]), {"mode":"deterministic_fallback"}
    context = {"候选": [{"name": x["name"], "metrics": x["metrics"], "code_diff": x["code_diff"]} for x in candidates]}
    body = json.dumps({"model":model,"temperature":.1,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(context,ensure_ascii=False,default=_json_default)}]},default=_json_default).encode()
    try:
        req=urllib.request.Request("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=45) as r: raw=json.loads(r.read().decode())
        text=raw["choices"][0]["message"]["content"]; plan=json.loads(re.search(r"\{.*\}",text,re.S).group(0))
        names={x["name"] for x in candidates}
        if plan.get("model_a") in names and plan.get("model_b") in names and plan["model_a"] != plan["model_b"]:
            return (plan["model_a"],plan["model_b"]), {"mode":"qwen","plan":plan,"response_text":text,"usage":raw.get("usage",{})}
    except Exception as exc: return (fallback[0]["name"],fallback[1]["name"]), {"mode":"fallback","error":repr(exc)}
    return (fallback[0]["name"],fallback[1]["name"]), {"mode":"invalid_plan_fallback"}

def _percentile_by_user(users, scores):
    out=np.empty(len(scores),np.float32); groups={}
    for i,user in enumerate(users): groups.setdefault(user,[]).append(i)
    for indices in groups.values():
        order=np.argsort(scores[indices],kind="mergesort"); ranks=np.empty(len(indices),np.float32); ranks[order]=np.arange(len(indices),dtype=np.float32)
        out[indices]=ranks/max(len(indices)-1,1)
    return out

def blend(users, labels, score_a, score_b):
    a,b=_percentile_by_user(users,score_a),_percentile_by_user(users,score_b)
    best=None
    for weight in np.arange(.2,.81,.1):
        metrics=evaluate(users,labels,weight*a+(1-weight)*b)
        if best is None or metrics["primary"]>best["metrics"]["primary"]: best={"weight_a":round(float(weight),2),"weight_b":round(float(1-weight),2),"metrics":metrics}
    return best
