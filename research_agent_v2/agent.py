"""V2 entry point: Chinese data research -> safe operation recipe.

This milestone creates the autonomous research front end. Candidate trainers
will consume emitted recipes in the next milestone; V1 remains untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_agent.data_analyst import analyze
from research_agent_v2.data import load_train_valid
from research_agent_v2.data_agent import ask_data_agent
from research_agent_v2.field_registry import manifest
from research_agent_v2.operations import ALLOWED_OPERATIONS, validate_operations
from research_agent_v2.planner import ask
from research_agent_v2.recipe_compiler import compile_recipes, deferred_operations
from research_agent_v2.runner import run_recipe
from research_agent_v2.reflector import reflect
from research_agent_v2.ensemble_agent import suggest_pair, blend


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(value):
    """Persist NumPy metrics without turning a successful run into an error."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def fallback(eda):
    """A deterministic Chinese plan keeps the run useful when Qwen is unavailable."""
    coverage = eda["validation_user_train_history"]["users_with_at_least_5_train_positive_events"]
    return {"中文结论": [f"{coverage:.2%} 的验证用户至少有 5 条训练期正反馈，适合验证近期兴趣聚合。",
                        "click 等反馈只能作为训练辅助标签，不能作为推理输入。"],
            "字段决策": [{"字段": "video_features_statistic_pure.csv", "决定": "暂不使用", "原因": "时间窗口未审计。"}],
            "操作计划": [{"操作": "recent_positive_author_history", "参数": {"窗口": 3},
                         "假设": "近期正反馈作者比单条视频 ID 更具泛化性。"},
                        {"操作": "auxiliary_weight_search", "参数": {"任务": ["click", "like", "hate"]},
                         "假设": "区分稠密正反馈和负反馈可改善主任务。"}],
            "风险": ["操作仅是候选，必须由 validation 指标决定保留与否。"], "下一轮目标": "生成并运行第一个安全 recipe。"}


def fallback_data_report(eda):
    coverage = eda["validation_user_train_history"]["users_with_at_least_5_train_positive_events"]
    return {"数据结论": [f"{coverage:.2%} 的验证用户有至少 5 条训练期正反馈，历史特征具有覆盖基础。"],
            "字段分类": [{"字段": "is_click", "决定": "训练辅助标签", "原因": "曝光后行为。"},
                         {"字段": "video_features_statistic_pure.csv", "决定": "潜在泄漏暂停", "原因": "时间窗口未审计。"}],
            "聚合建议": [{"操作": "recent_positive_author_history", "理由": "测试训练期兴趣泛化。", "风险": "序列过稀。"}],
            "泄漏警告": ["不读取 test 标签。"]}


def main():
    parser = argparse.ArgumentParser(description="RecResearcher V2: 中文数据研究与自动操作计划")
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="./v2_runs/run_001")
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument("--no-qwen", action="store_true", help="只运行确定性研究计划")
    parser.add_argument("--execute", action="store_true", help="自动编译 recipe 并训练验证候选")
    parser.add_argument("--candidate-epochs", type=int, default=18)
    parser.add_argument("--max-hours", type=float, default=2.0)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    eda = analyze(load_train_valid(args.data_dir))
    data_context = {"数据范围": "仅 train + validation；未加载 test", "字段安全清单": manifest(), "数据摘要": eda}
    write(out / "field_manifest.json", manifest()); write(out / "data_eda.json", eda)
    trace, recovery, data_trace = None, None, None
    try:
        if args.no_qwen:
            data_report, plan = fallback_data_report(eda), fallback(eda)
        else:
            data_report, data_trace = ask_data_agent(data_context, args.qwen_model)
            context = {**data_context, "Data Agent 中文报告": data_report, "可执行操作": ALLOWED_OPERATIONS}
            plan, trace = ask(context, args.qwen_model)
    except Exception as exc:
        data_report, plan, recovery = fallback_data_report(eda), fallback(eda), repr(exc)
    accepted, rejected = validate_operations(plan.get("操作计划"))
    deferred = deferred_operations(accepted)
    recipe = {"scope": "train + validation only; no hidden test", "data_agent_report": data_report,
              "data_agent_trace": data_trace, "plan": plan,
              "accepted_operations": accepted, "rejected_operations": rejected,
              "deferred_operations": deferred,
              "recovery": recovery, "qwen_trace": trace}
    write(out / "research_plan_zh.json", recipe)
    print("\n[Data Agent 中文结论]")
    for index, item in enumerate(data_report["数据结论"], 1): print(f"{index}. {item}")
    print("\n[Research Planner 中文结论]")
    for index, item in enumerate(plan["中文结论"], 1): print(f"{index}. {item}")
    print("\n[Agent 已批准的操作]")
    for item in accepted: print(f"- {item['操作']} {item['参数']}：{item['假设']}")
    if deferred:
        print("\n[已记录、待后续执行器支持的操作]")
        for item in deferred: print(f"- {item['操作']}")
    if args.execute:
        import numpy as np
        splits, events, best_score, completed = load_train_valid(args.data_dir), [], -float("inf"), []
        (out / "models").mkdir(exist_ok=True)
        deadline = time.time() + args.max_hours * 3600
        for index, candidate in enumerate(compile_recipes(accepted), 1):
            event = {"iteration": index, "recipe": candidate, "hypothesis": candidate["code_diff"],
                     "code_diff": candidate["code_diff"], "status": "started", "error_recovery": None}
            try:
                model, metrics, training = run_recipe(splits, candidate, epochs=args.candidate_epochs,
                                                      seed=index, deadline=deadline)
                event.update({"status": "completed", "metrics": metrics, "training": training})
                encoded, _, _ = __import__("research_agent_v2.data", fromlist=["encode"]).encode(
                    splits, candidate["use_author_history"], candidate.get("use_history_count",False), candidate.get("use_hour_weekday",False), candidate.get("watch_bucket",False))
                if candidate.get("model_family") in {"deepfm","dcn"}:
                    import torch
                    model.eval()
                    with torch.no_grad(): scores=model(torch.from_numpy(encoded["valid"]["X"]).long()).cpu().numpy()
                else:
                    scores=model.predict(encoded["valid"]["X"])
                score_path=out/"models"/(candidate["name"]+"_valid_scores.npy")
                np.save(score_path,scores)
                if candidate.get("model_family") in {"deepfm","dcn"}:
                    import torch
                    torch.save({"state_dict":model.state_dict(),"recipe":candidate},out/"models"/(candidate["name"]+".pt"))
                else:
                    np.savez_compressed(out/"models"/(candidate["name"]+".npz"),**model.state_dict())
                completed.append({"name":candidate["name"],"metrics":metrics,"code_diff":candidate["code_diff"],"score_path":str(score_path)})
                if metrics["primary"] > best_score:
                    best_score = metrics["primary"]
                    import numpy as np
                    if candidate.get("model_family") in {"deepfm","dcn"}:
                        import torch
                        torch.save({"state_dict":model.state_dict(),"recipe":candidate,"event":event},out/"validation_best.pt")
                    else:
                        np.savez_compressed(out / "validation_best.npz", **model.state_dict(),
                                            metadata=json.dumps(event, ensure_ascii=False, default=json_default))
            except Exception as exc:
                event.update({"status": "failed", "error_recovery": repr(exc)})
            events.append(event)
            print(f"[iter {index:02d}] {candidate['name']}: {event['status']} | "
                  f"{event.get('metrics', {}).get('primary', 'error')}")
        write(out / "runs.json", events)
        if len(completed) >= 2:
            try:
                pair, ensemble_trace = suggest_pair(completed, args.qwen_model)
                by_name={item["name"]:item for item in completed}; a,b=by_name[pair[0]],by_name[pair[1]]
                ensemble=blend(encoded["valid"]["users"],encoded["valid"]["y"],np.load(a["score_path"]),np.load(b["score_path"]))
                ensemble.update({"models":list(pair),"llm_advice":ensemble_trace,"accepted":ensemble["metrics"]["primary"]>max(a["metrics"]["primary"],b["metrics"]["primary"])})
                write(out/"ensemble_result.json",ensemble)
                print(f"[ensemble] {pair[0]} + {pair[1]} | {ensemble['metrics']['primary']:.6f} | accepted={ensemble['accepted']}")
            except Exception as exc:
                write(out/"ensemble_result.json",{"status":"failed","error_recovery":repr(exc)})
                print(f"[ensemble] failed; single-model results preserved: {exc!r}")
        if any(event["status"] == "completed" for event in events):
            reflection = reflect(events); write(out / "reflection_zh.json", reflection)
            print("\n[中文反思]", reflection["结论"])
    print(f"\n[审计文件] {out / 'research_plan_zh.json'}")


if __name__ == "__main__":
    main()
