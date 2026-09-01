"""Fast, inspectable Qwen-driven planning demo (no hidden-test access)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_agent.qwen_planner import ask_qwen


def latest_event(path):
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events:
        raise RuntimeError("No experiment log found. Run demo_agent.py or agent.py first.")
    return events[-1]


def main():
    parser = argparse.ArgumentParser(description="Ask Qwen to analyse a validation-only experiment log")
    parser.add_argument("--log", default="./demo_runs/demo_log.jsonl")
    parser.add_argument("--output", default="./demo_runs/qwen_next_plan.json")
    parser.add_argument("--model", default="qwen-plus")
    args = parser.parse_args()
    event = latest_event(args.log)
    context = {"task": "KuaiRand-Pure user-local long_view ranking; validation only",
               "official_validation_primary": 0.6016, "latest_event": event,
               "available_strategies": ["pairwise_bpr", "history_bpr", "multitask_engagement", "history_multitask"]}
    started = time.time()
    plan, usage = ask_qwen(context, model=args.model)
    artifact = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "context": context,
                "qwen_plan": plan, "token_usage": usage, "elapsed_seconds": round(time.time()-started, 2),
                "manual_interventions": 0, "hidden_test_access": False}
    Path(args.output).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
