"""Ask Qwen to turn validation-safe EDA into an experiment recommendation."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research_agent.data import load_train_valid
from research_agent.data_analyst import analyze
from research_agent.qwen_planner import ask_qwen_data_report


def main():
    parser = argparse.ArgumentParser(description="Validation-only EDA + Qwen research memo")
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="./rec_researcher_run")
    parser.add_argument("--qwen-model", default="qwen-plus")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    eda = analyze(load_train_valid(args.data_dir))
    eda_path = os.path.join(args.output_dir, "data_eda.json")
    with open(eda_path, "w", encoding="utf-8") as fh:
        json.dump(eda, fh, ensure_ascii=False, indent=2)
    report, usage, trace = ask_qwen_data_report(eda, model=args.qwen_model)
    report_path = os.path.join(args.output_dir, "data_research.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "usage": usage, "trace": trace}, fh, ensure_ascii=False, indent=2)
    print("[data_eda]", eda_path)
    print("[qwen_data_research]", report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
