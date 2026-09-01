"""Finish a failed ensemble stage without rerunning completed single models."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from research_agent_v2.data import load_train_valid
from research_agent_v2.ensemble_agent import suggest_pair, blend
from research_agent_v2.reflector import reflect

def main():
    parser=argparse.ArgumentParser(description="Recover validation-only two-model ensemble")
    parser.add_argument("--run-dir",required=True); parser.add_argument("--data-dir",default="./KuaiRand-Pure/data"); parser.add_argument("--qwen-model",default="qwen-plus")
    args=parser.parse_args(); run=Path(args.run_dir)
    events=json.loads((run/"runs.json").read_text(encoding="utf-8"))
    completed=[{"name":e["recipe"]["name"],"metrics":e["metrics"],"code_diff":e["code_diff"],"score_path":str(run/"models"/(e["recipe"]["name"]+"_valid_scores.npy"))} for e in events if e["status"]=="completed"]
    pair,trace=suggest_pair(completed,args.qwen_model); by={x["name"]:x for x in completed}; splits=load_train_valid(args.data_dir); valid=splits["valid"]
    result=blend([r["user"] for r in valid],np.asarray([r["long_view"] for r in valid],np.float32),np.load(by[pair[0]]["score_path"]),np.load(by[pair[1]]["score_path"]))
    result.update({"models":list(pair),"llm_advice":trace,"accepted":result["metrics"]["primary"]>max(by[pair[0]]["metrics"]["primary"],by[pair[1]]["metrics"]["primary"])})
    (run/"ensemble_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding="utf-8")
    (run/"reflection_zh.json").write_text(json.dumps(reflect(events),ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding="utf-8")
    print(f"[ensemble] {pair[0]} + {pair[1]} | {result['metrics']['primary']:.6f} | accepted={result['accepted']}")
if __name__=="__main__": main()
