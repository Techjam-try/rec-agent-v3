"""V3 bounded autonomous experiment orchestration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from research_agent_v3.cache import load_or_encode
from research_agent_v3.checkpoints import save_checkpoint
from research_agent_v3.data import EncoderConfig
from research_agent_v3.models import ModelConfig
from research_agent_v3.recipe_compiler import compile_recipes
from research_agent_v3.runner import TrainingConfig, train_candidate

def run_encoded(encoded, categorical_vocab_size, field_count, video_vocab_size, output_dir, device, epochs, evaluator, smoke=False, encoder_state=None):
    output = Path(output_dir); (output/"models").mkdir(parents=True, exist_ok=True)
    recipes = compile_recipes([{"操作":"din_history_attention"},{"操作":"multibehavior_auxiliary"}], gpu_available=device != "cpu")
    if smoke: recipes = recipes[:3] + [next(r for r in recipes if r["family"] == "din")]
    events=[]; best=-float("inf")
    for index, recipe in enumerate(recipes, 1):
        config=ModelConfig(recipe["family"], categorical_vocab_size, field_count, video_vocab_size,
                           embedding_dim=recipe["embedding_dim"], mlp_dims=tuple(recipe["mlp_dims"]), aux_tasks=tuple(recipe["aux_tasks"]))
        event={"iteration":index,"name":recipe["name"],"family":recipe["family"],"hypothesis":f"验证 {recipe['name']} 在官方指标上的收益","status":"started"}
        try:
            result=train_candidate(encoded, config, TrainingConfig(device=device,epochs=epochs,batch_size=4 if smoke else 16384,amp=device!="cpu",aux_weight=recipe["aux_weight"]),evaluator)
            event.update({"status":"completed","metrics":result.metrics,"best_epoch":result.best_epoch,"device":result.device_name,"elapsed_seconds":result.elapsed_seconds,"recovery_events":result.recovery_events})
            metadata={"recipe":recipe,"event":event}
            if encoder_state is not None: metadata["encoder_state"]=encoder_state.to_dict()
            checkpoint=output/"models"/f"{recipe['name']}.pt"; save_checkpoint(checkpoint,result.state_dict,config,metadata)
            if result.metrics["primary"] > best:
                best=result.metrics["primary"]; save_checkpoint(output/"validation_best.pt",result.state_dict,config,metadata)
        except Exception as exc: event.update({"status":"failed","error_recovery":repr(exc)})
        events.append(event)
    (output/"runs.json").write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding="utf-8")
    return events

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--device",default="auto"); parser.add_argument("--epochs",type=int,default=12); parser.add_argument("--cache-dir",default=".cache/v3")
    args=parser.parse_args(); encoded,state,_=load_or_encode(args.data_dir,EncoderConfig(),args.cache_dir)
    try: from evaluate import evaluate
    except ImportError as exc: raise SystemExit("Run from the Starter Kit root so official evaluate.py is importable") from exc
    run_encoded(encoded,state.categorical_vocab_size,len(state.fields),state.video_vocab_size,args.output_dir,args.device,args.epochs,evaluate,encoder_state=state)
if __name__ == "__main__": main()
