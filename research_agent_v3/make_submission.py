"""Label-free V3 checkpoint inference and Starter-Kit CSV export."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np, torch
from research_agent_v3.checkpoints import load_checkpoint
from research_agent_v3.data import EncoderState, load_test_features, transform_inference

def write_submission_csv(rows, scores, output):
    scores=np.asarray(scores)
    if len(rows)!=len(scores): raise ValueError("row count does not match scores")
    if not np.isfinite(scores).all(): raise ValueError("scores contain NaN or Inf")
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["row_id","user_id","video_id","score"])
        for index,(row,score) in enumerate(zip(rows,scores)): writer.writerow([index,row["user"],row["video"],f"{float(score):.12g}"])

def resolve_inference_device(requested):
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"auto","cpu","cuda"}: raise ValueError("device must be auto, cpu, or cuda")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")

def predict_checkpoint(checkpoint, data_dir, device="auto", batch_size=32768):
    actual=resolve_inference_device(device)
    model,metadata=load_checkpoint(checkpoint,actual); state=EncoderState.from_dict(metadata["encoder_state"])
    rows=load_test_features(data_dir); encoded=transform_inference([],rows,state); model.to(actual).eval(); scores=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            sl=slice(start,start+batch_size); args=[torch.as_tensor(encoded[k][sl],device=actual) for k in ("X","candidate_video_ids","history_video_ids","history_mask")]
            scores.append(model(*args)["long_view_logits"].float().cpu().numpy())
    if not scores: raise ValueError("test window contains no rows")
    return rows,np.concatenate(scores)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--output",required=True); parser.add_argument("--device",default="auto")
    args=parser.parse_args(); rows,scores=predict_checkpoint(args.checkpoint,args.data_dir,args.device); write_submission_csv(rows,scores,args.output); print(f"Wrote {len(rows):,} label-free predictions to {args.output}")
if __name__=="__main__": main()
