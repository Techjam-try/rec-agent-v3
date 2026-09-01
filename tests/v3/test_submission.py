import csv, math
import numpy as np
import pytest
from research_agent_v3.make_submission import write_submission_csv
from research_agent_v3.make_submission import resolve_inference_device

def test_submission_csv_has_exact_schema_alignment_and_finite_scores(tmp_path):
    rows=[{"user":"u1","video":"v1"},{"user":"u1","video":"v1"}]; path=tmp_path/"submission.csv"
    write_submission_csv(rows,np.array([.2,.8]),path)
    with path.open(encoding="utf-8") as handle: saved=list(csv.DictReader(handle))
    assert list(saved[0]) == ["row_id","user_id","video_id","score"]
    assert [int(r["row_id"]) for r in saved] == [0,1]
    assert all(math.isfinite(float(r["score"])) for r in saved)

def test_submission_rejects_nan(tmp_path):
    with pytest.raises(ValueError): write_submission_csv([{"user":"u","video":"v"}],np.array([np.nan]),tmp_path/"x.csv")

def test_explicit_cuda_inference_never_silently_falls_back(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_inference_device("cuda")
