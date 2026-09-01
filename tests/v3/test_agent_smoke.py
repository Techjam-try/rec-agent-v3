import json
from research_agent_v3.agent import run_encoded
from tests.v3.test_runner import encoded_fixture, evaluator

def test_agent_runs_bounded_candidates_and_writes_audit(tmp_path):
    events = run_encoded(encoded_fixture(), categorical_vocab_size=32, field_count=5, video_vocab_size=8,
                         output_dir=tmp_path, device="cpu", epochs=1, evaluator=evaluator, smoke=True)
    assert {event["family"] for event in events} == {"deepfm", "dcn", "din"}
    assert all(event["status"] == "completed" for event in events)
    persisted = json.loads((tmp_path/"runs.json").read_text(encoding="utf-8"))
    assert len(persisted) == len(events) and (tmp_path/"validation_best.pt").exists()
