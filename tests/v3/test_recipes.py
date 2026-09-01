from research_agent_v3.operations import validate_operations
from research_agent_v3.recipe_compiler import compile_recipes


def test_compiler_retains_controls_and_adds_requested_v3_candidates():
    accepted, rejected = validate_operations([{"操作":"din_history_attention"}, {"操作":"multibehavior_auxiliary"}])
    names = [item["name"] for item in compile_recipes(accepted, gpu_available=True)]
    assert rejected == []
    assert names[:3] == ["deepfm_v2_control", "dcn_v2_control", "deepfm_gpu"]
    assert "din_primary" in names and "din_multitask" in names

def test_default_full_search_includes_multitask_candidate():
    accepted, _ = validate_operations([{"操作":"din_history_attention"}, {"操作":"multibehavior_auxiliary"}])
    assert any(r["name"] == "din_multitask" for r in compile_recipes(accepted, True))


def test_unknown_operation_is_rejected():
    accepted, rejected = validate_operations([{"操作":"execute_arbitrary_python"}])
    assert accepted == [] and rejected[0]["原因"] == "不在 V3 安全操作白名单"
