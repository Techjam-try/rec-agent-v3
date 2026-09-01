import json
from pathlib import Path

def test_colab_notebook_contains_reproducible_gpu_workflow():
    notebook=json.loads(Path("colab/RecResearcher_V3_GPU.ipynb").read_text(encoding="utf-8"))
    source="\n".join("".join(cell.get("source",[])) for cell in notebook["cells"])
    for required in ("nvidia-smi","KuaiRand-Pure.tar.gz","0820331067a3784d9691136f772b35a7","pytest tests/v3","--device cuda","final_ensemble","submit.py --check"):
        assert required in source
    assert "API_KEY=" not in source
