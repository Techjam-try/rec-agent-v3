# RecResearcher V3 Colab GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested `research_agent_v3` package and Colab notebook that run causal DIN, configurable DeepFM/DCN, and multi-task recommendation experiments on CUDA while preserving KuaiRand-Pure evaluation and test-label isolation.

**Architecture:** V3 is an independent package beside unchanged V2. A serializable encoder state creates categorical and causal-history arrays; configurable PyTorch models consume batches through a device-aware runner; bounded recipes feed the existing autonomous orchestration concepts; checkpoints fully reconstruct label-free inference.

**Tech Stack:** Python 3.10+, NumPy, PyTorch 2.x, pytest, Google Colab, official KuaiRand `evaluate.py`.

**Spec:** `docs/superpowers/specs/2026-09-01-research-agent-v3-design.md`

## Global Constraints

- Keep `research_agent_v2` byte-for-byte unchanged.
- Train dates: `20220408..20220421`; validation dates: `20220422..20220428`; test dates: `20220429..20220508`.
- Never parse test labels or post-exposure test actions.
- Leave official `evaluate.py` unchanged and select models only with official validation primary.
- Use CUDA for full Colab experiments; permit CPU for tests and fallback.
- Keep arbitrary LLM-generated code outside the execution path; only bounded recipes may execute.
- All performance claims must come from persisted official validation output.

---

### Task 1: V3 package and causal encoder

**Files:**
- Create: `research_agent_v3/__init__.py`
- Create: `research_agent_v3/data.py`
- Create: `tests/v3/test_data.py`

**Interfaces:**
- Produces: `load_train_valid(data_dir) -> dict[str, list[dict]]`
- Produces: `load_test_features(data_dir) -> list[dict]`
- Produces: `EncoderConfig`, `EncoderState`, `fit_transform(splits, config)`, and `transform_inference(train_rows, inference_rows, state)`.
- Array split keys: `X`, `history_video_ids`, `history_mask`, `y`, `aux`, `users`.

- [ ] **Step 1: Write failing date-boundary and label-isolation tests**

```python
def test_test_loader_does_not_request_labels(tmp_path, monkeypatch):
    rows = load_test_features(make_fixture_dataset(tmp_path))
    assert rows and set(rows[0]) == {"date", "hour", "user", "video", "author", "tab", "duration"}

def test_load_train_valid_uses_fixed_dates(tmp_path):
    splits = load_train_valid(make_fixture_dataset(tmp_path))
    assert {r["date"] for r in splits["train"]} == {20220408, 20220421}
    assert {r["date"] for r in splits["valid"]} == {20220422, 20220428}
```

- [ ] **Step 2: Run the tests and confirm missing-module failures**

Run: `python -m pytest tests/v3/test_data.py -v`

Expected: FAIL because `research_agent_v3.data` does not exist.

- [ ] **Step 3: Implement the minimal fixed-date loaders**

Implement separate parsing functions so `load_test_features` reads only inference columns. Preserve V2 field names and `long_view`/auxiliary mapping only in train/validation.

- [ ] **Step 4: Add failing causal-history tests**

```python
def test_positive_history_is_causal_and_validation_is_frozen():
    encoded, state = fit_transform(synthetic_splits(), EncoderConfig(max_history=3))
    assert encoded["train"]["history_video_ids"][0].tolist() == [0, 0, 0]
    assert encoded["train"]["history_video_ids"][2, -1] == state.video_ids["v1"]
    assert np.array_equal(encoded["valid"]["history_video_ids"][0], encoded["valid"]["history_video_ids"][1])

def test_padding_and_unknown_are_distinct():
    _, state = fit_transform(synthetic_splits(), EncoderConfig(max_history=3))
    assert state.padding_id != state.unknown_video_id
```

- [ ] **Step 5: Verify the history tests fail for the missing encoder**

Run: `python -m pytest tests/v3/test_data.py -v`

Expected: loader tests PASS; encoder tests FAIL because `fit_transform` is missing.

- [ ] **Step 6: Implement serializable causal encoding**

Use a deque per user. Encode a row before appending its video when its training `long_view` is positive. Snapshot training histories before validation and never update them from validation. Store categorical vocabularies, offsets, video vocabulary, duration edges, fields, and configuration in `EncoderState`.

- [ ] **Step 7: Run and commit**

Run: `python -m pytest tests/v3/test_data.py -v`

Expected: PASS.

Commit: `git add research_agent_v3 tests/v3/test_data.py && git commit -m "feat: add causal V3 feature encoder"`

---

### Task 2: Fingerprinted feature cache

**Files:**
- Create: `research_agent_v3/cache.py`
- Create: `tests/v3/test_cache.py`

**Interfaces:**
- Produces: `cache_fingerprint(data_dir: Path, config: EncoderConfig) -> str`
- Produces: `load_or_encode(data_dir, config, cache_dir) -> tuple[dict, EncoderState, bool]`; final boolean is `cache_hit`.

- [ ] **Step 1: Write failing cache reuse and invalidation tests**

```python
def test_cache_reuses_identical_sources(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    first = load_or_encode(data_dir, EncoderConfig(max_history=5), tmp_path / "cache")
    second = load_or_encode(data_dir, EncoderConfig(max_history=5), tmp_path / "cache")
    assert first[2] is False and second[2] is True

def test_cache_invalidates_when_recipe_changes(tmp_path):
    data_dir = make_fixture_dataset(tmp_path)
    load_or_encode(data_dir, EncoderConfig(max_history=5), tmp_path / "cache")
    assert load_or_encode(data_dir, EncoderConfig(max_history=10), tmp_path / "cache")[2] is False
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/v3/test_cache.py -v`

Expected: FAIL because `research_agent_v3.cache` is missing.

- [ ] **Step 3: Implement cache fingerprints and atomic artifacts**

Hash schema version, normalized source names, sizes, mtimes, fixed date ranges, and `dataclasses.asdict(config)`. Store arrays in `.npz` and encoder state/manifest in JSON. Write to temporary files and replace only after successful serialization.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/v3/test_cache.py tests/v3/test_data.py -v`

Expected: PASS.

Commit: `git add research_agent_v3/cache.py tests/v3/test_cache.py && git commit -m "feat: cache deterministic V3 features"`

---

### Task 3: Configurable GPU models and multi-task objective

**Files:**
- Create: `research_agent_v3/models.py`
- Create: `tests/v3/test_models.py`

**Interfaces:**
- Produces: `ModelConfig(family, vocab_size, field_count, video_vocab_size, embedding_dim, mlp_dims, cross_layers, aux_tasks)`.
- Produces: `make_model(config) -> nn.Module`.
- Model call: `model(X, history_video_ids, history_mask) -> dict[str, Tensor]` with `long_view_logits` and optional `aux_logits`.
- Produces: `multitask_loss(outputs, y, aux_targets, aux_weight) -> Tensor`.

- [ ] **Step 1: Write failing output-shape tests for all families**

```python
@pytest.mark.parametrize("family", ["deepfm", "dcn", "din"])
def test_model_returns_primary_logits(family):
    model = make_model(tiny_config(family))
    out = model(*tiny_batch())
    assert out["long_view_logits"].shape == (4,)
    assert torch.isfinite(out["long_view_logits"]).all()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/v3/test_models.py -v`

Expected: FAIL because V3 models are missing.

- [ ] **Step 3: Implement shared categorical representation, DeepFMV3, and DCNV3**

Use configurable embeddings and MLP widths. Return dictionaries, not bare tensors, so later auxiliary heads do not change the runner interface.

- [ ] **Step 4: Implement DIN with mask-safe attention**

Compute candidate/history interactions from candidate video embeddings. Mask padding before softmax and return a zero interest vector for all-padding histories to avoid NaNs.

- [ ] **Step 5: Add failing padding-invariance and auxiliary-loss tests**

```python
def test_din_ignores_padded_positions():
    model = make_model(tiny_config("din"))
    x, history, mask = tiny_batch()
    changed = history.clone(); changed[~mask] = 7
    assert torch.allclose(model(x, history, mask)["long_view_logits"],
                          model(x, changed, mask)["long_view_logits"])

def test_auxiliary_heads_contribute_to_loss():
    config = dataclasses.replace(tiny_config("din"), aux_tasks=("click", "like"))
    outputs = make_model(config)(*tiny_batch())
    assert multitask_loss(outputs, torch.zeros(4), {"click": torch.ones(4), "like": torch.zeros(4)}, .1).isfinite()
```

- [ ] **Step 6: Implement auxiliary heads and loss, then run all model tests**

Run: `python -m pytest tests/v3/test_models.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

Commit: `git add research_agent_v3/models.py tests/v3/test_models.py && git commit -m "feat: add GPU-ready DIN and configurable neural models"`

---

### Task 4: Device-aware trainer, AMP, checkpoints, and recovery

**Files:**
- Create: `research_agent_v3/runner.py`
- Create: `tests/v3/test_runner.py`

**Interfaces:**
- Produces: `TrainingConfig(device="auto", batch_size=16384, amp=True, epochs=12, patience=3, learning_rate=1e-3, min_batch_size=1024)`.
- Produces: `resolve_device(requested) -> torch.device`.
- Produces: `train_candidate(encoded, encoder_state, model_config, training_config, evaluator=evaluate) -> TrainingResult`.
- `TrainingResult` contains model state, validation scores, official metrics, best epoch, elapsed seconds, device metadata, effective batch size, and recovery events.

- [ ] **Step 1: Write failing device and mini-batch tests**

```python
def test_auto_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"

def test_cpu_smoke_training_returns_official_metrics():
    result = train_candidate(tiny_encoded(), tiny_state(), tiny_config("deepfm"), TrainingConfig(device="cpu", epochs=2, batch_size=4))
    assert {"gauc", "ndcg", "primary"} <= result.metrics.keys()
    assert result.device_type == "cpu"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/v3/test_runner.py -v`

Expected: FAIL because the runner is missing.

- [ ] **Step 3: Implement batched device transfer, AMP, validation, and best-state restoration**

Keep full arrays in CPU memory and move only indexed batches. Use `torch.autocast` and `GradScaler` only for active CUDA AMP. Compute official validation metrics after batched inference and deep-copy the best CPU state dict.

- [ ] **Step 4: Add failing controlled-OOM test**

Inject a batch executor that raises `torch.cuda.OutOfMemoryError` once and assert the returned result uses half the initial batch size and contains one recovery event.

- [ ] **Step 5: Implement one bounded OOM retry**

Clear CUDA cache, rebuild the epoch iterator, halve the batch size no lower than `min_batch_size`, and retry once. Re-raise a second OOM.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/v3/test_runner.py tests/v3/test_models.py -v`

Expected: PASS; CUDA-only test skips when unavailable.

Commit: `git add research_agent_v3/runner.py tests/v3/test_runner.py && git commit -m "feat: add CUDA trainer and recovery metadata"`

---

### Task 5: Bounded recipes and autonomous V3 orchestration

**Files:**
- Create: `research_agent_v3/operations.py`
- Create: `research_agent_v3/recipe_compiler.py`
- Create: `research_agent_v3/agent.py`
- Create: `tests/v3/test_recipes.py`
- Create: `tests/v3/test_agent_smoke.py`

**Interfaces:**
- Produces: `validate_operations(items) -> (accepted, rejected)`.
- Produces: `compile_recipes(accepted, gpu_available) -> list[dict]`.
- Produces CLI options `--device`, `--batch-size`, `--amp`, `--cache-dir`, `--max-hours`, `--candidate-epochs`, and `--smoke`.

- [ ] **Step 1: Write failing whitelist and deterministic-queue tests**

```python
def test_compiler_keeps_controls_and_adds_din_multitask():
    accepted, _ = validate_operations([{"操作": "din_history_attention"}, {"操作": "multibehavior_auxiliary"}])
    names = [r["name"] for r in compile_recipes(accepted, gpu_available=True)]
    assert names[:3] == ["official_fm_control", "deepfm_v2_control", "dcn_v2_control"]
    assert "din_primary" in names and "din_multitask" in names

def test_unknown_operation_never_becomes_recipe():
    accepted, rejected = validate_operations([{"操作": "execute_arbitrary_python"}])
    assert accepted == [] and rejected
```

- [ ] **Step 2: Verify RED, then implement bounded operations and compiler**

Run: `python -m pytest tests/v3/test_recipes.py -v`

Expected before implementation: FAIL. Expected after implementation: PASS.

- [ ] **Step 3: Write a failing synthetic end-to-end agent smoke test**

The test invokes `agent.run(...)` with synthetic encoded data, one epoch, CPU, and no Qwen; it asserts `runs.json`, model checkpoints, validation scores, resource metadata, and a validation-best selection exist.

- [ ] **Step 4: Implement orchestration by adapting V2 concepts without modifying V2**

Compile deterministic fallback operations, load/cache encoded data once, train each candidate until budget/deadline, save reconstructable checkpoints, save validation score arrays, run ensemble search when at least two compatible candidates complete, and write reflection/audit files.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/v3/test_recipes.py tests/v3/test_agent_smoke.py -v`

Expected: PASS.

Commit: `git add research_agent_v3/operations.py research_agent_v3/recipe_compiler.py research_agent_v3/agent.py tests/v3 && git commit -m "feat: integrate V3 autonomous experiment loop"`

---

### Task 6: Reconstructable checkpoints and label-free submission

**Files:**
- Create: `research_agent_v3/checkpoints.py`
- Create: `research_agent_v3/make_submission.py`
- Create: `tests/v3/test_submission.py`

**Interfaces:**
- Produces: `save_checkpoint(path, model, model_config, encoder_state, recipe, training_result)`.
- Produces: `load_checkpoint(path, map_location) -> (model, metadata)`.
- Produces: `write_submission(data_dir, run_dir, output_path, device="auto", batch_size=32768)`.

- [ ] **Step 1: Write failing checkpoint round-trip test**

```python
def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    model, state, config = trained_tiny_din()
    expected = predict(model, tiny_inference_batch())
    save_checkpoint(tmp_path / "din.pt", model, config, state, {}, tiny_result())
    restored, _ = load_checkpoint(tmp_path / "din.pt", "cpu")
    assert np.allclose(expected, predict(restored, tiny_inference_batch()))
```

- [ ] **Step 2: Verify RED, implement checkpoint schema, and verify GREEN**

Run: `python -m pytest tests/v3/test_submission.py::test_checkpoint_roundtrip_preserves_predictions -v`

- [ ] **Step 3: Add failing CSV and forbidden-column tests**

```python
def test_submission_has_exact_schema_and_finite_scores(tmp_path):
    output = build_fixture_submission(tmp_path)
    rows = list(csv.DictReader(output.open()))
    assert list(rows[0]) == ["row_id", "user_id", "video_id", "score"]
    assert [int(r["row_id"]) for r in rows] == list(range(len(rows)))
    assert all(math.isfinite(float(r["score"])) for r in rows)
```

Instrument the test fixture CSV reader to fail if the test path requests `long_view`, `is_click`, `is_like`, `play_time_ms`, or another post-exposure field.

- [ ] **Step 4: Implement batched single-model and accepted-ensemble inference**

Recreate inference arrays from saved `EncoderState`, load selected checkpoints, use user-percentile blending only when the saved ensemble is accepted, validate all rows, and write the exact header.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/v3/test_submission.py -v`

Expected: PASS.

Commit: `git add research_agent_v3/checkpoints.py research_agent_v3/make_submission.py tests/v3/test_submission.py && git commit -m "feat: export V3 label-free submissions"`

---

### Task 7: Colab notebook, documentation, and complete verification

**Files:**
- Create: `colab/RecResearcher_V3_GPU.ipynb`
- Create: `research_agent_v3/README.md`
- Create: `requirements-v3.txt`
- Create: `tests/v3/test_notebook.py`
- Modify: `.gitignore`

**Interfaces:**
- Notebook consumes the public repository URL and KuaiRand-Pure Zenodo URL.
- Full run command: `python -m research_agent_v3.agent --data-dir ./KuaiRand-Pure/data --output-dir ./v3_runs/colab_run --execute --device cuda --amp --batch-size 16384 --candidate-epochs 12 --max-hours 2`.

- [ ] **Step 1: Write a failing notebook structure test**

Parse the notebook JSON and assert it contains executable cells for CUDA verification, repository setup, dataset download, MD5 verification, dependency installation, pytest, smoke run, full CUDA run, and submission generation. Assert no API key is embedded.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/v3/test_notebook.py -v`

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Create the Colab notebook and reproducibility README**

Use `wget` for `https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`; verify MD5 `0820331067a3784d9691136f772b35a7`; extract under the repository; print `torch.cuda.get_device_name(0)`; run tests before experiments; make Drive persistence optional and explicit.

- [ ] **Step 4: Run the complete local suite**

Run: `python -m pytest tests/v3 -v`

Expected: all CPU tests PASS; CUDA-only tests SKIP if the local host lacks CUDA.

- [ ] **Step 5: Run static and synthetic smoke checks**

Run: `python -m compileall -q research_agent_v3`

Run: `python -m research_agent_v3.agent --smoke --device cpu --output-dir ./work/v3_smoke`

Expected: compilation succeeds; smoke run creates completed DeepFM, DCN, DIN, and multi-task events with finite validation metrics.

- [ ] **Step 6: Prove V2 remained unchanged**

Compare SHA-256 hashes of the extracted V2 source snapshot against the final `research_agent_v2` tree and require no changed `.py`, `.md`, `.html`, or `.csv` files.

- [ ] **Step 7: Commit**

Commit: `git add colab research_agent_v3/README.md requirements-v3.txt tests/v3/test_notebook.py .gitignore && git commit -m "docs: add reproducible Colab GPU workflow"`

- [ ] **Step 8: Package the user-facing deliverable**

Create `outputs/rec-agent-v3-colab.zip` from the verified repository while excluding `.git`, caches, downloaded data, checkpoints, and temporary runs. Do not publish remotely yet.

---

## Remote publication gate

After the zip and verification report are ready, show the exact GitHub repository, target branch, commit list, and files to be uploaded. Request action-time confirmation before pushing because publishing changes remote state.
