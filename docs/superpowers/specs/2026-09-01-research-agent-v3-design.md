# RecResearcher V3 Colab GPU Design

## Goal

Build `research_agent_v3` alongside the unchanged `research_agent_v2`. V3 must run end to end in Google Colab, preserve the official KuaiRand-Pure evaluation and leakage boundaries, expand the autonomous search space with GPU-aware neural models, and produce auditable checkpoints and submissions.

## Scope

V3 includes:

- CUDA/CPU device selection, batched device transfer, optional automatic mixed precision, configurable batch size, and runtime/device reporting.
- Deterministic feature caching keyed by source-file metadata and feature recipe.
- Existing FM, DeepFM, and DCN controls with configurable neural capacity.
- A causal Deep Interest Network (DIN) candidate using bounded positive-video histories.
- Multi-behavior auxiliary heads for selected train-time action labels while retaining `long_view` as the only prediction target.
- V3 recipe operations, experiment logs, checkpoint metadata, ensemble compatibility, and label-free test inference.
- A Colab notebook that downloads the repository and KuaiRand-Pure, verifies the environment, runs tests and smoke experiments, and launches a full run.

V3 excludes SASRec, BERT4Rec, DIEN, distributed training, multi-GPU training, KuaiRand-1k/27k optimization, and automated GitHub publication. Those are intentionally deferred to keep the one-day implementation reliable.

## Non-Negotiable Data Rules

- Training uses only the KuaiRand-Pure training window `20220408` through `20220421`.
- Validation uses `20220422` through `20220428` and may influence model selection.
- Test uses `20220429` through `20220508`; test labels and post-exposure actions must never be parsed or used.
- `log_random_4_22_to_5_08_pure.csv` is not training data.
- KuaiRand-1k and KuaiRand-27k are not auxiliary training data for Pure.
- Official `evaluate.py` remains unchanged. Candidate selection uses official GAUC, nDCG@5, and primary.
- Validation history is initialized from the final state of training history and is then frozen; validation outcomes never update history.

## Architecture

### Package strategy

`research_agent_v2` remains untouched. `research_agent_v3` reuses stable V2 planner, reporting, and ensemble concepts where appropriate but owns its data representation, models, runner, recipes, checkpoints, and submission inference. V3 modules import from V3 rather than silently using V2 implementation details.

### Data pipeline

The loader preserves V2's date filtering and label-free test loader. A V3 encoder emits:

- `X`: categorical feature IDs for the current impression.
- `history_video_ids`: a padded, bounded sequence of prior positive videos.
- `history_mask`: valid-history positions.
- `y`: `long_view` for train/validation only.
- `aux`: selected training-only behavior labels.
- `users`: user IDs needed by the official evaluator.
- vocabulary and field metadata needed for reproducible inference.

The history buffer is updated only after encoding a positive training row. It is never updated from validation or test outcomes. Unknown and padding IDs are distinct.

Encoded arrays may be cached as compressed NumPy artifacts. The cache fingerprint includes source path, size, modification time, date boundaries, enabled features, maximum history length, and a schema version. A mismatched fingerprint forces recomputation.

### Models

All neural models return a dictionary with at least `long_view_logits`. Multi-task variants additionally return one logit tensor per enabled auxiliary task.

- `DeepFMV3`: configurable embedding dimension and MLP widths.
- `DCNV3`: configurable embedding dimension, cross depth, and MLP widths.
- `DINV3`: embeds the candidate video and historical videos, computes candidate-conditioned attention over valid history positions, combines the attended interest with current categorical embeddings, and predicts `long_view`.
- Optional multi-task heads share the learned representation. The training objective is primary BCE plus `aux_weight` times the mean auxiliary BCE. Auxiliary labels never affect evaluation or final score columns.

### Training runtime

The runner resolves `auto`, `cuda`, or `cpu`. CPU tensors remain in host memory; each batch is transferred to the selected device. Validation and inference are also batched. CUDA AMP is enabled only when requested and CUDA is active.

Early stopping tracks official validation primary. The validation-best state is restored. Each result records model family, full model configuration, feature configuration, vocabulary metadata, best epoch, official metrics, elapsed seconds, device name, CUDA availability, AMP state, batch size, and recovery events.

Out-of-memory failures cause one controlled retry with half the batch size, down to a documented minimum. Other exceptions are logged and returned to the orchestration layer without corrupting prior best checkpoints.

### Autonomous recipe integration

The operation whitelist gains bounded operations for:

- increasing embedding/MLP capacity;
- enabling DIN history attention;
- enabling selected multi-behavior auxiliary heads;
- searching auxiliary loss weights;
- enabling AMP and GPU-oriented batch sizes.

The recipe compiler always retains official FM, V2-scale DeepFM, and V2-scale DCN controls. It adds a small reproducible queue of V3 candidates rather than allowing arbitrary generated code. Validation results remain the only acceptance criterion.

### Ensemble and submission

V3 checkpoints include sufficient metadata to reconstruct the encoder and model. Ensemble search continues to operate on validation score vectors and uses the official evaluator. Label-free test inference reconstructs the frozen train vocabulary/history state, loads the selected checkpoint or ensemble members, predicts in batches, and writes exactly:

`row_id,user_id,video_id,score`

The exporter checks row count, strictly increasing zero-based `row_id`, alignment fields, numeric finite scores, and the absence of parsed test labels.

### Colab workflow

The notebook performs these stages:

1. Confirm a CUDA runtime and display the GPU; CPU fallback is allowed for tests.
2. Clone or upload the project.
3. Download `https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz` and verify MD5 `0820331067a3784d9691136f772b35a7`.
4. Extract data and verify required files.
5. Install dependencies.
6. Run the V3 unit suite.
7. Run a tiny smoke experiment for DeepFM, DCN, DIN, and multi-task loss.
8. Optionally reproduce the V2 result.
9. Launch the full V3 search and persist runs/checkpoints, optionally under Google Drive.
10. Generate a validation comparison table and final submission file.

## Testing Strategy

Development follows test-driven development.

- Data tests prove date filtering, frozen validation history, causal sequence construction, padding/masking, cache invalidation, and label-free test loading.
- Model tests prove output shapes, padding invariance, finite logits/losses, auxiliary-head behavior, and CPU/CUDA device consistency where CUDA is available.
- Runner tests prove batch movement, official evaluation integration, best-state restoration, metadata logging, and controlled OOM retry.
- Recipe tests prove only whitelisted bounded candidates are emitted.
- Submission tests prove schema, alignment, finite scores, and that test-label columns are never requested.
- A synthetic end-to-end smoke test runs every supported neural family without the KuaiRand download.

Real KuaiRand validation experiments are separate from deterministic automated tests. Reported performance claims must come from saved official validation outputs, never fabricated or extrapolated numbers.

## Success Criteria

- `research_agent_v2` is unchanged.
- All V3 automated tests pass on CPU; CUDA-specific tests pass or explicitly skip when CUDA is absent.
- A Colab GPU smoke run completes for DeepFM, DCN, DIN, and a multi-task candidate.
- A full V3 run emits per-iteration hypothesis, recipe/code-change description, official validation metrics, checkpoint, device/resource metadata, and error/recovery events.
- The validation comparison includes the V2 controls and V3 candidates and selects the actual validation-best model/ensemble.
- The final exporter creates a Starter-Kit-compatible CSV without reading test labels.
- README and notebook provide reproducible commands and clearly distinguish validation results from hidden-test results.

## Delivery and Publication

The initial deliverable is a local project copy and zip containing V3, tests, documentation, and the Colab notebook. Publishing to GitHub is a separate final action: immediately before pushing, the target repository/branch and files will be shown for confirmation because it changes a remote repository.
