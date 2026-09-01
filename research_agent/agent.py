"""A validation-only autonomous ML research loop for KuaiRand-Pure.

The agent never imports ``data.load`` and deliberately has no test option.
Every candidate is trainable with numpy on a Windows CPU.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluate import evaluate  # official evaluator; intentionally unmodified
from research_agent.data import encode, load_train_valid
from research_agent.models import FM, make_bpr_pairs
from research_agent.qwen_planner import ask_qwen_with_trace
from research_agent.diagnosis import diagnose, profile
from research_agent.experiment_store import tree_record
from research_agent.research_memory import retrieve
from research_agent.strategies import CATALOG, Strategy

# Evidence accumulated in this project: pointwise FM is ~0.60 primary whereas
# the sampled BPR objective collapsed to ~0.37.  Keep this cross-run knowledge
# as a guardrail; a new output directory must not erase an observed failure.
KNOWN_BLOCKED_STRATEGIES = {"pairwise_bpr", "history_bpr"}

def source_fingerprint():
    """Small auditable source snapshot used as the per-run code diff anchor."""
    chunks = []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        chunks.append(f"## {path.name}\n{path.read_text(encoding='utf-8')}")
    return hashlib.sha256("\n".join(chunks).encode()).hexdigest()[:16]


def json_default(value):
    """Convert NumPy values in metrics/checkpoint metadata to plain JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_jsonl(path, event):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=json_default) + "\n")


class ResearchAgent:
    def __init__(self, data_dir, output_dir, epochs=40, candidate_epochs=18, seed=0,
                 max_iterations=50, max_hours=6.0, epsilon=.002, patience=3,
                 manual_interventions="none", planner="fixed", qwen_model="qwen-plus", resume=False):
        self.data_dir, self.output_dir = data_dir, Path(output_dir)
        self.epochs, self.candidate_epochs, self.seed = epochs, candidate_epochs, seed
        self.max_iterations, self.deadline = max_iterations, time.time() + max_hours * 3600
        self.epsilon, self.patience = epsilon, patience
        self.manual_interventions = manual_interventions
        self.planner, self.qwen_model = planner, qwen_model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "runs.jsonl"
        self.best_path = self.output_dir / "validation_best.npz"
        self.splits = load_train_valid(data_dir)
        self.data_profile = profile(self.splits)
        self.cache = {}
        self.best_primary, self.best_event, self.stalls = -np.inf, None, 0
        self.history = []
        self.start_iteration = 0
        if resume:
            self._restore_run()

    def _restore_run(self):
        """Resume an experiment tree without retraining its immutable FM root."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Cannot resume: {self.log_path} does not exist.")
        self.history = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        completed = [e for e in self.history if e.get("status") == "completed" and e.get("metrics")]
        if not completed:
            raise RuntimeError("Cannot resume: no completed validation experiment is recorded.")
        self.best_event = max(completed, key=lambda e: float(e["metrics"]["primary"]))
        self.best_primary = float(self.best_event["metrics"]["primary"])
        self.start_iteration = len(self.history)
        self.stalls = 0
        for event in reversed(self.history):
            if "no_material_gain" in event.get("decision", ""):
                self.stalls += 1
            else:
                break

    def dataset(self, history):
        if history not in self.cache:
            self.cache[history] = encode(self.splits, use_history=history)
        return self.cache[history]

    def train_and_evaluate(self, strategy, iteration):
        encoded, dim, fields = self.dataset(strategy.use_history)
        train, valid = encoded["train"], encoded["valid"]
        heads = 7 if strategy.use_aux else 1
        model = FM(dim, k=16, lr=.001, seed=self.seed + iteration, heads=heads)
        rng = np.random.default_rng(self.seed + iteration)
        best_ep, best_score, best_state, bad = 0, -np.inf, None, 0
        max_epochs = self.epochs if strategy.name == "official_fm" else self.candidate_epochs
        pair_pos = pair_neg = None
        if strategy.kind == "bpr":
            pair_pos, pair_neg = make_bpr_pairs(train["users"], train["y"], rng)
            if not len(pair_pos):
                raise RuntimeError("BPR needs at least one user with both positive and negative train impressions.")
        for epoch in range(1, max_epochs + 1):
            if time.time() >= self.deadline:
                raise TimeoutError("Global time budget reached during training.")
            if strategy.kind == "bpr":
                order = rng.permutation(len(pair_pos)); losses = []
                for start in range(0, len(order), 8192):
                    idx = order[start:start+8192]
                    losses.append(model.step_bpr(train["X"][pair_pos[idx]], train["X"][pair_neg[idx]]))
            else:
                order = rng.permutation(len(train["y"])); losses = []
                for start in range(0, len(order), 8192):
                    idx = order[start:start+8192]
                    auxiliary = None
                    if strategy.use_aux:
                        auxiliary = [train["aux"][key][idx] for key in
                                     ("click", "like", "follow", "comment", "forward", "watch_ratio")]
                    losses.append(model.step_pointwise(train["X"][idx], train["y"][idx], auxiliary))
            metrics = evaluate(valid["users"], valid["y"], model.predict(valid["X"]))
            if metrics["primary"] > best_score + 1e-5:
                best_ep, best_score, bad = epoch, metrics["primary"], 0
                best_state = {k: v.copy() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 4:
                    break
        model.load_state_dict(best_state)
        metrics = evaluate(valid["users"], valid["y"], model.predict(valid["X"]))
        return model, metrics, {"best_epoch": best_ep, "last_loss": float(np.mean(losses)),
                                "dim": dim, "fields": fields, "n_train": len(train["y"]),
                                "n_valid": len(valid["y"])}

    def save_best(self, model, event):
        np.savez_compressed(self.best_path, **model.state_dict(),
                            metadata=json.dumps(event, ensure_ascii=False, default=json_default))

    def next_strategy(self, iteration):
        # Baseline is an immutable control. Subsequent candidates exclude prior
        # project-level failures even when this is a brand-new output directory.
        if iteration == 0:
            return CATALOG[0], {"mode": "fixed", "guardrails": sorted(KNOWN_BLOCKED_STRATEGIES)}
        diagnosis = diagnose(self.history, self.epsilon)
        blocked = set(diagnosis["blocked"]) | KNOWN_BLOCKED_STRATEGIES
        allowed = [s for s in CATALOG[1:] if s.name not in blocked]
        tried = {event.get("strategy", {}).get("name", "").replace("_seedcheck", "")
                 for event in self.history if event.get("status") == "completed"}
        untried = [strategy for strategy in allowed if strategy.name not in tried]
        base = allowed[(iteration - 1) % len(allowed)]
        fixed = Strategy(base.name + "_seedcheck", base.hypothesis + " Repeat with a new seed to check robustness.",
                         base.kind, base.use_history, base.use_aux,
                         base.code_diff + " | Seed-robustness rerun; no feature-capacity expansion.")
        if self.planner != "qwen":
            return fixed, {"mode": "fixed", "guardrails": sorted(blocked),
                           "recovery": "BPR family blocked by prior project evidence"}
        try:
            candidates = retrieve(diagnosis["bottleneck"], blocked)
            context = {"task": "KuaiRand-Pure user-local long_view ranking; validation only",
                       "data_profile": self.data_profile, "bottleneck_diagnosis": diagnosis,
                       "research_memory": candidates,
                       "convergence": {"epsilon": self.epsilon, "stalls": self.stalls, "patience": self.patience},
                       "previous_runs": [{"strategy": e["strategy"]["name"], "metrics": e.get("metrics"),
                                          "decision": e.get("decision")} for e in self.history],
                       "blocked_strategies": sorted(blocked),
                       "untried_strategies": [s.name for s in untried],
                       "allowed_strategies": [s.name for s in allowed]}
            plan, usage, trace = ask_qwen_with_trace(context, model=self.qwen_model)
            if plan["strategy"] == "stop":
                raise RuntimeError("Qwen requested stop; controller will use a safe fixed candidate until convergence.")
            if plan["strategy"] in blocked:
                chosen = next(s for s in allowed if s.name == "multitask_engagement")
                return chosen, {"mode": "qwen_guardrail_override", "plan": plan, "token_usage": usage,
                                "qwen_trace": trace,
                                "research_state": tree_record(iteration, self.history, diagnosis, candidates),
                                "recovery": "blocked a BPR-family retry after a severe BPR regression"}
            if untried and plan["strategy"] not in {strategy.name for strategy in untried}:
                chosen = untried[0]
                return chosen, {"mode": "qwen_coverage_override", "plan": plan, "token_usage": usage,
                                "qwen_trace": trace,
                                "research_state": tree_record(iteration, self.history, diagnosis, candidates),
                                "recovery": "tested every safe untried strategy before repeating a prior experiment"}
            chosen = next(s for s in allowed if s.name == plan["strategy"])
            return chosen, {"mode": "qwen", "plan": plan, "token_usage": usage, "qwen_trace": trace,
                            "research_state": tree_record(iteration, self.history, diagnosis, candidates)}
        except Exception as exc:
            fallback = next(s for s in allowed if s.name == "multitask_engagement")
            recovery = "Qwen unavailable; BPR family remains blocked and multitask_engagement was used"
            return fallback, {"mode": "qwen_fallback", "error": repr(exc), "recovery": recovery}

    def run(self):
        config = {"data_dir": os.path.abspath(self.data_dir), "validation_only": True,
                  "test_access": "disabled by design", "max_iterations": self.max_iterations,
                  "max_hours": round((self.deadline-time.time())/3600, 3), "epsilon": self.epsilon,
                  "convergence_patience": self.patience, "manual_interventions": self.manual_interventions,
                  "source_fingerprint": source_fingerprint(), "data_profile": self.data_profile,
                  "agent_architecture": "diagnosis -> research memory -> planner -> runner -> evaluator -> reflection"}
        (self.output_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
        for iteration in range(self.start_iteration, self.max_iterations):
            if time.time() >= self.deadline or self.stalls >= self.patience:
                break
            strategy, planner_record = self.next_strategy(iteration)
            started = time.time()
            event = {"iteration": iteration + 1, "strategy": asdict(strategy), "started_at": started,
                     "manual_interventions": self.manual_interventions, "error_recovery": None,
                     "source_fingerprint": source_fingerprint(), "planner": planner_record}
            try:
                model, metrics, training = self.train_and_evaluate(strategy, iteration)
                improvement = float(metrics["primary"] - self.best_primary) if np.isfinite(self.best_primary) else None
                event.update({"status": "completed", "metrics": metrics, "training": training,
                              "improvement_over_validation_best": improvement, "elapsed_s": round(time.time()-started, 2)})
                # Preserve the literal validation-best model, even for a small
                # improvement. Epsilon controls convergence only; it must not
                # cause the best available checkpoint to be discarded.
                is_new_best = improvement is None or improvement > 1e-5
                if is_new_best:
                    self.best_primary, self.best_event = metrics["primary"], event
                    self.save_best(model, event)
                if improvement is None or improvement > self.epsilon:
                    self.stalls = 0
                    event["decision"] = "new_validation_best_checkpoint_saved" if is_new_best else "material_gain"
                else:
                    self.stalls += 1
                    event["decision"] = (("new_validation_best_checkpoint_saved; " if is_new_best else "") +
                                         f"no_material_gain; convergence_stall={self.stalls}/{self.patience}")
            except Exception as exc:
                # A failed candidate does not corrupt the agent: log it and continue with the next safe idea.
                self.stalls += 1
                event.update({"status": "recovered_error", "error_recovery": {"error": repr(exc),
                              "action": "candidate skipped; preserved previous validation-best checkpoint"},
                              "elapsed_s": round(time.time()-started, 2)})
            write_jsonl(self.log_path, event)
            self.history.append(event)
            print(f"iter {event['iteration']:02d} {strategy.name}: {event['status']} | "
                  f"{event.get('metrics', {}).get('primary', 'n/a')} | stalls={self.stalls}")
        summary = {"status": "converged" if self.stalls >= self.patience else "budget_complete",
                   "best_validation_primary": self.best_primary, "best_run": self.best_event,
                   "iterations_completed": sum(1 for _ in open(self.log_path, encoding="utf-8")),
                   "stalls": self.stalls, "checkpoint": str(self.best_path) if self.best_event else None,
                   "development_data": "train + validation only; no test split is loaded"}
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Validation-only autonomous recommender research MVP")
    parser.add_argument("--data-dir", default="../KuaiRand-Pure/data")
    parser.add_argument("--output-dir", default="research_runs")
    parser.add_argument("--epochs", type=int, default=40, help="Official baseline max epochs")
    parser.add_argument("--candidate-epochs", type=int, default=18)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manual-interventions", default="none", help="Free-text audit note; does not change data use")
    parser.add_argument("--planner", choices=("fixed", "qwen"), default="fixed")
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse runs.jsonl and validation-best checkpoint; do not rerun the FM root experiment")
    args = parser.parse_args()
    result = ResearchAgent(**vars(args)).run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
