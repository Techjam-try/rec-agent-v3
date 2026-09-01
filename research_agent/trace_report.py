"""Turn machine-readable run logs into a demo-friendly decision timeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a readable RecResearcher audit trace")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = ["# RecResearcher decision trace", "", "## Guardrails", "",
             "- Development data: train + validation only; hidden test disabled.",
             f"- Convergence: epsilon={config['epsilon']}, patience={config['convergence_patience']}.",
             f"- Data profile: `{json.dumps(config.get('data_profile', {}), ensure_ascii=False)}`", ""]
    for event in events:
        lines += [f"## Experiment {event['iteration']}: {event['strategy']['name']}", "",
                  f"- Hypothesis: {event['strategy']['hypothesis']}",
                  f"- Code change: {event['strategy']['code_diff']}"]
        planner = event.get("planner", {})
        state = planner.get("research_state", {})
        if state:
            diagnosis = state.get("diagnosis", {})
            lines += [f"- Bottleneck: **{diagnosis.get('bottleneck')}** — {diagnosis.get('evidence')}",
                      f"- Parent experiment: {state.get('parent_id')}"]
        if planner.get("mode") == "qwen":
            plan = planner.get("plan", {})
            lines += [f"- Qwen decision: **{plan.get('strategy')}** — {plan.get('why')}",
                      f"- Qwen evidence: {plan.get('evidence')}",
                      f"- Token usage: `{json.dumps(planner.get('token_usage', {}))}`"]
        elif planner.get("mode", "").endswith("fallback"):
            lines += [f"- Recovery: {planner.get('recovery')}", f"- Error: `{planner.get('error')}`"]
        if event.get("metrics"):
            metrics = event["metrics"]
            lines.append(f"- Validation: GAUC={metrics['GAUC']:.4f}, nDCG@5={metrics['nDCG@5']:.4f}, primary={metrics['primary']:.4f}")
        lines += [f"- Decision: {event.get('decision', event.get('error_recovery'))}", ""]
    output = Path(args.output) if args.output else run_dir / "agent_trace.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
