# Autonomous ML Research Agent (CPU MVP)

## Start here: a five-stage demo

If you want to understand the agent before launching the full search, run:

```powershell
python research_agent/demo_agent.py --data-dir ./KuaiRand-Pure/data --output-dir ./demo_runs
```

It makes one clean, reproducible FM-control experiment and writes an easy-to-read
`demo_log.jsonl` with the five stages in the competition diagram: read problem,
inspect data, engineer features, train/evaluate, and reflect/revise. It saves
the validation-best FM checkpoint and proposes (but does not yet run) the next
multi-task experiment. No test split or test label is loaded.

## Qwen-driven planning demo

This is the fastest honest LLM integration: after the FM demo generates a
validation-only log, Qwen reads that log and returns a constrained JSON plan for
the next experiment. It cannot request test evaluation or arbitrary code.

```powershell
$env:DASHSCOPE_API_KEY = "your DashScope API key"
python research_agent/qwen_demo.py --log ./demo_runs/demo_log.jsonl
```

The plan and Qwen-reported token usage are saved to
`demo_runs/qwen_next_plan.json`. Use `QWEN_API_KEY` instead if you prefer that
environment-variable name. The endpoint/model can later be overridden in code
for a different DashScope region or Qwen model.

## Inspect an autonomous run

Every Qwen-planned iteration records a sanitized request/response trace (never
the API key) in `runs.jsonl`. Turn it into a presentation-ready timeline with:

```powershell
python research_agent/trace_report.py --run-dir ./rec_researcher_run
```

This produces `agent_trace.md`: data profile, diagnosis, candidate rationale,
Qwen plan, token use, validation metrics, failures and recovery decisions.

This is a deliberately small, auditable research loop for the TechJam track.
It is **validation-only**: `research_agent/data.py` never creates a test split,
and selects only `20220422–20220428` rows before parsing them into examples;
hidden-test rows are discarded as raw CSV values and their labels are never used.
`evaluate.py` is imported unchanged.

## Run

From the starter-kit directory on Windows PowerShell:

```powershell
python research_agent/agent.py --data-dir ./KuaiRand-Pure/data --output-dir ./research_runs
```

For a short smoke test first:

```powershell
python research_agent/agent.py --data-dir ./KuaiRand-Pure/data --output-dir ./research_runs_smoke --epochs 2 --candidate-epochs 1 --max-iterations 2 --max-hours 0.25
```

The first full iteration is the five-field pointwise FM control. It should
reproduce the **validation** FM baseline near the published result (small seed/
hardware variation is expected). The agent then tries BPR, causal sequence +
BPR, multi-task engagement/watch-ratio, and their composition. It intentionally
does not try static-field expansion or a larger FM embedding.

## Audit outputs

`runs.jsonl` has one record per iteration: hypothesis, built-in code diff,
metrics, timing, error recovery, and manual-intervention note. `validation_best.npz`
is overwritten only for a material validation improvement (> 0.002); `summary.json`
is the handoff record. The loop stops after three consecutive non-material gains,
or the supplied 50-iteration / 6-hour budget.

To extend the autonomous search, add a `Strategy` in `strategies.py`; do not add
test evaluation to the research loop. The submission workflow remains separate.
## 先做数据研究（推荐）

不要只靠“猜一个策略”。先让 Agent 只读取 train + validation，生成可审计的数据画像，
再请千问从画像中提出下一条可验证的假设：

```powershell
$env:DASHSCOPE_API_KEY = "你的千问 API Key"
python research_agent/qwen_data_research.py `
  --data-dir ./KuaiRand-Pure/data `
  --output-dir ./rec_researcher_run
```

输出包括：

- `data_eda.json`：确定性统计（不调用 LLM），含冷启动、用户长尾、tab、时长和辅助反馈信号；
- `data_research.json`：千问的发现、证据、推荐策略与可证伪的下一条假设；

两者都不加载 test split。之后运行主 Agent 时，它本身也会将同一份 EDA 传给千问 planner：

```powershell
python research_agent/agent.py `
  --data-dir ./KuaiRand-Pure/data `
  --output-dir ./rec_researcher_run `
  --planner qwen --max-iterations 6 --max-hours 2
```
