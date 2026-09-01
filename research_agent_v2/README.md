# RecResearcher V2：推荐系统自动研究智能体

面向 TikTok TechJam「Autonomous ML Research Agent for Recommender Systems」赛道的 KuaiRand-Pure 实现。V2 将数据分析、研究规划、模型搜索、双模型融合、官方评估和无标签提交导出串成可审计闭环。

**项目汇报页：**[打开中文版 HTML 架构与结果报告](./agent_report.html)

## 目标与约束

- 任务：在每个用户的曝光集合内，对 `long_view` 进行排序。
- 唯一的模型选择指标：官方 `evaluate.py` 计算的 `GAUC`、`nDCG@5` 和 `primary`。
- 开发阶段只用 `train + validation`；不修改官方评估器。
- test 只在最终提交导出时读取曝光时可见字段，不读取 `long_view`、click、like、comment、play_time 等 test 后验标签。
- 每个候选保存假设、代码变化描述、训练信息、validation 指标、checkpoint 和错误恢复信息。

## 当前结果：Run 008

| 方案 | GAUC | nDCG@5 | validation primary |
|---|---:|---:|---:|
| 本轮官方 5-field FM control | 0.667395 | 0.536127 | 0.601761 |
| DeepFM | 0.671300 | 0.538160 | 0.604730 |
| DCN | 0.671119 | 0.537426 | 0.604272 |
| **DeepFM 0.6 + DCN 0.4** | **0.671813** | **0.538292** | **0.605052** |

最佳融合相对本轮 FM control 提升 **+0.003291**，超过题目设定的实质提升阈值 `ε = 0.002`。这只是 validation 结果，不应当被表述为 hidden test 成绩。

## Agent 架构

```text
KuaiRand-Pure 任务与约束
            │
            ▼
Data Agent（确定性 EDA + 字段安全边界）
            │  数据事实
            ▼
Qwen Data / Research Planner（中文结论、假设、白名单操作）
            │  受控 recipe
            ▼
Model Search Agent（FM / DeepFM / DCN / 多任务候选）
            │
            ▼
Evaluation Agent（未修改的 evaluate.py）
            │  checkpoint、指标、反思日志
            ▼
Ensemble Agent（Qwen 推荐两个互补模型 + Python 搜索权重）
            │
            ▼
冻结 validation-best → 无标签 test 推理 → submission.csv
```

Qwen 负责解释数据、提出可审计的研究建议、推荐融合模型对；它不能执行任意代码，也不能覆盖 validation 指标。实际候选由 `operations.py` 白名单和 `recipe_compiler.py` 限制，最终裁决仍由官方指标完成。

## 安装

在 Starter Kit 根目录执行。官方基线仅需要 NumPy；本 V2 的 DeepFM/DCN 还需要 PyTorch（CPU 版本即可）。

```powershell
pip install numpy torch
```

如果希望连接千问，在 PowerShell 中设置 API Key；未设置时 Data Agent 和 Planner 会自动使用确定性中文 fallback，不会中断实验。

```powershell
$env:DASHSCOPE_API_KEY = "你的千问 API Key"
# 或 $env:QWEN_API_KEY = "你的千问 API Key"
```

## 一次完整自主运行

```powershell
cd C:\Users\86137\Downloads\kuairand-starter-kit\kuairand-starter-kit
$env:PYTHONIOENCODING = "utf-8"

python -u research_agent_v2/agent.py `
  --data-dir ./KuaiRand-Pure/data `
  --output-dir ./v2_runs/run_009 `
  --execute `
  --candidate-epochs 12 `
  --max-hours 2
```

运行输出包括：中文数据结论、Qwen 研究操作、每轮候选的 validation primary、融合结论和反思。完整审计文件位于指定运行目录：

| 文件 | 内容 |
|---|---|
| `data_eda.json` | Python 计算的原始数据统计 |
| `research_plan_zh.json` | Data Agent / Planner 中文结论、Qwen trace、已批准操作 |
| `runs.json` | 每轮 hypothesis、code diff、metrics、错误恢复 |
| `models/` | 所有完成候选的 checkpoint 与 validation 分数 |
| `ensemble_result.json` | 两模型选择、融合权重、official validation metrics |
| `reflection_zh.json` | 本轮反思与下一步方向 |

## 导出冻结的提交文件

Run 008 已生成：`submission_run_008.csv`。若要为一个已完成且有 `ensemble_result.json` 的运行重新生成 CSV：

```powershell
python research_agent_v2/make_submission.py `
  --data-dir ./KuaiRand-Pure/data `
  --run-dir ./v2_runs/run_008 `
  --output ./submission_run_008.csv
```

导出器只支持当前冻结的 5-field DeepFM/DCN 融合，并独立保证表头、连续 `row_id`、行数和有限 score。它不会计算 test 指标，也不会读取 test 标签。

## 目录说明

| 文件 | 作用 |
|---|---|
| `agent.py` | 总编排：EDA → Qwen → recipe → 训练 → 评估 → 融合 → 审计 |
| `data.py` | 仅 train/valid 开发加载、因果历史编码、无标签 test 特征加载 |
| `data_agent.py` | 将 Python EDA 交给 Qwen，产出中文数据结论 |
| `planner.py` | Qwen Research Planner；输出受约束的研究计划 |
| `operations.py` | 可执行操作白名单，阻止任意模型/特征代码注入 |
| `recipe_compiler.py` | 将操作计划编译为有限候选队列 |
| `runner.py` | FM 与 PyTorch 单模型训练、early stopping、官方验证 |
| `torch_models.py` | CPU 友好的 DeepFM 与 DCN |
| `ensemble_agent.py` | 双模型选择、用户内 percentile 融合与权重搜索 |
| `make_submission.py` | 无 test 标签的冻结模型推理与 CSV 导出 |
| `agent_report.html` | 可展示的中文项目报告 |

## 后续优化路线

当前优先保证 CPU 环境下的完整、可靠、可提交 MVP。因 GPU / 长序列训练资源暂缺，下列模型暂未接入候选池，并不表示它们无效：DIN/DIEN、SASRec/BERT4Rec、DCNv2、xDeepFM、MMoE/ESMM。

在 CPU 上仍可继续的研究包括：DeepFM/DCN 多 seed 稳定性、轻量历史特征、时间上下文、pairwise/listwise 损失、以及随机曝光日志的无偏审计。GPU 可用后，应将序列模型和更深的多任务模型接入同一 recipe 与融合框架，再由 validation 指标自动选择。
