# Skills Evaluation Cookbook

使用 OpenJudge 对 AI Agent Skill 包进行多维度自动化评估。

---

## 目录

- [简介](#简介)
- [评估维度](#评估维度)
- [运行教程](#运行教程)
- [报告示例](#报告示例)

---

## 简介

Agent Skills（技能包）是扩展 AI Agent 能力的本地目录包，每个技能包通常包含：

```
skill-name/
├── SKILL.md          # 必需：YAML frontmatter + markdown 指令
├── scripts/          # 可选：Agent 可执行的 Python / Bash 脚本
├── references/       # 可选：指令中引用的附加文档
└── assets/           # 可选：模板与资源文件
```

`cookbooks/skills_evaluation` 提供了一套端到端的技能评估流水线，通过 **5 个独立的 LLM-as-Judge Grader** 并发对技能包打分，输出加权综合分数，并生成 JSON 与 Markdown 格式的评估报告。

---

## 评估维度

评估框架包含以下 5 个维度（Grader），每个维度独立运行，最终加权平均为 0–100 的综合分数。

### 1. Threat Analysis（威胁分析）

| 属性 | 说明 |
|------|------|
| **类** | `SkillThreatAnalysisGrader` |
| **量表** | 1–4（4 = 安全，1 = 严重风险） |
| **默认通过阈值** | ≥ 3（Low 或更好） |

基于 **AITech 分类体系**，对技能包全组件进行 LLM 语义安全扫描，涵盖：

- `AITech-1.1` — 直接提示注入（越狱、指令覆盖）
- `AITech-1.2` — 间接提示注入（外部数据源嵌入恶意指令）
- `AITech-4.3` — 协议操控 / 能力膨胀（关键词诱导、品牌冒充）
- `AITech-8.2` — 数据外泄 / 暴露（硬编码凭据、未授权网络传输）
- `AITech-9.1` — Agent 系统操控（命令注入、代码注入）
- `AITech-9.2` — 检测规避（混淆、base64→exec 链）
- `AITech-12.1` — 工具利用（工具投毒、违反 allowed-tools 限制）
- `AITech-13.1` — 可用性破坏（无限循环、资源耗尽）
- `AITech-15.1` — 有害 / 误导性内容

每个发现包含：severity 等级、AITech 分类码、证据片段、文件位置和修复建议。

---

### 2. Declaration Alignment（声明对齐）

| 属性 | 说明 |
|------|------|
| **类** | `SkillDeclarationAlignmentGrader` |
| **量表** | 1–3（3 = 对齐，1 = 不匹配） |
| **默认通过阈值** | ≥ 2（Uncertain 或更好） |

检测 `SKILL.md` 声明的功能与脚本实际行为之间的**蓄意不一致**，聚焦于：

- 隐藏后门、隐蔽数据管道
- 未声明的网络操作（声称"本地处理"实则外传数据）
- 工具投毒（description 与脚本行为不符）

> **注意**：若技能包没有脚本文件，该维度自动跳过并标记为通过。

---

### 3. Completeness（完整性）

| 属性 | 说明 |
|------|------|
| **类** | `SkillCompletenessGrader` |
| **量表** | 1–3（3 = 完整，1 = 不完整） |
| **默认通过阈值** | ≥ 2（Partially complete 或更好） |

评估技能包是否提供足够的细节以完成任务，检查：

- 步骤、输入、输出是否明确
- 先决条件（环境、依赖、权限）是否说明
- 错误处理与边界情况是否覆盖
- 核心算法 / 公式是否正确
- `SKILL.md` 承诺的能力与实现是否一致（防止"空头支票"）

---

### 4. Relevance（相关性）

| 属性 | 说明 |
|------|------|
| **类** | `SkillRelevanceGrader` |
| **量表** | 1–3（3 = 完全匹配，1 = 不匹配） |
| **默认通过阈值** | ≥ 2（Partial match 或更好） |

评估技能包与**给定任务描述**的匹配程度：

- 技能的核心目的是否直接完成任务（而非仅测量/评估任务结果）
- 技能名称和描述是否明确定位到对应用例
- 是否提供具体可操作的技术模式，而非流程脚手架

> 若未提供 `task_description`，将使用技能自身的 `description` 字段做自洽性检验。

---

### 5. Structure / Design（结构设计）

| 属性 | 说明 |
|------|------|
| **类** | `SkillDesignGrader` |
| **量表** | 1–5（5 = 优秀，1 = 较差） |
| **默认通过阈值** | ≥ 3（Adequate 或更好） |

从 **7 个子维度**评估技能包的内部设计质量：

| 维度 | 考察点 |
|------|--------|
| **D1 Knowledge Delta** | 是否提供超越 Claude 基础知识的专家级内容 |
| **D2 Mindset + Procedures** | 是否传授专家思维框架和非显而易见的操作流程 |
| **D3 Specification Compliance** | `name` 格式是否合法；`description` 是否包含 WHAT / WHEN / KEYWORDS |
| **D4 Progressive Disclosure** | 内容分层是否合理（metadata → body → references） |
| **D5 Freedom Calibration** | 约束力度是否与任务脆弱性相匹配 |
| **D6 Practical Usability** | 代码示例是否可用；决策树是否完整；错误处理是否有 fallback |
| **D7 Anti-Pattern Quality** | 是否提供明确的 NEVER 列表（补充维度，加分项） |

---

## 运行教程

### 前提条件

安装依赖：

```bash
pip install -e .
```

在项目根目录的 `.env` 文件中配置模型：

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://your-api-endpoint/v1   # 可选，默认使用 OpenAI 官方端点
OPENAI_MODEL=qwen3.6-plus                       # 可选，默认 qwen3.6-plus
```

### 命令行运行

```bash
# 评估单个技能包目录
python cookbooks/skills_evaluation/evaluate_skills.py /path/to/my-skill

# 评估技能注册表（目录下每个子目录都是一个技能包）
python cookbooks/skills_evaluation/evaluate_skills.py /path/to/skills/

# 附带任务描述（用于 Relevance 和 Completeness 维度）
python cookbooks/skills_evaluation/evaluate_skills.py /path/to/skills/ "自动化代码审查 Pull Request"
```

评估完成后，结果将保存到：

```
cookbooks/skills_evaluation/results/
├── grading_results.json   # 结构化 JSON 报告
└── grading_report.md      # Markdown 可读报告
```

### 在代码中调用

```python
import asyncio
from openjudge.models.openai_chat_model import OpenAIChatModel
from cookbooks.skills_evaluation.runner import SkillsGradingRunner, build_markdown_report

model = OpenAIChatModel(api_key="sk-...", model="gpt-4o")

runner = SkillsGradingRunner(
    model=model,
    weights={
        "threat_analysis": 2.0,   # 安全维度权重加倍
        "alignment":       1.5,
        "completeness":    1.0,
        "relevance":       1.0,
        "structure":       0.5,   # 降低结构维度权重
    },
    # 自定义通过阈值（可选）
    thresholds={
        "threat_analysis": 3,     # 必须达到 Low 或更好
        "alignment":       2,
        "completeness":    2,
        "relevance":       2,
        "structure":       2,
    },
)

results = asyncio.run(
    runner.arun(
        "/path/to/skills/",
        task_description="自动化代码审查 Pull Request",
    )
)

for r in results:
    status = "PASS" if r.passed else "FAIL"
    print(f"{r.skill_name}: {r.weighted_score * 100:.1f}/100 — {status}")

# 生成 Markdown 报告
print(build_markdown_report(results))
```

### 禁用某个维度

将对应维度的权重设为 `0.0` 即可跳过该维度：

```python
runner = SkillsGradingRunner(
    model=model,
    weights={
        "threat_analysis": 1.0,
        "alignment":       0.0,   # 跳过 Alignment
        "completeness":    1.0,
        "relevance":       0.0,   # 跳过 Relevance
        "structure":       1.0,
    },
)
```

---

## 报告示例

以下为对 `agentic-eval` 技能包的实际评估输出。

### 终端输出

```
============================================================
Skill : agentic-eval
Path  : /workspace/OpenJudge/.agents/skills/agentic-eval
Score : 0.900  ✅ PASS
Time  : 5.0s
────────────────────────────────────────────────────────────
  [threat_analysis  ] ✅  score=4  norm=1.00  w=1.0
    reason: The skill package 'agentic-eval' contains no security findings…
  [alignment        ] ✅  score=3  norm=1.00  w=1.0
    reason: No scripts found; alignment check not applicable.
  [completeness     ] ✅  score=3  norm=1.00  w=1.0
    reason: The skill provides clear goals, explicit steps via Python code snippets…
  [relevance        ] ✅  score=3  norm=1.00  w=1.0
    reason: The skill's name, description, and content directly address the task…
  [structure        ] ✅  score=3  norm=0.50  w=1.0
    reason: The skill is adequate but has gaps in Knowledge Delta (D1)…
```

### Markdown 报告

---

# Skills Evaluation Report

_Total skills evaluated: **1** — Passed: **1** / 1_

## Summary

| Skill | Score | Result |
|-------|------:|--------|
| `agentic-eval` | 90.0 | ✅ Pass |

---

# Skill Evaluation Report: `agentic-eval`

> **Overall score: 90.0 / 100 — ✅ PASS**  _(evaluated in 5.0s)_

**Path:** `.agents/skills/agentic-eval`

## Dimension Summary

| Dimension | Score | Normalised | Weight | Result |
|-----------|------:|-----------:|-------:|--------|
| Threat Analysis | 4 | 1.00 | 1.0 | ✅ Pass |
| Alignment | 3 | 1.00 | 1.0 | ✅ Pass |
| Completeness | 3 | 1.00 | 1.0 | ✅ Pass |
| Relevance | 3 | 1.00 | 1.0 | ✅ Pass |
| Structure | 3 | 0.50 | 1.0 | ✅ Pass |

## Dimension Details

### Threat Analysis

- **Score:** 4  |  **Normalised:** 1.00  |  **Weight:** 1.0  |  **Result:** ✅ Pass

The skill package 'agentic-eval' contains no security findings. The YAML manifest and markdown instructions describe legitimate evaluation patterns without prompt injection, credential theft, or tool abuse.

### Alignment

- **Score:** 3  |  **Normalised:** 1.00  |  **Weight:** 1.0  |  **Result:** ✅ Pass

No scripts found; alignment check not applicable.

### Completeness

- **Score:** 3  |  **Normalised:** 1.00  |  **Weight:** 1.0  |  **Result:** ✅ Pass

The skill provides clear goals, explicit steps via Python code snippets for three distinct patterns (Basic Reflection, Evaluator-Optimizer, Code-Specific), and defines inputs/outputs within those examples. It addresses failure modes by including iteration limits and convergence checks.

### Relevance

- **Score:** 3  |  **Normalised:** 1.00  |  **Weight:** 1.0  |  **Result:** ✅ Pass

The skill's name, description, and content directly address the task of improving AI agent outputs. It provides concrete, actionable implementation patterns with code examples that explicitly demonstrate iterative refinement loops.

### Structure

- **Score:** 3  |  **Normalised:** 0.50  |  **Weight:** 1.0  |  **Result:** ✅ Pass

The skill is adequate but has gaps in Knowledge Delta (D1) and Mindset + Procedures (D2). Some content consists of generic tutorial patterns that an AI agent already knows. The description (D3) could use stronger KEYWORDS and concrete trigger scenarios. Practical Usability (D6) is limited because code examples rely on undefined dependencies (`llm`, `run_tests`). There is no Anti-Pattern (D7) section.

---

### JSON 报告结构

```json
[
  {
    "skill_name": "agentic-eval",
    "skill_path": "/workspace/OpenJudge/.agents/skills/agentic-eval",
    "weighted_score": 0.9,
    "passed": true,
    "grading_duration_seconds": 5.0,
    "dimensions": {
      "threat_analysis": {
        "score": 4,
        "normalized_score": 1.0,
        "weight": 1.0,
        "reason": "The skill package contains no security findings...",
        "passed": true,
        "error": null,
        "metadata": { "findings": [], "threshold": 3 }
      },
      "alignment": { "score": 3, "normalized_score": 1.0, "passed": true, "..." : "..." },
      "completeness": { "score": 3, "normalized_score": 1.0, "passed": true, "...": "..." },
      "relevance": { "score": 3, "normalized_score": 1.0, "passed": true, "...": "..." },
      "structure": { "score": 2, "normalized_score": 0.5, "passed": true, "...": "..." }
    },
    "errors": []
  }
]
```
