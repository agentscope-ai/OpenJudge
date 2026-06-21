---
name: meta-eval
description: >
  Use when the user wants to build an evaluation system for an LLM/agent application
  but doesn't know where to start — they have traces, prompts, RAG pipelines, or
  nothing at all. Also use when the user mentions evaluation, eval, benchmarking,
  testing LLM quality, measuring agent performance, assessing RAG accuracy, or
  wants to compare prompts/models. This skill is the entry router: it asks diagnostic
  questions then recommends which sub-skill (local workflow) to use next.
---

<HARD-GATE>
NO sub-skill recommendation WITHOUT identifying data_form + label_status (these two pick the entry workflow).
ALWAYS give a provisional recommendation once data_form + label_status are known, even if stakes/user_prior are still unknown — then ask the remaining questions to refine the downstream path. Do not withhold the route while waiting on stakes.
</HARD-GATE>

# Meta Eval

Entry router for the eval skill collection. You diagnose what the user has and route
them to the right sub-skill. You don't do evaluation yourself — you're the triage desk.

Each sub-skill is self-contained: it carries inline the data shapes, statistics, and data
principles it needs, so it can be installed and used on its own.

## Checklist

You MUST create a task for each item and complete them in order:

1. **Ask 4 diagnostic questions** — data, labels, stakes, domain knowledge
2. **Match triage table** — map user scenario to sub-skill
3. **Recommend sub-skill** — tell the user which workflow to use and why
4. **Record routing decision** — write a brief summary of what was diagnosed and recommended

## Diagnostic Questions

Ask these 4 questions (all at once — don't drip-feed):

```
To route you to the right evaluation skill, I need to understand your situation:

1. What data do you have?
   a) Agent traces / production logs
   b) Product spec / design docs
   c) Nothing yet — starting from scratch

2. Do you have human labels?
   a) Yes, ≥50 labeled examples
   b) Some, but fewer than 50
   c) None

3. What are the stakes?
   a) Low — internal experimentation, exploring options
   b) Production — customer-facing, quality matters
   c) Regulated — compliance requirements, audit trail needed

4. How well do you know this evaluation domain?
   a) Very well — have clear standards and criteria
   b) Somewhat — general idea but need structure
   c) Not well — exploring what "good" even means
```

**Shortcut rule**: `data_form` + `label_status` already determine the entry workflow
(see triage table). The moment those two are clear — even if stakes and domain knowledge
are not — give the provisional recommendation AND ask the remaining questions in the same
message. `stakes` and `user_prior` refine the *downstream* path (how much calibration rigor,
how fast a path), not the entry point. Never make the user wait a round-trip for a route you
can already determine.

Example: "no logs, no labels" → recommend `08-bootstrap` now, and ask stakes/domain to tune
the roadmap. Don't reply with only the questionnaire.

## Triage Table

Match the user's situation to a sub-skill:

These are local workflows under `skills/eval_pipeline/`, not packages to install — "use"
a workflow means open and follow that sub-skill.

| User says / has | Use workflow | What it does |
|----------------|---------|--------------|
| "I have agent traces / production logs" | `01-eval-design` | Extract eval dimensions from traces → design dataset in OpenJudge format |
| "I have principles/criteria but need test data" | `01-eval-design` | Stratified sampling + adversarial generation → OpenJudge dataset |
| "I have principles but don't know which graders to use" | `02-metric-design` | Select OpenJudge graders by output type → generate executable pipeline code |
| "I changed my prompt, is it better?" | `06-prompt-regression` | A/B comparison with PairwiseAnalyzer, win rates + statistical significance |
| "I have a RAG system" | `05-rag-eval` | Retrieval + generation separation, hallucination detection, diagnostic matrix |
| "I have a judge + labels, want to check accuracy" | `03-align-human` | TPR/TNR calibration, kappa agreement, human-reduction roadmap |
| "I want to do safety/security testing" | `07-redteam` | Attack surface analysis, jailbreak/injection generation, harmfulness grading |
| "I've run multiple skills, want a comprehensive report" | `04-eval-report` | Cross-skill analysis, maturity dashboard, prioritized actions |
| "Nothing — starting from scratch" | `08-bootstrap` | Zero-shot grader generation via SimpleRubricsGenerator, v0 in 30 minutes |
| None of the above match | — | Say "this scenario isn't covered yet" and suggest filing an issue |

## Output

After diagnosis, respond with:

```
Diagnosis: data=[data_form] | labels=[label_status] | stakes=[value or "asking"] | domain=[value or "asking"]

Recommended workflow: `[skill-name]`   (provisional if stakes/domain unknown)

Why: [one sentence explaining the routing decision from data_form + label_status]

What this workflow will do: [one sentence about the output — e.g., "produces an
OpenJudge-compatible dataset with stratified sampling"]

To refine the path, also tell me: [stakes / domain knowledge, if still unknown]
```

Recommend exactly ONE workflow as the immediate next step. Do NOT list a second
workflow as a current action — that splits the user's focus. If they ask "what comes
after," point them to the Canonical Workflow below as a *map for later*, explicitly
framed as "once you finish [recommended workflow]," not as a second thing to do now.

A `?` marks a field you are still asking about. Give the recommendation now; refine later.

## Canonical Workflow (the standard lifecycle)

Most evaluation builds follow this order. Use it to sequence sub-skills and to state
preconditions — recommend the *next* workflow only when its inputs exist.

```
1. 00-meta-eval        route to the right entry workflow
2. entry point:
     - have traces/spec  → 01-eval-design   (build the dataset)
     - nothing at all    → 08-bootstrap      (uncalibrated v0 + roadmap to labels)
3. 02-metric-design    select graders, build the GradingRunner pipeline
4. RUN the evaluation  (produces scores; needed before any A/B or calibration)
5. 03-align-human      ONLY once ≥50 human labels exist — calibrate before any
                       production gate. Production stakes REQUIRE this step.
6. scenario module (as needed):
     - 05-rag-eval          retrieval vs generation diagnosis
     - 06-prompt-regression REQUIRES paired baseline+candidate outputs on shared
                            queries — do not route here before both prompts have
                            been run and their outputs collected
     - 07-redteam           policy-first safety + over-refusal
7. 04-eval-report      synthesize maturity + ship readiness
```

Precondition rules to enforce when routing:

- **Do not recommend `06-prompt-regression`** until the user has *run both* the baseline
  and candidate prompts and has their outputs paired by query. Comparing prompts that
  haven't been run yet is impossible.
- **Do not call anything "production-ready"** without labels + `03-align-human` calibration.
  If stakes are production/regulated and no labels exist, the path MUST explicitly include
  two steps before any ship decision: (1) collect ≥50 human labels, (2) run `03-align-human`
  to calibrate. State both steps every time production is in scope — an unlabeled system is
  never production-ready, no matter how good the scores look.
- **Have traces but no labels** is the common case: go `01-eval-design` → `02-metric-design`
  → run → collect labels → `03-align-human`. Do not jump to bootstrap (you have data) or to
  prompt-regression (no paired outputs yet).

## Red Flags — STOP and Re-evaluate

If you catch yourself thinking:

- "I can skip the questions, the scenario is obvious" → STOP. Even obvious cases
  have hidden constraints (stakes, label availability) that change the routing.
- "I'll just recommend bootstrap, it's always safe" → STOP. Bootstrap is for
  "nothing" scenarios. If the user has data, they need a data-aware skill.
- "The user didn't mention stakes, so it's probably low" → STOP. Assuming low
  stakes when they might be production is how uncalibrated judges slip through.
- "I'll recommend multiple skills at once" → STOP. Recommend one at a time.
  Users can chain skills but the entry point should be singular.

**All of these mean: Stop. Return to the diagnostic questions.**

## Rationalization Defense

| You might think | Reality |
|----------------|---------|
| "This is just a simple eval question" | "Simple" questions hide complex trade-offs. The 4 questions catch them. |
| "They obviously need X" | Stake levels and label availability change the answer. Low stakes → fast path. Production → must calibrate. |
| "I'll figure it out as we go" | Routing to the wrong skill wastes more time than 4 questions. |
| "The triage table covers everything" | It covers common paths. If nothing matches, say so — don't force-fit. |

## Common Mistakes

- **Recommending bootstrap when the user has data.** Bootstrap's SimpleRubricsGenerator
  is zero-shot and ignores existing labels/traces. If the user has data, use `01-eval-design`.
- **Skipping the stakes question.** Low-stakes scenarios can use fast paths (skip calibration).
  Production scenarios need hybrid mode + calibration. Regulated needs audit trails.
- **Recommending metric-design before eval-design.** Without a dataset, graders have nothing
  to grade. Design the test data first, then the metrics.
- **Not suggesting follow-up skills.** Every skill has a natural next step. Mention it
  so the user knows the path forward.

## What This Skill Doesn't Cover

- Running evaluations directly (sub-skills do that)
- Continuous production monitoring (MLOps domain — use Arize, Braintrust, Datadog LLM)
- Benchmark leaderboards with versioned public releases
- Real-time signal stacks embedded in agent harnesses (engineering system design)