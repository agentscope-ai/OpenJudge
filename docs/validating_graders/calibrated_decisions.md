# Calibrated decisions and selective evaluation

Use this offline workflow when a grader supplies an explicit probability that its
binary decision equals an independently supplied reference decision. It measures
confidence reliability, fits a small monotone postprocessor, and selects a policy
that can intentionally abstain. It does not convert arbitrary rubric scores into
probabilities or change `GraderScore.score`.

Declare the correctness event, two decision labels, confidence field, model/judge
versions, sample IDs and task/group IDs. Keep repeated trajectories, alternative
responses and repeated votes for one question in the same group. Supply three
explicit partitions: calibration for fitting, validation for threshold selection,
and held-out test for reporting. Overlapping sample or group IDs fail, including
IDs belonging to excluded or failed observations.

```python
from openjudge.calibration import (
    CachedPartition, CalibrationContract, CalibrationScope, FitSettings,
    SelectionSettings, WorkflowSettings, run_calibration_workflow,
)

contract = CalibrationContract(
    target_id="judge_matches_independent_reference_v1",
    confidence_path="metadata.confidence",  # never silently falls back to score
    prediction_path="score", decision_labels=(0, 1),
)
scope = CalibrationScope(judge_id="judge_prompt_v1", model_id="judge_model_v1", scope_id="my_tasks")
report = run_calibration_workflow(
    CachedPartition.load("calibration.json"),
    CachedPartition.load("validation.json"),
    CachedPartition.load("test.json"),
    contract, scope,
    WorkflowSettings(
        fit=FitSettings(method="isotonic"),
        selection=SelectionSettings(risk_target=0.1, min_accepted=30),
    ),
)
report.save("report.json")
```

Create a native cache with
`CachedPartition.from_runner(dataset, runner_results, "grader_name", role="calibration")`.
Caches retain supplied sample fields and native result metadata/reasons; project
sensitive fields before saving. Reports include summary statistics, identity
fingerprints and safe numerical artifacts, without original prompts, reasoning or
per-row target labels. Content checksums detect accidental changes, not malicious
re-signing or source authenticity.

`fit_calibrator`, `select_policy` and `evaluate_policy` are separate public calls.
`CalibrationAnalyzer` and `SelectivePredictionAnalyzer` follow the native
`BaseValidationAnalyzer` interface and only evaluate fixed inputs. They never fit
or optimize on analyzed labels. `DecisionAdapter` is opt-in, needs no target label,
and returns an accepted/abstained/unavailable envelope referencing the unmodified
native result. It does not make more model calls or alter voting/average strategies.

Identity, sigmoid on raw probability and isotonic methods are available. Sigmoid
uses a nonnegative slope; constant inputs/outcomes and a negative unconstrained
relationship produce an explicitly recorded constant empirical-rate model.
`FitSettings.regularization` is sklearn's inverse L2 strength `C`. Isotonic uses
constant endpoint extension outside its fitted confidence range. Reports flag
raw test confidences outside that range and unseen supplied domain labels.

Strict input validation is the default. `invalid_policy="exclude"` retains distinct
missing/invalid-label, prediction, confidence and unsupported-result statuses.
Provider errors remain failures and explicit user exclusions remain visible.
Binary log-loss endpoint clipping is configured and disclosed; out-of-range raw
confidence is rejected, not clipped into validity.

Reliability reports include bin support, mean confidence, empirical correctness,
ECE/MCE, Brier, clipped log loss, decision accuracy and ROC AUC where defined.
Selective curves add complete confidence tie groups. AURC is the right-endpoint
step area over evaluable coverage; total-input coverage is separately reported.
All-abstain accepted risk is null. Unconditional error rate counts accepted wrong
decisions plus provider failures over all inputs; omissions and abstentions have
separate counts and must not be read as successful predictions.

Default selection maximizes validation support under an empirical risk target,
with no population-risk guarantee. An empty feasible set produces abstain-all.
An optional `criterion="hoeffding"` requires an explicit fixed increasing
`candidates` grid, `assume_iid_observations=True`, one valid observation per group,
and a declared `alpha`. It adjusts across every declared candidate with a union
bound. IDs do not prove independence: the grid must be chosen before inspecting
validation outcomes and observation pairs must be independent and identically
distributed. Distribution shift invalidates the interpretation.

Group bootstrap intervals are descriptive percentile intervals from whole-task
resampling with an isolated deterministic RNG. They keep calibrator and policy
fixed, report effective replicate counts, and cannot repair dependence between
undeclared duplicate tasks. Keep final-test labels held out from prompt/model
selection and disclose repeated test inspection; software cannot enforce that
experimental discipline or establish label quality.

Run the self-contained native-runner example from the repository root:

```bash
python cookbooks/grader_validation/calibrated_decisions.py --output-dir /tmp/calibration-example
```

Its labels and domain shift are deliberately constructed. The example verifies
numerical and integration behavior; it is not a benchmark or evidence that real
agent performance improves.
