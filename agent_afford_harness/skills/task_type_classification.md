# Skill: task_type_classification

This skill is the first routing policy in the harness. It takes a user question (and optionally benchmark metadata) and assigns the sample to one of three branches: `object_affordance_recognition`, `object_affordance_prediction`, or `spatial_affordance_localization`. Because every downstream decision depends on this branch, classification quality has first-order impact on final reward.

In the current implementation, the orchestrator first checks benchmark category labels when they are available. If labels are not available, the skill falls back to lexical heuristics. When an LLM planner is enabled, planner output can override the rule-based decision if it matches a valid task type.

The skill returns a structured decision object with `task_type`, `confidence`, and `reason`. This is important for auditability: traces can later be grouped by misclassification patterns instead of only by final score.

Failure policy is conservative. If the model is uncertain, the pipeline defaults to `object_affordance_prediction`, which keeps the run in a geometry-capable path rather than terminating early. The selected decision is recorded in `selected_skills` and persisted as `metadata.task_info`.
