name: task_type_classification
type: skill
description: Classify a case into recognition/part/spatial task types for downstream routing.
input_schema:
  question: str
  benchmark_category: str | null
  llm_hint_task_type: str | null
output_schema:
  task_type: enum[object_affordance_recognition, object_affordance_prediction, spatial_affordance_localization]
  confidence: float
  reason: str
policy:
  - Prefer benchmark category mapping when provided.
  - Otherwise use rule-based lexical cues or valid llm planner hint.
  - Fall back to object_affordance_prediction when uncertain.

# Skill: task_type_classification

This skill is the first routing policy in the harness. It returns a structured decision object with `task_type`, `confidence`, and `reason` and is persisted in trace metadata.
