name: lami_prompt_engineering
type: skill
description: Convert benchmark question to detector-friendly prompts and class descriptors.
input_schema:
  question: str
  task_info: dict
  search_result: dict | null
  llm_prompt_overrides: dict | null
output_schema:
  object_prompt: str
  part_prompt: str
  spatial_query: str
  negative_hint: str
  lami_classes: dict[str, list[str]]
policy:
  - Keep prompt compact and detector-oriented.
  - Inject search cues when available.
  - Respect planner prompt overrides if valid.

# Skill: lami_prompt_engineering

Policy skill for prompt shaping before grounding.
