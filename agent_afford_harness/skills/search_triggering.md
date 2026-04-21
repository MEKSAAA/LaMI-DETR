name: search_triggering
type: skill
description: Decide if external search is worth latency for long-tail/attribute-heavy questions.
input_schema:
  question: str
  task_info: dict
  llm_need_search: bool | null
  llm_search_queries: list[str] | null
output_schema:
  need_search: bool
  queries: list[str]
  reasons: list[str]
policy:
  - Trigger search for long-tail attributes or long prompts.
  - Allow planner override for need_search and query list.

# Skill: search_triggering

Policy skill for optional knowledge expansion.
