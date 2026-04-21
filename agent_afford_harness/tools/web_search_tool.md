name: web_search_tool
type: tool
description: Retrieve external visual cues/concepts for long-tail object references.
input_schema:
  queries: list[str]
output_schema:
  queries: list[str]
  results: dict
  backend: str
  fallback_reason: str | null
system_prompt: "Search and summarize concise visual cues useful for detector prompting."

# Tool: web_search_tool

Optional external knowledge tool.
