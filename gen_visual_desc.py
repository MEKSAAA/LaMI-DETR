import os
import json
import asyncio
import random
from typing import Dict, List

from openai import OpenAI

# ================== 配置 ==================
INPUT_JSON = "all_categories.json"              # 你的物体类别 json
OUTPUT_JSONL = "visual_desc.jsonl"            # 增量写
OUTPUT_JSON = "visual_desc.json"              # 最终合并

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
CONCURRENCY = 80
MAX_RETRIES = 5
TIMEOUT = 120
TEMPERATURE = 0.2

# ================== DeepSeek Client ==================
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY","sk-e8c333950a3c4ccc9b4a3f0232cb2ac8"),
    base_url=BASE_URL
)

# ================== Prompt ==================
def build_messages(category: str):
    return [
        {
            "role": "system",
            "content": "You are a precise visual description generator for object categories."
        },
        {
            "role": "user",
            "content": f"""
Task: Convert the given category name into a list of visual attributes.

Requirements:
- Output ONLY valid JSON, no extra text.
- Format:
  "{category}": [attribute1, attribute2, ...]
- Attributes must cover:
  • basic visual type
  • typical colors
  • key shape or outline
  • notable visible parts
- Use English short phrases only.
- Only visually identifiable features.
- Each attribute must be an independent list item.

Category: {category}
"""
        }
    ]

# ================== Utils ==================
def load_categories(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for k in ["merged_categories", "categories"]:
            if k in data and isinstance(data[k], list):
                return sorted(set(str(x).strip() for x in data[k] if str(x).strip()))

        merged = []
        for v in data.values():
            if isinstance(v, list):
                merged.extend(str(x).strip() for x in v if str(x).strip())
        return sorted(set(merged))

    raise ValueError("Unsupported JSON format")

def load_existing(jsonl_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(jsonl_path):
        return {}

    results = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    results.update(obj)
            except json.JSONDecodeError:
                continue
    return results

def append_jsonl(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def parse_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l, r = text.find("{"), text.rfind("}")
        if l != -1 and r != -1:
            return json.loads(text[l:r+1])
        raise

# ================== API Call ==================
async def generate_one(category: str, sem: asyncio.Semaphore) -> Dict[str, List[str]]:
    messages = build_messages(category)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                def _call():
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=TEMPERATURE,
                        stream=False,
                    )
                    return resp.choices[0].message.content

                content = await asyncio.wait_for(
                    asyncio.to_thread(_call),
                    timeout=TIMEOUT
                )

            parsed = parse_json(content)
            
            # 优先精确匹配
            if category in parsed and isinstance(parsed[category], list):
                return {category: [str(x).strip() for x in parsed[category] if str(x).strip()]}
            
            # 如果精确匹配失败，取返回的第一个有效列表值（模型可能纠正了拼写）
            for key, val in parsed.items():
                if isinstance(val, list) and val:
                    return {category: [str(x).strip() for x in val if str(x).strip()]}

            raise ValueError("Category key missing in response")

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = min(2 ** attempt, 30) + random.random()
            print(f"[Retry {attempt}] {category} -> {e}, sleep {wait:.1f}s")
            await asyncio.sleep(wait)

    return {}

# ================== Main ==================
async def main():
    categories = load_categories(INPUT_JSON)
    print(f"总类别数: {len(categories)}")

    existing = load_existing(OUTPUT_JSONL)
    print(f"已完成类别数: {len(existing)}")

    pending = [c for c in categories if c not in existing]
    print(f"待处理类别数: {len(pending)}")

    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        asyncio.create_task(generate_one(cat, sem))
        for cat in pending
    ]

    results = dict(existing)
    done = 0

    for fut in asyncio.as_completed(tasks):
        out = await fut
        done += 1
        if out:
            append_jsonl(OUTPUT_JSONL, out)
            results.update(out)
            k = next(iter(out))
            print(f"[{done}/{len(pending)}] ✅ {k}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("全部完成 ✅")

if __name__ == "__main__":
    asyncio.run(main())
