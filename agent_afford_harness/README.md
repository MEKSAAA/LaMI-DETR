# Agentic Affordance Harness (RoboAfford-Eval)

参数冻结的 **Orchestrator + Skills + Tools** 流水线：输入图像与问题，产出 RoboAfford 可用的 **归一化坐标点**，并写出结构化 **trace**（便于后续 self-evolve）。

## 依赖

与 LaMI-DETR 一致（`detectron2`、`torch`、权重等）。可选：`opencv-python`、`PyYAML`、`pytest`。

将仓库根目录 **LaMI-DETR** 加入 `PYTHONPATH`，以便导入包 `agent_afford_harness`。

## 环境变量

| 变量 | 含义 |
|------|------|
| `AGENT_HARNESS_LAMI_CONFIG` | 覆盖 LaMI 推理 yaml 路径 |
| `ZHIPU_API_KEY` / `GLM_API_KEY` | 启用智谱 **联网搜索**；不设则使用确定性 stub |
| `AGENT_HARNESS_LLM_MODE` | 主推理器模式：`rules` / `local_qwen_vl` / `api_vlm` |
| `AGENT_HARNESS_LOCAL_VLM_PATH` | 本地 VLM 路径（默认 `/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct`） |
| `AGENT_HARNESS_API_MODEL` | API 模型名（如 `qwen-vl-max`、`gpt-4o-mini`） |
| `AGENT_HARNESS_API_BASE_URL` | API base URL（默认 Doubao ARK: `https://ark.cn-beijing.volces.com/api/v3`） |
| `AGENT_HARNESS_API_KEY` | API key（也可用 `ARK_API_KEY` / `OPENAI_API_KEY`） |
| `ARK_API_KEY` | Doubao/方舟 API key（推荐） |
| `AGENT_HARNESS_DOUBAO_GROUNDING_MODEL` | Grounding 模型（输出 `<bbox>` 的模型） |
| `AGENT_HARNESS_GROUNDING_BACKEND` | Grounding 后端：`lami` / `doubao_grounding`（覆盖 yaml） |

## 运行（在 `LaMI-DETR` 目录下）

```bash
export PYTHONPATH=/path/to/LaMI-DETR

python -m agent_afford_harness.harness.run_single_case \
  --image examples/richhf/images/1.jpg \
  --question "Highlight points on the flower." \
  --sample-id demo_richhf
```

## Bash 脚本（推荐）

已提供 `agent_afford_harness/scripts/`：

- `run_single_demo.sh`：规则链快速跑通（使用真实 LaMI）
- `run_single_local_qwen.sh`：本地 Qwen2.5-VL 主推理
- `run_single_api_doubao.sh`：API 主推理（Doubao）
- `run_single_doubao_grounding.sh`：强制 Doubao Grounding（1000x1000 坐标）
- `run_showcase_subset.sh`：跑 9-case 子集
- `visualize_trace.sh`：渲染 trace 到图片

先给执行权限：

```bash
chmod +x agent_afford_harness/scripts/*.sh
```

示例：

```bash
# 1) demo
agent_afford_harness/scripts/run_single_demo.sh

# 2) 本地Qwen
agent_afford_harness/scripts/run_single_local_qwen.sh

# 3) API Doubao（需 ARK_API_KEY）
ARK_API_KEY=xxxx agent_afford_harness/scripts/run_single_api_doubao.sh

# 4) Doubao grounding
ARK_API_KEY=xxxx agent_afford_harness/scripts/run_single_doubao_grounding.sh

# 5) 跑子集
agent_afford_harness/scripts/run_showcase_subset.sh

# 6) 可视化
agent_afford_harness/scripts/visualize_trace.sh \
  agent_afford_harness/outputs/traces/demo_richhf.json \
  agent_afford_harness/outputs/debug_images/demo.png
```

### 主推理器：本地 Qwen2.5-VL

```bash
export AGENT_HARNESS_LLM_MODE=local_qwen_vl
export AGENT_HARNESS_LOCAL_VLM_PATH=/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct
python -m agent_afford_harness.harness.run_single_case \
  --image examples/richhf/images/1.jpg \
  --question "What part of a mug should be gripped to lift it?" \
  --sample-id demo_local_qwen
```

### 主推理器：API 模式

```bash
export AGENT_HARNESS_LLM_MODE=api_vlm
export AGENT_HARNESS_API_MODEL=doubao-seed-2-0-lite-250821
export AGENT_HARNESS_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export ARK_API_KEY=xxxx
python -m agent_afford_harness.harness.run_single_case \
  --image examples/richhf/images/1.jpg \
  --question "Locate points in the free area between two objects." \
  --sample-id demo_api_vlm
```

### 使用 Doubao Grounding（1000x1000 坐标）

在 `agent_afford_harness/configs/pipeline.yaml` 中将：

```yaml
orchestrator:
  grounding_backend: doubao_grounding
```

也可以直接用环境变量临时切换（无需改 yaml）：

```bash
export AGENT_HARNESS_GROUNDING_BACKEND=doubao_grounding
```

说明：
- Doubao grounding 返回 `<bbox>x1 y1 x2 y2</bbox>`，坐标在 **[0,999] 的 1000x1000 网格**。
- Pipeline 内部统一使用像素坐标；工具层会自动双向转换：
  - 输入：1000 坐标 -> 像素
  - 输出：像素 -> 1000（在每个 box 的 `bbox_1000` 字段）
- 最终点仍输出 RoboAfford 所需 `[0,1]`，同时在 trace 增加 `metadata.final_points_1000`。

> 若 `local_qwen_vl/api_vlm` 调用失败，会自动回退到规则路由，并在 trace 的 `metadata.llm_plan_error` 中记录错误。

子集评测（9 例 showcase；需 `projects26/RoboAfford/images` 与 `masks` 存在）：

```bash
python -m agent_afford_harness.harness.run_subset
```

可视化 trace（LaMI 框 + 红点）：

```bash
python -m agent_afford_harness.harness.visualize_trace \
  --trace agent_afford_harness/outputs/traces/demo_richhf.json \
  --out agent_afford_harness/outputs/debug_images/demo.png
```

## 布局

见仓库内 `configs/`、`agent/`、`tools/`、`data/`、`harness/`、`skills/*.md`。

文档入口：
- Skill 规范：`skills/*.md`
- Tool 规范：`tools/*_tool.md` 与 `tools/*_grounder.md`
- 机器可读注册表：`configs/skills.yaml`、`configs/tools.yaml`

## 测试

```bash
cd /path/to/LaMI-DETR
PYTHONPATH=. pytest agent_afford_harness/tests -q
```
