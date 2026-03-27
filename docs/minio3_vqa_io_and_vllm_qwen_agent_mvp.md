# Mini-o3 图片问答（含 Crop）输入/输出格式与最小可行 Demo

本文给出三件事：

1. **Mini-o3 在本仓库中训练/推理时的多轮图像问答 I/O 格式**（重点是 `<think> / <grounding> / <answer>`）。
2. **模型会进行几次对话、何时停止**（按 Mini-o3 原生逻辑抽象）。
3. **是否可以用 vLLM 实现一次业务调用**，并采用 Mini-o3 原生 crop 方式。

> 结论：可以做“单次业务调用”（你只调一次 `ask_once()`），但内部通常仍是“模型输出 -> 工具执行 -> 模型续答”的多轮推理。

---

## 1) Mini-o3 的关键输出协议（本仓库）

系统提示词定义了标准格式：

- 思考过程放在 `<think>...</think>`。
- 需要放大局部时输出：
  - `<grounding>{"bbox_2d": [x0, y0, x1, y1], "source": "original_image"}</grounding>`
  - `source` 也可为 `observation_i`（第 i 次观察图）。
- 最终答案放在 `<answer>...</answer>`。

仓库默认训练配置也是 `tool_crop` + `data.tool_call=crop` + `vllm_multi_turn_tool_call`，即基于 vLLM 的多轮工具调用流程。

---

## 2) 模型会进行几次对话？何时停止？

在 Mini-o3 的实现中，本质是**循环多轮**，每轮检查模型是否还在请求 `<grounding>`：

1. 如果输出包含 `<grounding>`，触发 crop 并追加 Observation，继续下一轮；
2. 如果本轮没有 `<grounding>`，说明模型选择直接回答，停止；
3. 若触达最大轮数（训练参数示例里常见 `max_generation_round=6`），强制停止；
4. 还会有额外的安全停止条件（如上下文长度/最大图像数量约束）。

对应你线上服务可抽象为 3 个主要 stop 条件：

- `final_answer`：检测到 `<answer>...</answer>`；
- `no_grounding`：未检测到 `<grounding>`；
- `max_rounds`：达到上限轮数。

---

## 3) Mini-o3 的 crop 原生实现方式（优先）

Mini-o3 原生 crop 核心是：

- 解析 `bbox_2d` + `source`；
- 若使用相对坐标，则乘以当前图像尺寸得到像素坐标；
- 进行边界裁剪（clamp）和合法性检查（`x0 < x1`, `y0 < y1`）；
- 用 PIL 裁图并把新图作为 observation 追加到多模态上下文。

这就是我在示例脚本里优先采用的实现路径（native）。

---

## 4) 一次完整交互 I/O（建议）

### 输入（给模型）

```json
{
  "system": "You are a helpful assistant...<think>/<grounding>/<answer>...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "图中蓝色招牌的店名是什么？"},
        {"type": "image_url", "image_url": {"url": "file:///abs/path/demo.jpg"}}
      ]
    }
  ]
}
```

### 第一次输出（可能触发 crop）

```text
<think>我需要放大右侧蓝色招牌以辨认文字。</think>
<grounding>{"bbox_2d": [0.62, 0.31, 0.79, 0.45], "source": "original_image"}</grounding>
```

### 工具执行输入（native）

```json
{
  "image": "/abs/path/demo.jpg",
  "bbox_2d": [0.62, 0.31, 0.79, 0.45],
  "source": "original_image"
}
```

### 追加 Observation 后再次请求模型

```json
{
  "messages_append": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "After Action 0, here is Observation 1. Continue."},
        {"type": "image_url", "image_url": {"url": "file:///tmp/observation_1.jpg"}}
      ]
    }
  ]
}
```

### 最终输出

```text
<think>放大后可读到招牌文字为 J&optica。</think>
<answer>J&optica</answer>
```

---

## 5) 代码说明

`examples/generation/minio3_vllm_qwen_agent_mvp.py` 已实现：

- `ask_once()`：对外单次调用，内部自动多轮；
- `max_rounds` 控制最大轮数；
- 使用 `_native_crop_relative()` 原生裁剪。



## 6) 一键启动 vLLM（单卡，本地模型）

已新增脚本：`examples/generation/run_vllm_minio3_1gpu.sh`

- 默认本地模型路径：`/data4/home/models/Mini-o3-7B-v1`
- 默认单卡：`GPU_ID=0`
- 默认端口：`8000`
- 多模态约束：`--limit-mm-per-prompt '{"image": 10}'`

启动方式：

```bash
bash examples/generation/run_vllm_minio3_1gpu.sh
```

若要改端口/卡号/模型路径：

```bash
MODEL_PATH=/data4/home/models/Mini-o3-7B-v1 PORT=8000 GPU_ID=0 \
  bash examples/generation/run_vllm_minio3_1gpu.sh
```
