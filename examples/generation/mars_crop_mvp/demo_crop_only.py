from __future__ import annotations
import argparse
import base64
import io
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
import yaml
from urllib import request
from urllib.error import HTTPError, URLError
from PIL import Image
REPO_SYSTEM_PROMPT = "#Role\nYou are a step-by-step reasoning assistant. \nGiven a question, your task is to solve the problem **one substep at a time**.  \n\n## Guiding Principles  \nAt each turn, you must **either**:  \n1. Issue **one specific tool** enclosed in <tool_call> </tool_call> tags,  \n2. Or provide the **final answer** enclosed in <answer> </answer> tags.  \n\nAll outputs **must begin with a thought** enclosed in <thinking> </thinking> tags, explaining your current reasoning and what to do next.  \n\n## Output Format (strict):  \nAlways start with <thinking>. Do not output the previous reasoning chain. Then, depending on the case, output one of the following:\n\n1. If reasoning continues:  \n<thinking> Your current reasoning and next plan </thinking>  \n<tool_call> One precise, tool call to assist your reasoning </tool_call>\n\n2. If ready to conclude:  \n<thinking> Summarize all reasoning and derive the answer </thinking>  \n<answer> Final answer </answer>"
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
DEFAULT_TOOL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tools_crop_only.yaml")
@dataclass
class DemoConfig:
    base_url: str = os.getenv("MARS_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key: str = os.getenv("MARS_API_KEY", "EMPTY")
    model: str = os.getenv("MARS_MODEL", "SenseNova-MARS-8B")
    max_tokens: int = int(os.getenv("MARS_MAX_TOKENS", "4096"))
    max_rounds: int = int(os.getenv("MARS_MAX_ROUNDS", "10"))
    work_dir: str = os.getenv("MARS_WORK_DIR", "/tmp/sensenova_mars_crop_demo")
    tool_config_path: str = os.getenv("MARS_TOOL_CONFIG", DEFAULT_TOOL_CONFIG_PATH)
    min_pixels: int = int(os.getenv("MARS_MIN_PIXELS", "65536"))
    max_pixels: int = int(os.getenv("MARS_MAX_PIXELS", "8294400"))
    factor: int = int(os.getenv("MARS_FACTOR", "32"))
    qwen_vl_processing: bool = os.getenv("MARS_QWEN_VL_PROCESSING", "true").lower() == "true"
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor
def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor
def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor
def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 8294400) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio too extreme")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar
def process_image(image: Image.Image, min_pixels: int = 65536, max_pixels: int = 8294400, factor: int = 32, qwen_vl_processing: bool = True) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if not qwen_vl_processing:
        return image
    width, height = image.size
    resized_height, resized_width = smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if (resized_width, resized_height) != (width, height):
        image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    return image
def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
def load_tool_config(tool_config_path: str) -> str:
    with open(tool_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tools = config.get("tools", [])
    tool_definitions = []
    for tool in tools:
        schema = tool.get("tool_schema", {})
        tool_definitions.append(json.dumps(schema, ensure_ascii=False))
    tool_def_str = "\n".join(tool_definitions)
    return f'''# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tool_def_str}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'''
def build_system_prompt(tool_config_path: str) -> str:
    return REPO_SYSTEM_PROMPT + "\n\n" + load_tool_config(tool_config_path)
def parse_tool_call(text: str) -> Optional[dict[str, Any]]:
    match = TOOL_CALL_PATTERN.search(text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
def has_answer_tag(text: str) -> bool:
    return ANSWER_PATTERN.search(text or "") is not None
def crop_image(image: Image.Image, bbox: list[float], coord_scale: float = 1000.0, min_pixels: int = 65536, max_pixels: int = 8294400, factor: int = 32, qwen_vl_processing: bool = True, padding: tuple[float, float] = (0.0, 0.0)) -> Image.Image:
    img_w, img_h = image.size
    x1, y1, x2, y2 = [float(c) / coord_scale for c in bbox]
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if padding[0] > 0 or padding[1] > 0:
        padding_cap = (600.0 / img_w, 600.0 / img_h)
        actual_padding = (min(padding[0], padding_cap[0]), min(padding[1], padding_cap[1]))
        x1 = max(0.0, x1 - actual_padding[0])
        y1 = max(0.0, y1 - actual_padding[1])
        x2 = min(1.0, x2 + actual_padding[0])
        y2 = min(1.0, y2 + actual_padding[1])
    crop_box = (int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h))
    if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
        raise ValueError(f"Invalid bbox after scaling/clamping: {bbox} -> {crop_box}")
    cropped = image.crop(crop_box)
    if qwen_vl_processing:
        w, h = cropped.size
        if w < 28 or h < 28:
            cropped = cropped.resize((max(w, 28), max(h, 28)), Image.Resampling.LANCZOS)
        return process_image(cropped, min_pixels=min_pixels, max_pixels=max_pixels, factor=factor, qwen_vl_processing=qwen_vl_processing)
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    return cropped
def save_image(image: Image.Image, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, "JPEG", quality=95)
    return os.path.abspath(output_path)
def _chat_once(cfg: DemoConfig, messages: list[dict[str, Any]]) -> str:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions" if cfg.base_url.rstrip('/').endswith('/v1') else f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    body = {"model": cfg.model, "messages": messages, "max_tokens": cfg.max_tokens}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
    req = request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {detail[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"Request failed: {e}") from e
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Invalid response: {payload}")
    return choices[0].get("message", {}).get("content") or ""
def ask_once(question: str, image_path: str, cfg: DemoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemoConfig()
    with Image.open(image_path) as image:
        image.load()
        original_image = image.convert("RGB")
    first_view = process_image(original_image, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, factor=cfg.factor, qwen_vl_processing=cfg.qwen_vl_processing)
    observations: list[Image.Image] = [first_view]
    observation_paths: list[str] = [save_image(first_view, os.path.join(cfg.work_dir, "observation_0.jpg"))]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(cfg.tool_config_path)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(first_view)}},
                {"type": "text", "text": question},
            ],
        },
    ]
    trace: list[dict[str, Any]] = []
    for round_id in range(cfg.max_rounds):
        response = _chat_once(cfg, messages)
        step: dict[str, Any] = {"round": round_id, "assistant": response}
        trace.append(step)
        tool_call = parse_tool_call(response)
        if tool_call is None:
            return {
                "stopped_by": "final_answer" if has_answer_tag(response) else "no_tool_call",
                "rounds": round_id + 1,
                "trace": trace,
                "final_response": response,
                "observation_paths": observation_paths,
            }
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {}) or {}
        messages.append({"role": "assistant", "content": response})
        if tool_name != "image_zoom_in_tool":
            tool_response = f"<tool_response>\nError: Tool '{tool_name}' is disabled in this demo. Only image_zoom_in_tool is available.\n</tool_response>"
            messages.append({"role": "user", "content": tool_response})
            step["tool_call"] = tool_call
            step["tool_response"] = tool_response
            continue
        bbox = arguments.get("bbox") or arguments.get("bbox_2d")
        img_idx = arguments.get("img_idx")
        if img_idx is None:
            img_idx = 0
        if not isinstance(bbox, list) or len(bbox) != 4:
            tool_response = "<tool_response>\nError: Invalid bbox format.\n</tool_response>"
            messages.append({"role": "user", "content": tool_response})
            step["tool_call"] = tool_call
            step["tool_response"] = tool_response
            continue
        if not isinstance(img_idx, int) or img_idx < 0 or img_idx >= len(observations):
            tool_response = f"<tool_response>\nError: Image at index {img_idx} not found. Available images: {len(observations)}\n</tool_response>"
            messages.append({"role": "user", "content": tool_response})
            step["tool_call"] = tool_call
            step["tool_response"] = tool_response
            continue
        cropped = crop_image(observations[img_idx], bbox, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, factor=cfg.factor, qwen_vl_processing=cfg.qwen_vl_processing)
        observations.append(cropped)
        crop_path = save_image(cropped, os.path.join(cfg.work_dir, f"observation_{len(observations) - 1}.jpg"))
        observation_paths.append(crop_path)
        tool_response_text = "<tool_response>\nHere is the zoomed image:\n</tool_response>"
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<tool_response>\nHere is the zoomed image:"},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(cropped)}},
                    {"type": "text", "text": "\n</tool_response>"},
                ],
            }
        )
        step["tool_call"] = tool_call
        step["tool_response"] = tool_response_text
        step["observation"] = crop_path
    return {
        "stopped_by": "max_rounds",
        "rounds": cfg.max_rounds,
        "trace": trace,
        "final_response": trace[-1]["assistant"] if trace else "",
        "observation_paths": observation_paths,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description="SenseNova-MARS-8B crop-only MVP demo")
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--base-url", default=os.getenv("MARS_BASE_URL", DemoConfig.base_url))
    parser.add_argument("--api-key", default=os.getenv("MARS_API_KEY", DemoConfig.api_key))
    parser.add_argument("--model", default=os.getenv("MARS_MODEL", DemoConfig.model))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MARS_MAX_TOKENS", DemoConfig.max_tokens)))
    parser.add_argument("--max-rounds", type=int, default=int(os.getenv("MARS_MAX_ROUNDS", DemoConfig.max_rounds)))
    parser.add_argument("--work-dir", default=os.getenv("MARS_WORK_DIR", DemoConfig.work_dir))
    parser.add_argument("--tool-config", default=os.getenv("MARS_TOOL_CONFIG", DEFAULT_TOOL_CONFIG_PATH))
    args = parser.parse_args()
    cfg = DemoConfig(base_url=args.base_url, api_key=args.api_key, model=args.model, max_tokens=args.max_tokens, max_rounds=args.max_rounds, work_dir=args.work_dir, tool_config_path=args.tool_config)
    result = ask_once(args.question, args.image, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
