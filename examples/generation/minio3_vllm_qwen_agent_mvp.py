"""
Minimal viable demo (simplified):
- Serve Mini-o3/Qwen-VL style model via vLLM OpenAI-compatible endpoint
- Use Mini-o3 native-style crop only (relative bbox -> pixel bbox -> crop)
- Stop based on Mini-o3-like multi-turn criteria
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from PIL import Image

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the image provided. "
    "Output your thinking process within <think> and </think>. "
    "When unclear, output <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>. "
    "Coordinates are relative to image width/height (0~1). "
    "If final answer is ready, output <answer>...</answer>."
)

GROUNDING_PATTERN = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL)


@dataclass
class DemoConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    model: str = os.getenv("VLLM_MODEL", "/data4/home/models/Mini-o3-7B-v1")
    max_tokens: int = 512
    max_rounds: int = 6
    work_dir: str = os.getenv("MINIO3_DEMO_WORK_DIR", "/tmp/minio3_demo")


def _chat_once(client: OpenAI, cfg: DemoConfig, messages: list[dict[str, Any]]) -> str:
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=0.2,
        max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content or ""


def _parse_grounding(text: str) -> dict[str, Any] | None:
    m = GROUNDING_PATTERN.search(text)
    if not m:
        return None
    return json.loads(m.group(1).strip())


def _has_final_answer(text: str) -> bool:
    return ANSWER_PATTERN.search(text or "") is not None


def _native_crop_relative(image_path: str, bbox_2d: list[float], output_path: str) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        w, h = image.size

        x0 = int(max(0, min(w - 1, bbox_2d[0] * w)))
        y0 = int(max(0, min(h - 1, bbox_2d[1] * h)))
        x1 = int(max(0, min(w - 1, bbox_2d[2] * w)))
        y1 = int(max(0, min(h - 1, bbox_2d[3] * h)))

        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Invalid bbox after scaling/clamping: {[x0, y0, x1, y1]}")

        crop = image.crop((x0, y0, x1, y1))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        crop.save(output_path)
    return output_path


def ask_once(question: str, image_path: str, cfg: DemoConfig | None = None) -> dict[str, Any]:
    """
    Single business-call wrapper. Internal loop can run multi rounds.

    Stop criteria:
    1) model outputs final <answer> tag
    2) model no longer outputs <grounding>
    3) reaches max_rounds
    """
    cfg = cfg or DemoConfig()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    observations = [os.path.abspath(image_path)]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"file://{observations[0]}"}},
            ],
        },
    ]

    trace: list[dict[str, Any]] = []

    for round_id in range(cfg.max_rounds):
        response = _chat_once(client, cfg, messages)
        trace.append({"round": round_id, "assistant": response})

        if _has_final_answer(response):
            return {
                "stopped_by": "final_answer",
                "rounds": round_id + 1,
                "trace": trace,
                "final_response": response,
            }

        grounding = _parse_grounding(response)
        if grounding is None:
            return {
                "stopped_by": "no_grounding",
                "rounds": round_id + 1,
                "trace": trace,
                "final_response": response,
            }

        bbox = grounding.get("bbox_2d")
        source = grounding.get("source", "original_image")
        if source == "original_image":
            src_idx = 0
        else:
            m = re.match(r"observation_(\d+)", source)
            if not m:
                raise ValueError(f"Invalid source: {source}")
            src_idx = int(m.group(1))
            if src_idx >= len(observations):
                raise ValueError(f"source {source} out of range. known: 0..{len(observations)-1}")

        src_image = observations[src_idx]
        out_path = os.path.join(cfg.work_dir, f"observation_{len(observations)}.jpg")
        cropped_path = _native_crop_relative(src_image, bbox, out_path)
        observations.append(os.path.abspath(cropped_path))

        messages.append({"role": "assistant", "content": response})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"After Action {round_id}, here is Observation {len(observations)-1}. "
                            "Continue and give <answer> when ready."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"file://{observations[-1]}"}},
                ],
            }
        )
        trace[-1]["grounding"] = grounding
        trace[-1]["crop_backend"] = "native"
        trace[-1]["observation"] = observations[-1]

    return {
        "stopped_by": "max_rounds",
        "rounds": cfg.max_rounds,
        "trace": trace,
        "final_response": trace[-1]["assistant"] if trace else "",
    }


if __name__ == "__main__":
    demo_question = "How much does a bottle of mineral water on the far right cost?"
    demo_image = "assets/visual_probe_medium_62.jpg"
    answer = "3 yuan and 80 cents"
    result = ask_once(demo_question, demo_image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
