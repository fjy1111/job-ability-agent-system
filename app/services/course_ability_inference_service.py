from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

from app.services.llm_errors import LLMCallError
from app.services.model_config_service import create_configured_chat_model


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

COURSE_ABILITY_PROMPT_VERSION = "course_ability_infer_v1"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        raise LLMCallError()

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise LLMCallError()
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise LLMCallError()

    if not isinstance(data, dict):
        raise LLMCallError()
    return data


def _resolve_model_name() -> str:
    return (
        os.getenv("COURSE_ABILITY_MODEL", "").strip()
        or os.getenv("COURSE_MAPPING_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or os.getenv("DASHSCOPE_MODEL", "").strip()
    )


def _create_llm() -> Any:
    if ChatOpenAI is None:
        raise LLMCallError()
    return create_configured_chat_model(
        temperature=0.2,
        timeout=45,
        max_retries=0,
        task_name="COURSE_ABILITY",
        legacy_task_model_envs=("COURSE_ABILITY_MODEL", "COURSE_MAPPING_MODEL"),
    )


def _clean_abilities(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return _clean_abilities(loaded)
        except Exception:
            items = re.split(r"[,，、;；/\n\r]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = []

    seen: set[str] = set()
    abilities: list[str] = []
    for item in items:
        text = _safe_text(item).strip(" []【】()（）\"'")
        if not text or len(text) > 40:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        abilities.append(text)
        if len(abilities) >= 8:
            break
    return abilities


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.65
    if number > 1:
        number = number / 100
    return round(max(0.05, min(0.95, number)), 2)


def _validate_result(
    data: dict[str, Any],
    course_name: str,
    raw_response: str,
) -> dict[str, Any]:
    abilities = _clean_abilities(data.get("abilities"))
    if not abilities:
        raise LLMCallError()

    reason = _safe_text(data.get("reason"))
    if not reason:
        reason = "模型根据课程名称推理该课程可能培养的专业能力标签。"

    return {
        "course_name": _safe_text(data.get("course_name")) or course_name,
        "abilities": abilities,
        "confidence": _confidence(data.get("confidence", 0.65)),
        "reason": reason[:600],
        "source_label": "AI推理",
        "source_type": "llm_inference",
        "prompt_version": COURSE_ABILITY_PROMPT_VERSION,
        "model_name": _resolve_model_name(),
        "raw_response": raw_response,
    }


def infer_course_abilities_with_llm(
    course_name: str,
    resume_text: str = "",
) -> dict[str, Any]:
    course_name = _safe_text(course_name)
    if not course_name:
        raise LLMCallError()

    prompt = f"""
你是高校课程能力标签抽取专家。现在系统已经在本地课程知识库中找不到这门课，
请只根据课程名称和简历上下文，保守推理它可能培养的 IT、软件、数据、AI、网络安全或工程实践能力标签。

要求：
1. 不联网，不编造学校培养计划、成绩、项目经历或招聘岗位。
2. 只输出能力标签，不输出岗位名称。
3. 能力标签要能用于和招聘岗位 required_skills、recommended_courses 做文本匹配。
4. 最多 8 个能力标签，优先使用行业常见短语，例如 Python、SQL、数据建模、接口开发、机器学习、Linux。
5. confidence 使用 0 到 1 的小数；未知性越高，分数越保守。
6. 只返回严格 JSON，不要 Markdown。

课程名称：{course_name}

简历上下文片段：
{resume_text[:1800]}

JSON 格式：
{{
  "course_name": "{course_name}",
  "abilities": ["能力1", "能力2", "能力3"],
  "confidence": 0.65,
  "reason": "一句话说明推理依据"
}}
"""

    try:
        response = _create_llm().invoke(prompt)
        raw_response = _safe_text(response.content)
        return _validate_result(_safe_json_loads(raw_response), course_name, raw_response)
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()
