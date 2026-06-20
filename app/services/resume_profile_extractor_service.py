from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.services.llm_errors import LLMCallError
from app.services.model_config_service import create_configured_chat_model

load_dotenv()


PROFILE_FIELDS = (
    "name",
    "major",
    "grade",
    "target_job",
    "skills",
    "projects",
    "competitions",
    "certificates",
    "self_intro",
)

MAX_LENGTHS = {
    "name": 50,
    "major": 100,
    "grade": 30,
    "target_job": 100,
}

MISSING_VALUE = "无"


def _safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        raise LLMCallError()

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

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


def _create_llm() -> ChatOpenAI:
    return create_configured_chat_model(
        temperature=0,
        timeout=60,
        max_retries=0,
        task_name="RESUME_PROFILE",
        legacy_task_model_envs=("RESUME_PROFILE_MODEL",),
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_stringify(item) for item in value if _stringify(item).strip())
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {_stringify(item)}"
            for key, item in value.items()
            if _stringify(item).strip()
        )
    return str(value)


def _normalize_field(key: str, value: Any) -> str:
    text = _stringify(value).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text or text.lower() in {"none", "null", "n/a", "na"}:
        return MISSING_VALUE
    if text in {"未填写", "未知", "无对应字段", "未提供", "暂无"}:
        return MISSING_VALUE

    max_length = MAX_LENGTHS.get(key)
    if max_length:
        return text[:max_length]
    return text


def _validate_profile(data: dict[str, Any]) -> dict[str, str]:
    if any(field not in data for field in PROFILE_FIELDS):
        raise LLMCallError()

    profile = {
        field: _normalize_field(field, data.get(field))
        for field in PROFILE_FIELDS
    }

    unexpected_fields = set(data.keys()) - set(PROFILE_FIELDS)
    if unexpected_fields:
        raise LLMCallError()

    return profile


def extract_student_profile_from_resume(message: str, resume_text: str) -> dict[str, str]:
    """
    调用大模型把简历文本映射到数据库字段。失败时只抛出“调用LLM失败”。
    """
    if not resume_text.strip():
        raise LLMCallError()

    prompt = f"""
你是简历信息抽取智能体。请把用户上传的简历文本严格映射到数据库字段。

数据库字段只有这 9 个：
1. name：姓名，只填真实姓名。
2. major：专业。
3. grade：年级或学历阶段，例如“大三”“研一”“本科”“硕士”。
4. target_job：目标岗位、求职意向或应聘岗位。
5. skills：技能、工具、编程语言、框架、软件能力，只放技能类信息。
6. projects：项目经历、实习经历、工作经历、科研经历，只放经历类信息。
7. competitions：竞赛、奖项、获奖经历。
8. certificates：证书、语言等级、资格认证。
9. self_intro：个人简介、自我评价、职业优势总结。

规则：
- 必须返回严格 JSON，不要 Markdown，不要解释。
- JSON 只能包含上面 9 个键，不能新增字段。
- 如果某个字段在简历中没有明确对应信息，填“无”。
- 简历中有信息但数据库没有对应字段时忽略，不要塞进其他字段。
- 不要把整份简历原文放进 skills、projects 或 self_intro。
- 不要把用户指令“帮我生成画像/帮我优化简历”等当成简历内容。
- 不要编造简历中没有出现的信息。

用户指令：
{message[:1000]}

简历文本：
{resume_text[:8000]}

返回 JSON 格式：
{{
  "name": "无",
  "major": "无",
  "grade": "无",
  "target_job": "无",
  "skills": "无",
  "projects": "无",
  "competitions": "无",
  "certificates": "无",
  "self_intro": "无"
}}
"""

    try:
        response = _create_llm().invoke(prompt)
        return _validate_profile(_safe_json_loads(str(response.content)))
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()
