import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.services.llm_errors import LLMCallError

load_dotenv()


def safe_json_loads(text: str) -> Dict[str, Any]:
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


def _create_llm() -> ChatOpenAI:
    if os.getenv("USE_LLM", "true").lower() != "true":
        raise LLMCallError()

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    model = (
        os.getenv("GAP_PATH_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or os.getenv("DASHSCOPE_MODEL")
    )
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )

    if not api_key or not model:
        raise LLMCallError()

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.25,
        "timeout": 60,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _validate_gap_paths(parsed: Dict[str, Any], expected_count: int) -> List[Dict[str, Any]]:
    paths = parsed.get("top5_gap_paths")
    if not isinstance(paths, list) or len(paths) != expected_count:
        raise LLMCallError()

    required_fields = {"job_name", "gap_list", "recommended_projects", "learning_stages"}
    normalized: List[Dict[str, Any]] = []
    for item in paths:
        if not isinstance(item, dict) or not required_fields.issubset(item.keys()):
            raise LLMCallError()
        if not isinstance(item.get("learning_stages"), list) or len(item["learning_stages"]) != 3:
            raise LLMCallError()
        normalized.append(item)

    return normalized


def generate_top5_gap_paths(
    student_data: Dict[str, Any],
    job_recommendations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    为 TOP5 岗位生成岗位差距、推荐项目和三阶段补齐路径。
    大模型不可用、异常或返回结构不完整时，直接抛出“调用LLM失败”。
    """
    top_jobs = job_recommendations[:5]
    if not top_jobs:
        return {
            "top5_gap_paths": [],
            "used_llm": True,
            "agent_warning": ""
        }

    prompt = f"""
你是大学生就业能力诊断系统中的岗位成长路径规划智能体。

下面是学生信息：
{json.dumps(student_data, ensure_ascii=False)}

下面是系统已经计算出的 TOP5 岗位推荐结果：
{json.dumps(top_jobs, ensure_ascii=False)}

请你必须为 TOP5 中的每一个岗位都生成个性化成长路径，不能只生成第一个岗位。
返回严格 JSON，不要 Markdown，不要解释文字。

字段要求如下：
{{
  "top5_gap_paths": [
    {{
      "job_name": "岗位名称，必须和输入中的 job_name 完全一致",
      "gap_list": ["该岗位当前最主要的差距1", "差距2", "差距3"],
      "recommended_projects": ["推荐项目1", "推荐项目2"],
      "learning_stages": [
        {{
          "stage": "第一阶段：基础补强",
          "duration": "第1-2个月",
          "goal": "阶段目标",
          "actions": ["行动任务1", "行动任务2", "行动任务3"],
          "deliverables": ["验收成果1", "验收成果2"]
        }},
        {{
          "stage": "第二阶段：项目实践",
          "duration": "第3-4个月",
          "goal": "阶段目标",
          "actions": ["行动任务1", "行动任务2", "行动任务3"],
          "deliverables": ["验收成果1", "验收成果2"]
        }},
        {{
          "stage": "第三阶段：就业准备",
          "duration": "第5-6个月",
          "goal": "阶段目标",
          "actions": ["行动任务1", "行动任务2", "行动任务3"],
          "deliverables": ["验收成果1", "验收成果2"]
        }}
      ]
    }}
  ]
}}

要求：
1. top5_gap_paths 的长度必须等于输入岗位数量，最多 5 个。
2. 每个岗位都要有自己的 gap_list、recommended_projects、learning_stages。
3. gap_list 要结合该岗位的 skill_gaps 和学生已有技能。
4. recommended_projects 要贴合岗位方向。
5. learning_stages 必须是三个阶段。
"""

    try:
        response = _create_llm().invoke(prompt)
        parsed = safe_json_loads(str(response.content))
        return {
            "top5_gap_paths": _validate_gap_paths(parsed, len(top_jobs)),
            "used_llm": True,
            "agent_warning": ""
        }
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()
