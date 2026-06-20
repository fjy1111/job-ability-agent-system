import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.services.llm_errors import LLMCallError
from app.services.model_config_service import create_configured_chat_model

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
    return create_configured_chat_model(
        temperature=0.25,
        timeout=int(os.getenv("JOB_MATCH_LLM_TIMEOUT_SECONDS", "30")),
        max_retries=0,
        task_name="GAP_PATH",
        legacy_task_model_envs=("GAP_PATH_MODEL",),
    )


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


def _build_local_gap_path(job: Dict[str, Any]) -> Dict[str, Any]:
    job_name = str(job.get("job_name") or "目标岗位")
    missing = job.get("skill_gaps") or job.get("missing_skills") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    missing = [str(item).strip() for item in missing if str(item).strip()][:3]
    if not missing:
        missing = ["补充岗位核心项目证据", "完善工程化实践", "加强面试表达与复盘"]

    primary = missing[0]
    secondary = missing[1] if len(missing) > 1 else "工程化能力"
    return {
        "job_name": job_name,
        "path_summary": f"围绕{job_name}的核心缺口，先补基础，再做项目，最后完成求职准备。",
        "gap_list": missing,
        "recommended_projects": [
            f"完成一个覆盖{primary}的{job_name}方向项目",
            f"为项目补充{secondary}实践、部署说明和技术复盘",
        ],
        "learning_stages": [
            {
                "stage": "第一阶段：基础补强",
                "duration": "第1-2个月",
                "goal": f"掌握{primary}并形成可验证练习",
                "actions": [f"系统学习{primary}", "完成配套练习", "整理知识笔记和常见面试题"],
                "deliverables": ["基础练习仓库", "知识与面试题清单"],
            },
            {
                "stage": "第二阶段：项目实践",
                "duration": "第3-4个月",
                "goal": f"形成与{job_name}要求对应的项目证据",
                "actions": ["设计项目功能和技术方案", "完成核心模块与测试", "部署并记录问题解决过程"],
                "deliverables": ["可运行项目", "README、测试和部署文档"],
            },
            {
                "stage": "第三阶段：就业准备",
                "duration": "第5-6个月",
                "goal": "把能力证据转化为简历和面试表达",
                "actions": ["量化项目成果并更新简历", "进行岗位题库训练", "完成模拟面试和复盘"],
                "deliverables": ["岗位定制简历", "项目讲解稿和面试复盘"],
            },
        ],
    }


def build_local_gap_paths(job_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "top5_gap_paths": [
            _build_local_gap_path(job)
            for job in job_recommendations[:5]
        ],
        "used_llm": False,
        "agent_warning": "",
    }


def generate_top5_gap_paths(
    student_data: Dict[str, Any],
    job_recommendations: List[Dict[str, Any]],
    use_llm: bool | None = None,
) -> Dict[str, Any]:
    """
    为 TOP5 岗位生成岗位差距、推荐项目和三阶段补齐路径。
    大模型不可用、异常或返回结构不完整时，直接抛出“调用LLM失败”。
    """
    top_jobs = job_recommendations[:5]
    if not top_jobs:
        return {
            "top5_gap_paths": [],
            "used_llm": False,
            "agent_warning": ""
        }

    if use_llm is None:
        use_llm = os.getenv("JOB_MATCH_GAP_PATH_USE_LLM", "false").lower() == "true"
    if not use_llm:
        return build_local_gap_paths(top_jobs)

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
    except Exception:
        fallback = build_local_gap_paths(top_jobs)
        fallback["agent_warning"] = "AI 路径生成超时或失败，已返回本地成长路径。"
        return fallback
