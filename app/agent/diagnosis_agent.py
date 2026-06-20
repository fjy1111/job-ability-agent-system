from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from app.services.llm_errors import LLMCallError
from app.services.model_config_service import create_configured_chat_model


# =========================================================
# 读取环境变量
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


GENERAL_SKILLS = [
    "Python", "Java", "C++", "HTML", "CSS", "JavaScript",
    "Vue", "React", "TypeScript",
    "FastAPI", "Django", "Flask", "Spring Boot",
    "MySQL", "Redis", "数据库", "SQL",
    "Git", "Linux", "Docker", "云平台",
    "PyTorch", "TensorFlow", "OpenCV",
    "机器学习", "深度学习", "算法", "数据结构",
    "数据分析", "Pandas", "NumPy", "可视化", "统计学",
    "大模型", "LangChain", "LangGraph",
    "接口测试", "自动化测试",
    "需求分析", "原型设计", "项目管理", "沟通表达"
]


# =========================================================
# LangGraph 状态定义
# =========================================================

class DiagnosisState(TypedDict, total=False):
    """
    整个智能体执行过程中的共享状态。
    每一个节点都可以读取前面节点的结果，并添加自己的输出。
    """

    student: dict[str, str]

    normalized_text: str
    recognized_skills: list[str]

    ability_scores: dict[str, int]
    score_evidence: dict[str, list[str]]

    profile_tags: list[str]
    risk_flags: list[str]
    evidence_cards: list[dict[str, Any]]
    dimension_insights: list[dict[str, Any]]
    development_focus: list[dict[str, Any]]
    quality_review: list[str]
    workflow_steps: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    collaboration_log: list[dict[str, Any]]
    review_findings: list[dict[str, Any]]
    shared_workspace: dict[str, Any]
    collaborative_draft: dict[str, Any]
    llm_agents: list[str]

    summary: str
    advantages: list[str]
    weaknesses: list[str]

    used_llm: bool
    agent_warning: str


ABILITY_DIMENSIONS = {
    "professional": {
        "name": "专业基础能力",
        "agent": "专业基础评估智能体",
        "description": "关注课程知识、专业概念、证书与知识体系完整度。",
        "next_action": "补齐核心课程知识图谱，形成可复述的知识笔记和题目练习记录。"
    },
    "practice": {
        "name": "技术实践能力",
        "agent": "项目实践评估智能体",
        "description": "关注项目经历、竞赛经历、实习经历和可交付作品。",
        "next_action": "选择一个真实问题做成可演示项目，并沉淀代码仓库、README 和复盘文档。"
    },
    "tools": {
        "name": "工具技能能力",
        "agent": "工具栈识别智能体",
        "description": "关注编程语言、框架、数据库、工程工具和 AI 工具掌握情况。",
        "next_action": "围绕当前常用技术栈补齐工程工具链，完成一次从开发到部署的闭环练习。"
    },
    "career": {
        "name": "职业发展能力",
        "agent": "职业准备评估智能体",
        "description": "关注目标清晰度、表达能力、简历素材和面试准备。",
        "next_action": "把经历改写成 STAR 素材，准备一版能力证据型简历和自我介绍。"
    }
}


AGENT_ROSTER = [
    "画像采集智能体",
    "四维评分智能体",
    "证据抽取智能体",
    "能力归因智能体",
    "质量复核智能体"
]


# =========================================================
# 通用工具函数
# =========================================================

def _safe_text(value: str | None) -> str:
    return (value or "").strip()


MISSING_PROFILE_VALUES = {"", "无", "未填写", "未知", "暂无", "未提供"}


def _has_profile_value(value: Any) -> bool:
    return _safe_text(value).strip() not in MISSING_PROFILE_VALUES


def _clamp_score(score: int) -> int:
    return max(0, min(100, score))


def _contains(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def _score_level(score: int) -> str:
    if score >= 85:
        return "优势突出"
    if score >= 70:
        return "稳定具备"
    if score >= 55:
        return "已有基础"
    if score >= 40:
        return "需要补强"
    return "信息不足"


def _append_workflow_step(
    state: DiagnosisState,
    *,
    step: str,
    agent: str,
    task: str,
    status: str,
    output: str,
    llm_role: str = "规则智能体"
) -> list[dict[str, Any]]:
    steps = list(state.get("workflow_steps", []))
    steps.append({
        "step": step,
        "agent": agent,
        "llm_role": llm_role,
        "task": task,
        "status": status,
        "output": output
    })
    return steps


def _summarize_tool_output(output: Any) -> str:
    if isinstance(output, dict):
        parts = []
        for key, value in output.items():
            if isinstance(value, list):
                parts.append(f"{key}={len(value)}项")
            elif isinstance(value, dict):
                parts.append(f"{key}={len(value)}类")
            elif isinstance(value, str) and len(value) > 32:
                parts.append(f"{key}={value[:32]}...")
            else:
                parts.append(f"{key}={value}")
        return "，".join(parts[:4])

    if isinstance(output, list):
        return f"返回 {len(output)} 项结果"

    return str(output)[:120]


def _append_tool_call(
    state: DiagnosisState,
    *,
    called_by: str,
    tool_name: str,
    purpose: str,
    input_summary: str,
    output: Any
) -> list[dict[str, Any]]:
    calls = list(state.get("tool_calls", []))
    calls.append({
        "tool_name": tool_name,
        "called_by": called_by,
        "purpose": purpose,
        "input_summary": input_summary,
        "output_summary": _summarize_tool_output(output),
        "status": "completed"
    })
    return calls


def _append_collaboration_log(
    state: DiagnosisState,
    *,
    sender: str,
    receiver: str,
    message: str,
    artifact: str
) -> list[dict[str, Any]]:
    logs = list(state.get("collaboration_log", []))
    logs.append({
        "sender": sender,
        "receiver": receiver,
        "message": message,
        "artifact": artifact
    })
    return logs


def _merge_workspace(
    state: DiagnosisState,
    **updates: Any
) -> dict[str, Any]:
    workspace = dict(state.get("shared_workspace", {}))
    workspace.update(updates)
    return workspace


def _append_warning(state: DiagnosisState, warning: str) -> str:
    current = state.get("agent_warning", "")

    if not warning:
        return current

    if current:
        return f"{current}；{warning}"

    return warning


def _append_llm_agent(state: DiagnosisState, agent: str, used_llm: bool) -> list[str]:
    agents = list(state.get("llm_agents", []))

    if used_llm and agent not in agents:
        agents.append(agent)

    return agents


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _invoke_json_agent(agent_name: str, prompt: str) -> tuple[dict[str, Any] | None, bool, str]:
    """
    用同一个 OpenAI 兼容模型扮演不同专家智能体。
    任一专家失败时直接终止工作流。
    """

    try:
        llm = _create_llm()

        if llm is None:
            raise LLMCallError()

        response = llm.invoke(
            prompt,
            response_format={"type": "json_object"}
        )
        content = response.content

        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        parsed = extract_json_from_llm_text(str(content))
        if not parsed:
            raise LLMCallError()
        return parsed, True, ""

    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


def profile_text_normalizer_tool(student: dict[str, str]) -> dict[str, Any]:
    fields = {
        "major": _safe_text(student.get("major")),
        "target_job": _safe_text(student.get("target_job")),
        "skills": _safe_text(student.get("skills")),
        "projects": _safe_text(student.get("projects")),
        "competitions": _safe_text(student.get("competitions")),
        "certificates": _safe_text(student.get("certificates")),
        "self_intro": _safe_text(student.get("self_intro"))
    }
    resume_text = _safe_text(student.get("resume_text") or student.get("normalized_text"))
    filled_fields = [
        key for key, value in fields.items()
        if _has_profile_value(value)
    ]
    missing_fields = [
        key for key, value in fields.items()
        if not _has_profile_value(value)
    ]
    raw_parts = [
        f"{key}: {value}"
        for key, value in fields.items()
        if _has_profile_value(value)
    ]
    if resume_text.strip():
        raw_parts.append(f"resume_text: {resume_text}")
    raw_text = "\n".join(raw_parts)

    return {
        "normalized_text": raw_text,
        "filled_fields": filled_fields,
        "missing_fields": missing_fields,
        "text_length": len(raw_text)
    }


def skill_keyword_scanner_tool(text: str) -> dict[str, Any]:
    recognized_skills = [
        skill for skill in GENERAL_SKILLS
        if _contains(text, skill)
    ]
    recognized_skills = list(dict.fromkeys(recognized_skills))

    skill_groups = {
        "programming": [
            skill for skill in recognized_skills
            if skill in {"Python", "Java", "C++", "HTML", "CSS", "JavaScript", "TypeScript"}
        ],
        "frameworks": [
            skill for skill in recognized_skills
            if skill in {"Vue", "React", "FastAPI", "Django", "Flask", "Spring Boot", "LangChain", "LangGraph"}
        ],
        "data_ai": [
            skill for skill in recognized_skills
            if skill in {"PyTorch", "TensorFlow", "OpenCV", "机器学习", "深度学习", "算法", "数据结构", "数据分析", "Pandas", "NumPy", "可视化", "统计学", "大模型"}
        ],
        "engineering": [
            skill for skill in recognized_skills
            if skill in {"MySQL", "Redis", "数据库", "SQL", "Git", "Linux", "Docker", "云平台"}
        ],
        "product_quality": [
            skill for skill in recognized_skills
            if skill in {"接口测试", "自动化测试", "需求分析", "原型设计", "项目管理", "沟通表达"}
        ]
    }

    return {
        "recognized_skills": recognized_skills,
        "skill_groups": {key: value for key, value in skill_groups.items() if value},
        "coverage_count": len(recognized_skills)
    }


def rubric_score_tool(student: dict[str, str], text: str, skills: list[str]) -> dict[str, Any]:
    professional_keywords = [
        "数据结构", "算法", "数据库", "计算机网络",
        "操作系统", "机器学习", "深度学习", "统计学"
    ]

    tool_keywords = [
        "Python", "Java", "C++", "FastAPI", "Django",
        "Flask", "Spring Boot", "MySQL", "Redis",
        "Git", "Linux", "Docker", "PyTorch",
        "Vue", "React", "LangGraph"
    ]

    professional_hits = [
        item for item in professional_keywords
        if _contains(text, item)
    ]

    tool_hits = [
        item for item in tool_keywords
        if _contains(text, item)
    ]

    has_project = _has_profile_value(student.get("projects"))
    has_competition = _has_profile_value(student.get("competitions"))
    has_certificate = _has_profile_value(student.get("certificates"))
    has_target_job = _has_profile_value(student.get("target_job"))
    has_intro = (
        _has_profile_value(student.get("self_intro"))
        and len(_safe_text(student.get("self_intro"))) >= 30
    )

    professional_score = 45 + len(professional_hits) * 6
    if has_certificate:
        professional_score += 8

    practice_score = 35
    if has_project:
        practice_score += 25
    if has_competition:
        practice_score += 18
    if "实习" in text:
        practice_score += 12
    if any(word in text for word in ["开发", "设计", "完成", "实现", "负责"]):
        practice_score += 6

    tools_score = 35 + len(tool_hits) * 5
    if len(skills) >= 6:
        tools_score += 5

    career_score = 35
    if has_target_job:
        career_score += 25
    if has_intro:
        career_score += 12
    if has_certificate:
        career_score += 8
    if has_competition or has_project:
        career_score += 8

    scores = {
        "professional": _clamp_score(professional_score),
        "practice": _clamp_score(practice_score),
        "tools": _clamp_score(tools_score),
        "career": _clamp_score(career_score)
    }

    evidence = {
        "professional": professional_hits or ["暂未识别到明确的专业课程或专业知识证据"],
        "practice": [
            item for item, flag in [
                ("存在项目经历", has_project),
                ("存在竞赛经历", has_competition),
                ("文本中提及实习经历", "实习" in text)
            ]
            if flag
        ] or ["暂未填写项目、竞赛或实习经历"],
        "tools": tool_hits or ["暂未识别到明确工具技能"],
        "career": [
            item for item, flag in [
                ("已填写目标岗位", has_target_job),
                ("具有较完整自我介绍", has_intro),
                ("已填写相关证书", has_certificate)
            ]
            if flag
        ] or ["目标岗位和职业准备信息较少"]
    }

    audit_notes = []
    if scores["practice"] >= 70 and not has_project:
        audit_notes.append("实践分较高但缺少项目经历，需要复核。")
    if scores["tools"] >= 70 and len(tool_hits) < 5:
        audit_notes.append("工具分较高但工具命中数量不足，需要复核。")
    if not audit_notes:
        audit_notes.append("评分与当前证据基本一致。")

    return {
        "ability_scores": scores,
        "score_evidence": evidence,
        "rubric_trace": {
            "professional_hits": professional_hits,
            "tool_hits": tool_hits,
            "has_project": has_project,
            "has_competition": has_competition,
            "has_certificate": has_certificate,
            "has_target_job": has_target_job,
            "has_intro": has_intro
        },
        "audit_notes": audit_notes
    }


def evidence_matrix_tool(
    scores: dict[str, int],
    score_evidence: dict[str, list[str]],
    student: dict[str, str]
) -> dict[str, Any]:
    matrix = []
    weak_dimensions = []

    for key, meta in ABILITY_DIMENSIONS.items():
        score = scores.get(key, 0)
        evidence = score_evidence.get(key, [])
        useful_evidence = [
            item for item in evidence
            if item and not item.startswith("暂未") and "较少" not in item
        ]
        confidence = _evidence_confidence(score, evidence)

        if score < 55 or confidence == "低":
            weak_dimensions.append(meta["name"])

        matrix.append({
            "dimension": key,
            "name": meta["name"],
            "score": score,
            "evidence_count": len(useful_evidence),
            "confidence": confidence,
            "evidence": evidence,
            "fact_boundary": "仅引用学生表单和规则识别结果"
        })

    missing_materials = []
    if not _safe_text(student.get("projects")):
        missing_materials.append("项目经历")
    if not _safe_text(student.get("competitions")):
        missing_materials.append("竞赛经历")
    if not _safe_text(student.get("certificates")):
        missing_materials.append("证书材料")
    if not _safe_text(student.get("self_intro")):
        missing_materials.append("自我介绍")

    return {
        "matrix": matrix,
        "weak_dimensions": weak_dimensions,
        "missing_materials": missing_materials
    }


def consistency_audit_tool(state: DiagnosisState) -> dict[str, Any]:
    scores = state.get("ability_scores", {})
    cards = state.get("evidence_cards", [])
    findings = []

    for card in cards:
        key = card.get("dimension")
        score = scores.get(key, 0)
        confidence = card.get("confidence", "")

        if score >= 75 and confidence == "低":
            findings.append({
                "severity": "high",
                "dimension": card.get("name", key),
                "finding": "高分维度缺少高可信证据，需要下调表达强度。"
            })
        elif score < 55 and confidence in {"高", "中"}:
            findings.append({
                "severity": "medium",
                "dimension": card.get("name", key),
                "finding": "低分维度已有部分证据，建议表达为待发展而非完全缺失。"
            })

    if not findings:
        findings.append({
            "severity": "low",
            "dimension": "整体画像",
            "finding": "分数、证据和结论之间未发现明显冲突。"
        })

    return {
        "findings": findings,
        "decision": "通过复核" if all(item["severity"] == "low" for item in findings) else "需带着限制条件生成结论"
    }


def extract_json_from_llm_text(text: str) -> dict:
    """
    从大模型返回文本中提取 JSON。
    兼容 ```json ... ``` 包裹的情况。
    """

    text = text.strip()

    # 去掉 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 如果前后有解释文字，截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        return json.loads(json_text)

    raise ValueError("大模型返回内容不是合法 JSON")


def _create_llm() -> ChatOpenAI | None:
    return create_configured_chat_model(
        temperature=0.2,
        timeout=60,
        max_retries=2,
        task_name="DIAGNOSIS",
    )


# =========================================================
# 节点一：提取学生画像
# =========================================================

def extract_profile_node(state: DiagnosisState) -> dict[str, Any]:
    """
    将学生原始表单拼接成可分析文本，并识别明确出现过的技能。
    """

    student = state["student"]
    normalize_result = profile_text_normalizer_tool(student)
    raw_text = normalize_result["normalized_text"]
    skill_scan_result = skill_keyword_scanner_tool(raw_text)
    recognized_skills = skill_scan_result["recognized_skills"]
    tool_calls = _append_tool_call(
        state,
        called_by="画像采集智能体",
        tool_name="ProfileTextNormalizer",
        purpose="清洗学生表单并生成共享分析文本",
        input_summary="学生基础信息、技能、项目、竞赛、证书、自我介绍",
        output=normalize_result
    )
    tool_calls = _append_tool_call(
        {"tool_calls": tool_calls},
        called_by="画像采集智能体",
        tool_name="SkillKeywordScanner",
        purpose="从共享文本中识别显性技能和技能组",
        input_summary=f"normalized_text length={normalize_result['text_length']}",
        output=skill_scan_result
    )

    return {
        "normalized_text": raw_text,
        "recognized_skills": recognized_skills,
        "tool_calls": tool_calls,
        "shared_workspace": _merge_workspace(
            state,
            input_profile=normalize_result,
            skill_scan=skill_scan_result
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="画像采集智能体",
            receiver="四维评分智能体",
            message="已完成输入标准化和技能扫描，请基于同一份共享文本进行评分，不要自行补写经历。",
            artifact="normalized_text + skill_scan"
        ),
        "workflow_steps": _append_workflow_step(
            state,
            step="01",
            agent="画像采集智能体",
            task="调用文本标准化和技能扫描工具，生成后续 LLM 共用的任务上下文",
            status="completed",
            output=(
                f"调用 2 个工具，识别到 {len(recognized_skills)} 个显性技能，"
                f"缺失字段 {len(normalize_result['missing_fields'])} 项。"
            )
        )
    }


# =========================================================
# 节点二：能力画像评分
# =========================================================

def _normalize_score_analysis(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        raise LLMCallError()

    raw_scores = raw.get("ability_scores")
    raw_evidence = raw.get("score_evidence")
    if not isinstance(raw_scores, dict) or not isinstance(raw_evidence, dict):
        raise LLMCallError()

    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}

    for key in ABILITY_DIMENSIONS:
        try:
            scores[key] = _clamp_score(int(raw_scores[key]))
        except (KeyError, TypeError, ValueError):
            raise LLMCallError()

        evidence[key] = [
            _safe_text(item)
            for item in _safe_list(raw_evidence.get(key))
            if _safe_text(item)
        ]
        if not evidence[key]:
            raise LLMCallError()

    return {
        "ability_scores": scores,
        "score_evidence": evidence,
        "recognized_skills": [
            _safe_text(item)
            for item in _safe_list(raw.get("recognized_skills"))
            if _safe_text(item)
        ],
        "assessment_summary": _safe_text(raw.get("assessment_summary"))
    }


def score_ability_node(state: DiagnosisState) -> dict[str, Any]:
    """
    只调用一次大模型生成后续专家共享的完整画像草案。
    """

    student = state["student"]
    text = state["normalized_text"]
    skills = state["recognized_skills"]
    rubric_result = rubric_score_tool(student, text, skills)
    prompt_student = {
        key: student.get(key, "")
        for key in ("name", "major", "grade", "target_job")
    }
    prompt = f"""
你是学生能力画像工作流中的“综合画像协作智能体”。你需要一次生成四个专家角色共用的精简语义草案：
1. 四维评分专家：输出四维分数和精简评分证据。
2. 证据抽取专家：识别关键技能。
3. 能力归因专家：输出短摘要、优势、短板和每个维度的下一步动作。
4. 质量复核专家：输出简短复核提示。

后续 LangGraph 节点会分别调用 EvidenceMatrixTool、ConsistencyAuditTool 和 FinalAuditTool
扩展并复核你的草案，因此不得编造经历、不得修改输入事实、不得输出岗位排名。

学生基础信息：
{json.dumps(prompt_student, ensure_ascii=False)}

简历规范化文本：
{text}

系统识别技能：
{json.dumps(skills, ensure_ascii=False)}

RubricScoreCalculator 评分参考：
{json.dumps(rubric_result, ensure_ascii=False)}

请只输出一个严格 JSON 对象，不要 Markdown，不要解释文字。所有文本必须精简，字段必须完整：
{{
  "ability_scores": {{
    "professional": 0,
    "practice": 0,
    "tools": 0,
    "career": 0
  }},
  "score_evidence": {{
    "professional": ["不超过30字的核心证据"],
    "practice": ["不超过30字的核心证据"],
    "tools": ["不超过30字的核心证据"],
    "career": ["不超过30字的核心证据"]
  }},
  "recognized_skills": ["最多10个技能"],
  "assessment_summary": "不超过80字的整体画像摘要",
  "advantages": ["最多3条，每条不超过25字"],
  "weaknesses": ["最多3条，每条不超过25字"],
  "dimension_actions": {{
    "professional": "不超过35字的下一步动作",
    "practice": "不超过35字的下一步动作",
    "tools": "不超过35字的下一步动作",
    "career": "不超过35字的下一步动作"
  }},
  "quality_notes": ["最多2条，每条不超过30字"]
}}
"""
    collaborative_draft, used_llm, warning = _invoke_json_agent(
        "综合画像协作智能体",
        prompt
    )
    score_result = _normalize_score_analysis(collaborative_draft)
    scores = score_result["ability_scores"]
    evidence = score_result["score_evidence"]
    tool_calls = _append_tool_call(
        state,
        called_by="四维评分智能体",
        tool_name="RubricScoreCalculator",
        purpose="生成可审计的评分参考，供 LLM 四维评分校验边界",
        input_summary=f"recognized_skills={len(skills)}，text_length={len(text)}",
        output=rubric_result
    )
    tool_calls = _append_tool_call(
        {"tool_calls": tool_calls},
        called_by="四维评分智能体",
        tool_name="LLMAbilityScorer",
        purpose="单次调用大模型生成四个专家共享的完整画像草案",
        input_summary="student profile + normalized_text + rubric reference",
        output={
            "ability_scores": scores,
            "score_evidence": evidence,
            "draft_sections": [
                "ability_scores",
                "score_evidence",
                "semantic_summary",
                "dimension_actions"
            ]
        }
    )

    return {
        "ability_scores": scores,
        "score_evidence": evidence,
        "recognized_skills": score_result["recognized_skills"] or skills,
        "assessment_summary": score_result["assessment_summary"],
        "collaborative_draft": collaborative_draft,
        "tool_calls": tool_calls,
        "shared_workspace": _merge_workspace(
            state,
            score_sheet=rubric_result,
            llm_collaboration={
                "mode": "single_request",
                "agent": "综合画像协作智能体",
                "sections": [
                    "ability_scores",
                    "score_evidence",
                    "semantic_summary",
                    "dimension_actions"
                ]
            }
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="四维评分智能体",
            receiver="证据抽取智能体",
            message="单次 LLM 协作已生成共享画像草案，请调用证据矩阵工具复核证据边界，不允许自行补写经历。",
            artifact="ability_scores + score_evidence + collaborative_draft"
        ),
        "used_llm": used_llm,
        "llm_agents": _append_llm_agent(
            state,
            "综合画像协作智能体",
            used_llm
        ),
        "agent_warning": _append_warning(state, warning),
        "workflow_steps": _append_workflow_step(
            state,
            step="02",
            agent="四维评分智能体",
            llm_role="LLM 协作发起者",
            task="调用评分规程工具审计输入，并以一次 LLM 请求生成四个专家共享草案",
            status="llm_completed",
            output=(
                "完成四维评分："
                f"专业 {scores['professional']}，实践 {scores['practice']}，"
                f"工具 {scores['tools']}，职业 {scores['career']}；"
                f"审计提示 {len(rubric_result['audit_notes'])} 条。"
            )
        )
    }


# =========================================================
# 节点三：证据抽取智能体
# =========================================================

def _evidence_confidence(score: int, evidence: list[str]) -> str:
    useful_evidence = [
        item for item in evidence
        if item and not item.startswith("暂未") and "较少" not in item
    ]

    if score >= 75 and len(useful_evidence) >= 2:
        return "高"
    if score >= 55 or useful_evidence:
        return "中"
    return "低"


def _build_evidence_analysis(
    state: DiagnosisState,
    matrix_result: dict[str, Any]
) -> dict[str, Any]:
    confidence_by_key = {
        item.get("dimension"): item.get("confidence", "中")
        for item in matrix_result.get("matrix", [])
        if isinstance(item, dict)
    }
    cards = []

    for key, meta in ABILITY_DIMENSIONS.items():
        score = state["ability_scores"].get(key, 0)
        evidence = state.get("score_evidence", {}).get(key, [])
        confidence = confidence_by_key.get(key, "中")
        cards.append({
            "dimension": key,
            "name": meta["name"],
            "agent": meta["agent"],
            "evidence": evidence,
            "confidence": confidence,
            "interpretation": (
                f"{meta['name']}得分为 {score} 分，"
                f"当前证据可信度为{confidence}，结论仅基于已提供简历材料。"
            )
        })

    profile_tags = [
        _safe_text(item)
        for item in state.get("recognized_skills", [])[:6]
        if _safe_text(item)
    ]
    if not profile_tags and _has_profile_value(state["student"].get("target_job")):
        profile_tags.append(_safe_text(state["student"].get("target_job")))

    risk_flags = [
        f"缺少{item}，相关判断可信度受限"
        for item in matrix_result.get("missing_materials", [])[:3]
    ]
    risk_flags.extend(
        f"{item}证据相对薄弱"
        for item in matrix_result.get("weak_dimensions", [])[:2]
    )
    if not risk_flags:
        risk_flags.append("当前画像仍需结合面试或作品材料进一步复核")

    return {
        "evidence_cards": cards,
        "profile_tags": profile_tags,
        "risk_flags": risk_flags
    }


def analyze_profile_evidence_node(state: DiagnosisState) -> dict[str, Any]:
    """
    证据抽取智能体调用本地工具复核共享草案中的证据卡。
    """

    matrix_result = evidence_matrix_tool(
        scores=state["ability_scores"],
        score_evidence=state.get("score_evidence", {}),
        student=state["student"]
    )

    analysis = _build_evidence_analysis(state, matrix_result)
    tool_calls = _append_tool_call(
        state,
        called_by="证据抽取智能体",
        tool_name="EvidenceMatrixTool",
        purpose="把评分证据转成可复核的四维证据矩阵",
        input_summary="ability_scores + score_evidence + student material completeness",
        output=matrix_result
    )

    return {
        **analysis,
        "tool_calls": tool_calls,
        "shared_workspace": _merge_workspace(
            state,
            evidence_matrix=matrix_result,
            evidence_cards=analysis["evidence_cards"]
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="证据抽取智能体",
            receiver="能力归因智能体",
            message="已用证据矩阵工具复核共享草案的证据边界，请继续执行一致性审计并校验画像结论。",
            artifact="evidence_matrix + evidence_cards + risk_flags"
        ),
        "used_llm": bool(state.get("used_llm", False)),
        "llm_agents": list(state.get("llm_agents", [])),
        "agent_warning": state.get("agent_warning", ""),
        "workflow_steps": _append_workflow_step(
            state,
            step="03",
            agent="证据抽取智能体",
            llm_role="共享草案校验专家",
            task="调用证据矩阵工具，复核单次 LLM 协作草案中的证据卡和事实边界",
            status="completed",
            output=(
                f"工具产出 {len(matrix_result['matrix'])} 个维度矩阵；"
                f"校验通过 {len(analysis['evidence_cards'])} 张证据卡、"
                f"{len(analysis['profile_tags'])} 个画像标签。"
            )
        )
    }


# =========================================================
# 节点四：能力归因智能体
# =========================================================

def _build_ability_report(state: DiagnosisState) -> dict[str, Any]:
    draft = state.get("collaborative_draft") or {}
    scores = state["ability_scores"]
    evidence = state.get("score_evidence", {})
    actions = draft.get("dimension_actions")
    if not isinstance(actions, dict):
        actions = {}

    advantages = [
        _safe_text(item)
        for item in _safe_list(draft.get("advantages"))[:3]
        if _safe_text(item)
    ]
    weaknesses = [
        _safe_text(item)
        for item in _safe_list(draft.get("weaknesses"))[:3]
        if _safe_text(item)
    ]
    ranked_dimensions = sorted(
        ABILITY_DIMENSIONS,
        key=lambda key: scores.get(key, 0),
        reverse=True
    )

    if not advantages:
        advantages = [
            f"{ABILITY_DIMENSIONS[key]['name']}相对突出"
            for key in ranked_dimensions[:2]
        ]
    if not weaknesses:
        weaknesses = [
            f"{ABILITY_DIMENSIONS[key]['name']}仍需补强"
            for key in ranked_dimensions[-2:]
        ]

    dimension_insights = []
    for key, meta in ABILITY_DIMENSIONS.items():
        score = scores.get(key, 0)
        dimension_insights.append({
            "key": key,
            "name": meta["name"],
            "score": score,
            "level": _score_level(score),
            "conclusion": (
                f"当前证据显示，{meta['name']}得分为 {score} 分，"
                "具体判断以已提供的简历事实为边界。"
            ),
            "evidence": evidence.get(key, []),
            "next_action": (
                _safe_text(actions.get(key))
                or meta["next_action"]
            )
        })

    focus_keys = ranked_dimensions[-2:]
    development_focus = [
        {
            "name": ABILITY_DIMENSIONS[key]["name"],
            "priority": "高" if scores.get(key, 0) < 60 else "中",
            "reason": f"当前得分为 {scores.get(key, 0)} 分，是相对需要优先提升的维度。",
            "action": (
                _safe_text(actions.get(key))
                or ABILITY_DIMENSIONS[key]["next_action"]
            )
        }
        for key in focus_keys
    ]

    return {
        "summary": (
            _safe_text(draft.get("assessment_summary"))
            or "系统已根据简历事实完成四维能力分析，请结合证据卡和风险提示理解结论。"
        ),
        "advantages": advantages,
        "weaknesses": weaknesses,
        "dimension_insights": dimension_insights,
        "development_focus": development_focus
    }


def diagnose_ability_node(state: DiagnosisState) -> dict[str, Any]:
    """
    能力归因智能体调用一致性工具复核共享草案中的画像结论。
    """

    audit_result = consistency_audit_tool(state)
    report = _build_ability_report(state)
    tool_calls = _append_tool_call(
        state,
        called_by="能力归因智能体",
        tool_name="ConsistencyAuditTool",
        purpose="检查分数、证据卡和画像结论之间是否存在冲突",
        input_summary="ability_scores + evidence_cards",
        output=audit_result
    )

    return {
        **report,
        "tool_calls": tool_calls,
        "review_findings": audit_result.get("findings", []),
        "shared_workspace": _merge_workspace(
            state,
            consistency_audit=audit_result,
            ability_report={
                "summary": report["summary"],
                "advantages": report["advantages"],
                "weaknesses": report["weaknesses"],
                "dimension_insights": report["dimension_insights"],
                "development_focus": report["development_focus"]
            }
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="能力归因智能体",
            receiver="质量复核智能体",
            message="共享草案中的画像结论已通过一致性审计，请执行最终工具链和事实边界复核。",
            artifact="ability_report + consistency_audit"
        ),
        "used_llm": bool(state.get("used_llm", False)),
        "llm_agents": list(state.get("llm_agents", [])),
        "agent_warning": state.get("agent_warning", ""),
        "workflow_steps": _append_workflow_step(
            state,
            step="04",
            agent="能力归因智能体",
            llm_role="共享草案校验专家",
            task="调用一致性审计工具，复核单次 LLM 协作草案中的分数、证据和结论",
            status="completed",
            output=(
                f"审计结论：{audit_result['decision']}；完成 {len(report['dimension_insights'])} 个维度洞察，"
                f"提炼 {len(report['development_focus'])} 个发展焦点。"
            )
        )
    }


# =========================================================
# 节点五：质量复核智能体
# =========================================================

def review_profile_node(state: DiagnosisState) -> dict[str, Any]:
    """
    质量复核智能体调用最终审计工具检查共享草案。
    """

    consistency_result = consistency_audit_tool(state)
    final_audit_result = {
        "consistency": consistency_result,
        "tool_call_count": len(state.get("tool_calls", [])),
        "handoff_count": len(state.get("collaboration_log", [])),
        "has_job_ranking": False,
        "decision": (
            "可发布"
            if state.get("tool_calls") and state.get("collaboration_log")
            else "可发布，但旧记录缺少完整工具链"
        )
    }

    collaborative_draft = state.get("collaborative_draft") or {}
    quality_review = [
        _safe_text(item)
        for item in _safe_list(collaborative_draft.get("quality_notes"))[:2]
        if _safe_text(item)
    ]
    quality_review.extend([
        f"FinalAuditTool 结论：{final_audit_result['decision']}。",
        "已确认最终画像未混入岗位匹配排名。"
    ])
    tool_calls = _append_tool_call(
        state,
        called_by="质量复核智能体",
        tool_name="FinalAuditTool",
        purpose="复核工具链完整性、专家交接和最终画像边界",
        input_summary="tool_calls + collaboration_log + final profile report",
        output=final_audit_result
    )

    return {
        "quality_review": quality_review[:4],
        "tool_calls": tool_calls,
        "review_findings": consistency_result.get("findings", []),
        "shared_workspace": _merge_workspace(
            state,
            final_audit=final_audit_result,
            quality_review=quality_review[:4]
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="质量复核智能体",
            receiver="最终画像报告",
            message=f"复核结论：{final_audit_result['decision']}。报告可展示，但须保留工具链和证据边界。",
            artifact="quality_review + final_audit"
        ),
        "used_llm": bool(state.get("used_llm", False)),
        "llm_agents": list(state.get("llm_agents", [])),
        "agent_warning": state.get("agent_warning", ""),
        "workflow_steps": _append_workflow_step(
            state,
            step="05",
            agent="质量复核智能体",
            llm_role="共享草案校验专家",
            task="调用最终审计工具，复核单次 LLM 协作草案、工具链、专家交接和事实边界",
            status="completed",
            output=(
                f"工具链 {final_audit_result['tool_call_count'] + 1} 次调用，"
                f"专家交接 {final_audit_result['handoff_count'] + 1} 次；"
                f"完成 {len(quality_review[:4])} 条复核结论。"
            )
        )
    }


# =========================================================
# 构建 LangGraph 智能体
# =========================================================

def build_diagnosis_graph():
    builder = StateGraph(DiagnosisState)

    builder.add_node("extract_profile", extract_profile_node)
    builder.add_node("score_ability", score_ability_node)
    builder.add_node("analyze_profile_evidence", analyze_profile_evidence_node)
    builder.add_node("diagnose_ability", diagnose_ability_node)
    builder.add_node("review_profile", review_profile_node)

    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "score_ability")
    builder.add_edge("score_ability", "analyze_profile_evidence")
    builder.add_edge("analyze_profile_evidence", "diagnose_ability")
    builder.add_edge("diagnose_ability", "review_profile")
    builder.add_edge("review_profile", END)

    return builder.compile()


diagnosis_graph = build_diagnosis_graph()


# =========================================================
# 对外调用函数
# =========================================================

def _build_diagnosis_result(result: DiagnosisState) -> dict[str, Any]:
    """把 LangGraph 内部状态收敛成页面和数据库使用的公开结果。"""
    return {
        "ability_scores": result["ability_scores"],
        "score_evidence": result["score_evidence"],
        "recognized_skills": result["recognized_skills"],
        "profile_tags": result.get("profile_tags", []),
        "risk_flags": result.get("risk_flags", []),
        "evidence_cards": result.get("evidence_cards", []),
        "summary": result["summary"],
        "advantages": result["advantages"],
        "weaknesses": result["weaknesses"],
        "dimension_insights": result.get("dimension_insights", []),
        "development_focus": result.get("development_focus", []),
        "quality_review": result.get("quality_review", []),
        "workflow_steps": result.get("workflow_steps", []),
        "tool_calls": result.get("tool_calls", []),
        "collaboration_log": result.get("collaboration_log", []),
        "review_findings": result.get("review_findings", []),
        "shared_workspace": result.get("shared_workspace", {}),
        "agent_roster": AGENT_ROSTER,
        "llm_agents": result.get("llm_agents", []),
        "used_llm": result.get("used_llm", False),
        "agent_warning": result.get("agent_warning", "")
    }


def run_diagnosis_agent_stream(student_data: dict[str, str]):
    """
    按 LangGraph 节点完成顺序返回增量状态，供 Web 端实时展示生成过程。

    每个事件都带有当前节点、本节点输出和截至当前的合并状态；最后一个
    ``complete`` 事件包含与 ``run_diagnosis_agent`` 完全相同的公开结果。
    """
    merged_state: DiagnosisState = {"student": student_data}

    for update in diagnosis_graph.stream(
        {"student": student_data},
        stream_mode="updates",
    ):
        for node_name, node_output in update.items():
            if not isinstance(node_output, dict):
                continue
            merged_state.update(node_output)
            yield {
                "type": "node",
                "node": node_name,
                "output": node_output,
                "state": dict(merged_state),
            }

    yield {
        "type": "complete",
        "result": _build_diagnosis_result(merged_state),
    }


def run_diagnosis_agent(student_data: dict[str, str]) -> dict[str, Any]:
    """
    main.py 只需要调用这个函数即可获得完整诊断结果。
    """
    result = diagnosis_graph.invoke({
        "student": student_data
    })
    return _build_diagnosis_result(result)
