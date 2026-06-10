from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


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
    learning_tasks: list[dict[str, Any]]
    quality_review: list[str]
    workflow_steps: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    collaboration_log: list[dict[str, Any]]
    review_findings: list[dict[str, Any]]
    shared_workspace: dict[str, Any]
    llm_agents: list[str]

    summary: str
    advantages: list[str]
    weaknesses: list[str]
    growth_path: list[dict[str, Any]]

    used_llm: bool
    agent_warning: str


# =========================================================
# 大模型结构化输出格式
# =========================================================

class GrowthStage(BaseModel):
    stage: str = Field(description="阶段名称，例如第一阶段：基础补强")
    duration: str = Field(description="时间范围，例如第1-2个月")
    goal: str = Field(description="本阶段核心目标")
    actions: list[str] = Field(description="本阶段具体行动，建议3项")
    deliverables: list[str] = Field(description="可验证成果，建议1到2项")


class DimensionInsight(BaseModel):
    key: str = Field(description="能力维度英文键名")
    name: str = Field(description="能力维度中文名称")
    score: int = Field(description="能力分数")
    level: str = Field(description="能力等级")
    conclusion: str = Field(description="该维度画像结论")
    evidence: list[str] = Field(description="支撑该结论的学生证据")
    next_action: str = Field(description="下一步提升动作")


class AIReport(BaseModel):
    summary: str = Field(description="针对该学生的整体诊断总结")
    advantages: list[str] = Field(description="学生当前优势，建议2到3项")
    weaknesses: list[str] = Field(description="学生目前不足，建议2到3项")
    dimension_insights: list[DimensionInsight] = Field(description="四维能力洞察")
    development_focus: list[dict[str, str]] = Field(description="优先发展焦点")
    growth_path: list[GrowthStage] = Field(description="三阶段成长路径规划")


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
    "成长规划智能体",
    "质量复核智能体"
]


# =========================================================
# 通用工具函数
# =========================================================

def _safe_text(value: str | None) -> str:
    return (value or "").strip()


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
    任一专家失败时只降级当前节点，整个画像工作流继续运行。
    """

    try:
        llm = _create_llm()

        if llm is None:
            return None, False, ""

        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        return extract_json_from_llm_text(str(content)), True, ""

    except Exception as exc:
        return (
            None,
            False,
            f"{agent_name}调用失败，已切换规则兜底：{type(exc).__name__}: {exc}"
        )


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
    raw_text = " ".join(value for value in fields.values() if value)
    filled_fields = [key for key, value in fields.items() if value]
    missing_fields = [key for key, value in fields.items() if not value]

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

    has_project = bool(_safe_text(student.get("projects")))
    has_competition = bool(_safe_text(student.get("competitions")))
    has_certificate = bool(_safe_text(student.get("certificates")))
    has_target_job = bool(_safe_text(student.get("target_job")))
    has_intro = len(_safe_text(student.get("self_intro"))) >= 30

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


def task_decomposition_tool(state: DiagnosisState) -> dict[str, Any]:
    focus_items = state.get("development_focus", [])
    tasks = []

    for index, item in enumerate(focus_items[:4], start=1):
        tasks.append({
            "id": f"T{index}",
            "dimension": item.get("name", "能力补强"),
            "task": item.get("action", "完成一次能力补强任务"),
            "tool_check": "产物检查 + 二次画像对比",
            "acceptance": "能提交学习记录、项目材料或表达素材",
            "timebox": "1周"
        })

    if not tasks:
        tasks.append({
            "id": "T1",
            "dimension": "能力证据沉淀",
            "task": "补充技能、项目、竞赛、证书等画像证据。",
            "tool_check": "表单完整度检查",
            "acceptance": "更新后的学生信息表",
            "timebox": "1周"
        })

    return {
        "tasks": tasks,
        "acceptance_policy": "每个任务必须有可验收产物，并能回写到下一次能力画像。"
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
    """
    根据 .env 创建大模型。
    支持 DeepSeek 等 OpenAI 兼容接口。
    """

    use_llm = os.getenv("USE_LLM", "true").lower() == "true"
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not use_llm:
        return None

    if not api_key:
        raise RuntimeError("USE_LLM=true，但没有配置 LLM_API_KEY")

    if not model:
        raise RuntimeError("USE_LLM=true，但没有配置 LLM_MODEL")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.2,
        timeout=60,
        max_retries=2
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

def score_ability_node(state: DiagnosisState) -> dict[str, Any]:
    """
    使用可解释的规则生成四维能力分数。
    第一版避免让大模型直接随意修改分数，便于展示和答辩。
    """

    student = state["student"]
    text = state["normalized_text"]
    skills = state["recognized_skills"]
    score_result = rubric_score_tool(student, text, skills)
    scores = score_result["ability_scores"]
    evidence = score_result["score_evidence"]
    tool_calls = _append_tool_call(
        state,
        called_by="四维评分智能体",
        tool_name="RubricScoreCalculator",
        purpose="按固定评分规程计算四维分数并输出评分证据",
        input_summary=f"recognized_skills={len(skills)}，text_length={len(text)}",
        output=score_result
    )

    return {
        "ability_scores": scores,
        "score_evidence": evidence,
        "tool_calls": tool_calls,
        "shared_workspace": _merge_workspace(
            state,
            score_sheet=score_result
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="四维评分智能体",
            receiver="证据抽取智能体",
            message="四维分数已锁定，后续 LLM 只能解释和质询，不允许改分。",
            artifact="ability_scores + score_evidence + rubric_trace"
        ),
        "workflow_steps": _append_workflow_step(
            state,
            step="02",
            agent="四维评分智能体",
            task="调用评分规程工具，生成不可由 LLM 随意修改的四维基准分",
            status="completed",
            output=(
                "完成四维评分："
                f"专业 {scores['professional']}，实践 {scores['practice']}，"
                f"工具 {scores['tools']}，职业 {scores['career']}；"
                f"审计提示 {len(score_result['audit_notes'])} 条。"
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


def _fallback_evidence_analysis(state: DiagnosisState) -> dict[str, Any]:
    scores = state["ability_scores"]
    score_evidence = state["score_evidence"]
    student = state["student"]
    recognized_skills = state.get("recognized_skills", [])

    evidence_cards = []
    for key, meta in ABILITY_DIMENSIONS.items():
        evidence = score_evidence.get(key, [])
        evidence_cards.append({
            "dimension": key,
            "name": meta["name"],
            "agent": meta["agent"],
            "evidence": evidence,
            "confidence": _evidence_confidence(scores.get(key, 0), evidence),
            "interpretation": (
                f"{meta['name']}当前为{_score_level(scores.get(key, 0))}，"
                f"主要依据是：{'、'.join(evidence[:3])}。"
            )
        })

    profile_tags = []
    if recognized_skills:
        profile_tags.append(f"显性技能 {len(recognized_skills)} 项")
    if _safe_text(student.get("projects")):
        profile_tags.append("有项目经历")
    else:
        profile_tags.append("项目证据待补充")
    if _safe_text(student.get("target_job")):
        profile_tags.append("职业目标已填写")
    if scores.get("practice", 0) >= 70:
        profile_tags.append("实践基础较好")
    if scores.get("tools", 0) >= 70:
        profile_tags.append("工具栈较清晰")

    risk_flags = []
    if not _safe_text(student.get("projects")):
        risk_flags.append("项目经历为空，实践能力缺少可展示证据。")
    if not _safe_text(student.get("self_intro")):
        risk_flags.append("自我介绍为空，职业表达和经历复盘证据不足。")
    for key, score in scores.items():
        if score < 55:
            risk_flags.append(f"{ABILITY_DIMENSIONS[key]['name']}低于 55 分，需要优先补强。")

    return {
        "evidence_cards": evidence_cards,
        "profile_tags": profile_tags[:6] or ["画像信息待补充"],
        "risk_flags": risk_flags[:5] or ["暂未发现高风险短板，建议继续补充可量化成果。"]
    }


def _normalize_evidence_analysis(
    raw: dict[str, Any] | None,
    fallback: dict[str, Any]
) -> dict[str, Any]:
    if not raw:
        return fallback

    fallback_cards = fallback["evidence_cards"]
    raw_cards = _safe_list(raw.get("evidence_cards"))
    normalized_cards: list[dict[str, Any]] = []
    seen_dimensions: set[str] = set()

    for item in raw_cards:
        if not isinstance(item, dict):
            continue

        key = item.get("dimension") or item.get("key")
        name = item.get("name", "")

        if key not in ABILITY_DIMENSIONS:
            key = next(
                (
                    dimension_key
                    for dimension_key, meta in ABILITY_DIMENSIONS.items()
                    if meta["name"] == name
                ),
                None
            )

        if key not in ABILITY_DIMENSIONS or key in seen_dimensions:
            continue

        meta = ABILITY_DIMENSIONS[key]
        normalized_cards.append({
            "dimension": key,
            "name": meta["name"],
            "agent": item.get("agent") or meta["agent"],
            "evidence": _safe_list(item.get("evidence")) or fallback_cards[len(normalized_cards)]["evidence"],
            "confidence": item.get("confidence") or fallback_cards[len(normalized_cards)]["confidence"],
            "interpretation": item.get("interpretation") or fallback_cards[len(normalized_cards)]["interpretation"]
        })
        seen_dimensions.add(key)

    for card in fallback_cards:
        if card["dimension"] not in seen_dimensions:
            normalized_cards.append(card)

    return {
        "evidence_cards": normalized_cards,
        "profile_tags": _safe_list(raw.get("profile_tags")) or fallback["profile_tags"],
        "risk_flags": _safe_list(raw.get("risk_flags")) or fallback["risk_flags"]
    }


def analyze_profile_evidence_node(state: DiagnosisState) -> dict[str, Any]:
    """
    由证据抽取智能体把表单文本拆成画像证据卡。
    """

    fallback = _fallback_evidence_analysis(state)
    matrix_result = evidence_matrix_tool(
        scores=state["ability_scores"],
        score_evidence=state.get("score_evidence", {}),
        student=state["student"]
    )

    prompt = f"""
你是学生能力画像工作流中的“证据抽取智能体”。

任务：先读取上游共享工作区和 EvidenceMatrixTool 输出，再把能力画像证据拆成四类证据卡。
你必须服从工具输出中的 fact_boundary，不得编造未出现的项目、证书、竞赛、实习。

学生信息：
{json.dumps(state["student"], ensure_ascii=False)}

系统识别技能：
{json.dumps(state.get("recognized_skills", []), ensure_ascii=False)}

规则评分证据：
{json.dumps(state.get("score_evidence", {}), ensure_ascii=False)}

EvidenceMatrixTool 输出：
{json.dumps(matrix_result, ensure_ascii=False)}

请只输出 JSON，不要 Markdown，不要解释文字。格式如下：
{{
  "profile_tags": ["标签1", "标签2"],
  "risk_flags": ["风险提示1"],
  "evidence_cards": [
    {{
      "dimension": "professional",
      "name": "专业基础能力",
      "evidence": ["证据1", "证据2"],
      "confidence": "高/中/低",
      "interpretation": "证据解释"
    }}
  ]
}}
"""

    raw, used_llm, warning = _invoke_json_agent("证据抽取智能体", prompt)
    analysis = _normalize_evidence_analysis(raw, fallback)
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
            message="已按工具矩阵建立证据边界，请基于证据强弱给出画像结论，并标注推理限制。",
            artifact="evidence_matrix + evidence_cards + risk_flags"
        ),
        "used_llm": bool(state.get("used_llm", False)) or used_llm,
        "llm_agents": _append_llm_agent(state, "证据抽取智能体", used_llm),
        "agent_warning": _append_warning(state, warning),
        "workflow_steps": _append_workflow_step(
            state,
            step="03",
            agent="证据抽取智能体",
            llm_role="LLM 专家" if used_llm else "规则兜底智能体",
            task="调用证据矩阵工具，约束 LLM 只在事实边界内生成证据卡",
            status="llm_completed" if used_llm else "fallback_completed",
            output=(
                f"工具产出 {len(matrix_result['matrix'])} 个维度矩阵；"
                f"LLM 生成 {len(analysis['evidence_cards'])} 张证据卡、"
                f"{len(analysis['profile_tags'])} 个画像标签。"
            )
        )
    }


# =========================================================
# 节点四：能力归因智能体
# =========================================================

def _fallback_ability_report(state: DiagnosisState) -> dict[str, Any]:
    scores = state["ability_scores"]
    recognized_skills = state.get("recognized_skills", [])
    risk_flags = state.get("risk_flags", [])

    strongest_key = max(ABILITY_DIMENSIONS, key=lambda key: scores.get(key, 0))
    weakest_key = min(ABILITY_DIMENSIONS, key=lambda key: scores.get(key, 0))
    average_score = round(sum(scores.values()) / max(len(scores), 1), 1)

    dimension_insights = []
    for key, meta in ABILITY_DIMENSIONS.items():
        score = scores.get(key, 0)
        evidence = state.get("score_evidence", {}).get(key, [])
        level = _score_level(score)
        dimension_insights.append({
            "key": key,
            "name": meta["name"],
            "score": score,
            "level": level,
            "conclusion": (
                f"{meta['name']}处于“{level}”水平，"
                f"当前证据强度为{_evidence_confidence(score, evidence)}。"
            ),
            "evidence": evidence,
            "next_action": meta["next_action"]
        })

    advantages = [
        f"{ABILITY_DIMENSIONS[strongest_key]['name']}得分最高，当前为 {scores[strongest_key]} 分。",
        (
            f"已识别到显性技能：{'、'.join(recognized_skills[:6])}。"
            if recognized_skills
            else "学生已提交基础画像信息，可继续补充技能证据。"
        )
    ]

    if state["student"].get("projects"):
        advantages.append("已填写项目经历，可作为实践能力画像的核心证据。")
    else:
        advantages.append("画像已识别出实践证据缺口，后续提升方向较清晰。")

    weaknesses = [
        f"{ABILITY_DIMENSIONS[weakest_key]['name']}是当前最低维度，得分为 {scores[weakest_key]} 分。",
        risk_flags[0] if risk_flags else "经历证据仍需进一步量化，例如项目结果、个人职责和可验证成果。"
    ]
    weaknesses.append("建议把技能、项目、竞赛、证书分别补充为可核验材料，提升画像可信度。")

    development_focus = []
    for index, key in enumerate(
        sorted(ABILITY_DIMENSIONS, key=lambda item: scores.get(item, 0))[:3],
        start=1
    ):
        meta = ABILITY_DIMENSIONS[key]
        score = scores.get(key, 0)
        development_focus.append({
            "name": meta["name"],
            "priority": "高" if index == 1 or score < 55 else "中",
            "reason": f"当前得分 {score} 分，处于“{_score_level(score)}”水平。",
            "action": meta["next_action"]
        })

    return {
        "summary": (
            f"多智能体画像工作流显示，该学生四维能力均值约为 {average_score} 分。"
            f"其中{ABILITY_DIMENSIONS[strongest_key]['name']}相对突出，"
            f"{ABILITY_DIMENSIONS[weakest_key]['name']}需要优先补强。"
            "当前画像更适合用于能力诊断、成长跟踪和学习任务拆解，岗位匹配分析已拆分到独立模块。"
        ),
        "advantages": advantages[:3],
        "weaknesses": weaknesses[:3],
        "dimension_insights": dimension_insights,
        "development_focus": development_focus
    }


def _normalize_ability_report(
    raw: dict[str, Any] | None,
    fallback: dict[str, Any]
) -> dict[str, Any]:
    if not raw:
        return fallback

    report = {
        "summary": raw.get("summary") or fallback["summary"],
        "advantages": _safe_list(raw.get("advantages")) or fallback["advantages"],
        "weaknesses": _safe_list(raw.get("weaknesses")) or fallback["weaknesses"],
        "development_focus": _safe_list(raw.get("development_focus")) or fallback["development_focus"],
        "dimension_insights": []
    }

    raw_insights = _safe_list(raw.get("dimension_insights"))
    insight_by_key = {
        item.get("key"): item
        for item in raw_insights
        if isinstance(item, dict) and item.get("key") in ABILITY_DIMENSIONS
    }

    fallback_by_key = {
        item["key"]: item
        for item in fallback["dimension_insights"]
    }

    for key, meta in ABILITY_DIMENSIONS.items():
        source = insight_by_key.get(key, {})
        fallback_item = fallback_by_key[key]
        report["dimension_insights"].append({
            "key": key,
            "name": meta["name"],
            "score": fallback_item["score"],
            "level": source.get("level") or fallback_item["level"],
            "conclusion": source.get("conclusion") or fallback_item["conclusion"],
            "evidence": _safe_list(source.get("evidence")) or fallback_item["evidence"],
            "next_action": source.get("next_action") or fallback_item["next_action"]
        })

    return report


def diagnose_ability_node(state: DiagnosisState) -> dict[str, Any]:
    """
    由能力归因智能体解释分数和证据，产出画像结论。
    """

    fallback = _fallback_ability_report(state)
    audit_result = consistency_audit_tool(state)

    prompt = f"""
你是学生能力画像工作流中的“能力归因智能体”。

任务：根据四维分数、证据卡和 ConsistencyAuditTool 结果，输出学生能力画像结论。
你只分析学生能力，不做岗位推荐，不输出岗位匹配排名。
要求：
1. 不得修改四维分数。
2. 不得编造学生未填写的经历。
3. 如果审计工具指出证据不足，结论必须降低确定性，并用“当前证据显示/暂未看到”等表达。
4. 只输出 JSON，不要 Markdown。

学生信息：
{json.dumps(state["student"], ensure_ascii=False)}

四维能力分数：
{json.dumps(state["ability_scores"], ensure_ascii=False)}

证据卡：
{json.dumps(state.get("evidence_cards", []), ensure_ascii=False)}

ConsistencyAuditTool 输出：
{json.dumps(audit_result, ensure_ascii=False)}

请按格式输出：
{{
  "summary": "整体画像总结，120到180字",
  "advantages": ["优势1", "优势2", "优势3"],
  "weaknesses": ["短板1", "短板2", "短板3"],
  "dimension_insights": [
    {{
      "key": "professional",
      "level": "稳定具备",
      "conclusion": "该维度结论",
      "evidence": ["证据1", "证据2"],
      "next_action": "下一步动作"
    }}
  ],
  "development_focus": [
    {{
      "name": "优先发展方向",
      "priority": "高/中",
      "reason": "为什么优先",
      "action": "具体动作"
    }}
  ]
}}
"""

    raw, used_llm, warning = _invoke_json_agent("能力归因智能体", prompt)
    report = _normalize_ability_report(raw, fallback)
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
            receiver="成长规划智能体",
            message="画像结论已根据一致性审计收敛，请把发展焦点拆成可验收任务，不要引入岗位排名。",
            artifact="ability_report + consistency_audit"
        ),
        "used_llm": bool(state.get("used_llm", False)) or used_llm,
        "llm_agents": _append_llm_agent(state, "能力归因智能体", used_llm),
        "agent_warning": _append_warning(state, warning),
        "workflow_steps": _append_workflow_step(
            state,
            step="04",
            agent="能力归因智能体",
            llm_role="LLM 专家" if used_llm else "规则兜底智能体",
            task="调用一致性审计工具，再由 LLM 解释分数和证据之间的因果关系",
            status="llm_completed" if used_llm else "fallback_completed",
            output=(
                f"审计结论：{audit_result['decision']}；完成 {len(report['dimension_insights'])} 个维度洞察，"
                f"提炼 {len(report['development_focus'])} 个发展焦点。"
            )
        )
    }


# =========================================================
# 节点五：成长规划智能体
# =========================================================

def _fallback_growth_plan(state: DiagnosisState) -> dict[str, Any]:
    focus_items = state.get("development_focus", [])
    primary_focus = focus_items[0]["name"] if focus_items else "能力证据沉淀"
    secondary_focus = focus_items[1]["name"] if len(focus_items) > 1 else "项目实践能力"

    growth_path = [
        {
            "stage": "第一阶段：画像校准与基础补强",
            "duration": "第1-2周",
            "goal": f"围绕{primary_focus}补齐基础证据，让画像从“可评分”变成“可解释”。",
            "actions": [
                "把课程、技能、项目、证书分别整理成清单，标注掌握程度。",
                "针对最低分维度完成一轮知识点复盘和练习记录。",
                "补充自我介绍中的个人职责、技术动作和结果数据。"
            ],
            "deliverables": [
                "能力证据清单",
                "最低分维度补强笔记"
            ]
        },
        {
            "stage": "第二阶段：项目作品与能力闭环",
            "duration": "第3-6周",
            "goal": f"通过一个小型项目强化{secondary_focus}，形成可展示成果。",
            "actions": [
                "选择一个真实场景，设计需求、技术方案和验收标准。",
                "用 Git 记录开发过程，补齐 README、运行截图和问题复盘。",
                "把项目经历改写成“问题-行动-结果”的能力证据。"
            ],
            "deliverables": [
                "可运行项目仓库",
                "项目复盘文档"
            ]
        },
        {
            "stage": "第三阶段：表达复盘与成长追踪",
            "duration": "第7-8周",
            "goal": "把能力画像转化为简历素材、面试表达和下一轮成长数据。",
            "actions": [
                "依据四维能力重新整理简历中的技能和项目表述。",
                "围绕优势和短板准备 3 段自我介绍与问答素材。",
                "再次提交画像信息，对比四维分数和证据变化。"
            ],
            "deliverables": [
                "能力证据型简历素材",
                "二次画像对比记录"
            ]
        }
    ]

    learning_tasks = []
    for item in focus_items[:4]:
        learning_tasks.append({
            "dimension": item.get("name", "能力补强"),
            "task": item.get("action", "完成一次能力补强任务"),
            "timebox": "1周",
            "output": "可检查的学习记录或项目材料"
        })

    if not learning_tasks:
        learning_tasks.append({
            "dimension": "能力证据沉淀",
            "task": "补充技能、项目、竞赛、证书等画像证据。",
            "timebox": "1周",
            "output": "更新后的学生信息表"
        })

    return {
        "growth_path": growth_path,
        "learning_tasks": learning_tasks
    }


def _normalize_growth_plan(
    raw: dict[str, Any] | None,
    fallback: dict[str, Any],
    tool_tasks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if not raw:
        result = dict(fallback)
        if tool_tasks:
            result["learning_tasks"] = tool_tasks
        return result

    growth_path = _safe_list(raw.get("growth_path")) or fallback["growth_path"]
    learning_tasks = _safe_list(raw.get("learning_tasks")) or tool_tasks or fallback["learning_tasks"]

    if len(growth_path) < 3:
        growth_path = fallback["growth_path"]

    return {
        "growth_path": growth_path,
        "learning_tasks": learning_tasks
    }


def plan_growth_node(state: DiagnosisState) -> dict[str, Any]:
    """
    由成长规划智能体把画像结论拆成阶段任务。
    """

    fallback = _fallback_growth_plan(state)
    task_result = task_decomposition_tool(state)

    prompt = f"""
你是学生能力画像工作流中的“成长规划智能体”。

任务：根据能力画像和 TaskDecompositionTool 输出，生成学生能力成长路径。
要求：
1. growth_path 必须是三个阶段。
2. learning_tasks 必须保留工具给出的验收口径，细化到可执行、可验收。
3. 不输出岗位推荐，不输出岗位匹配度。
4. 只输出 JSON，不要 Markdown。

画像总结：
{state.get("summary", "")}

维度洞察：
{json.dumps(state.get("dimension_insights", []), ensure_ascii=False)}

发展焦点：
{json.dumps(state.get("development_focus", []), ensure_ascii=False)}

TaskDecompositionTool 输出：
{json.dumps(task_result, ensure_ascii=False)}

请按格式输出：
{{
  "growth_path": [
    {{
      "stage": "第一阶段：阶段名",
      "duration": "时间范围",
      "goal": "阶段目标",
      "actions": ["行动1", "行动2", "行动3"],
      "deliverables": ["成果1", "成果2"]
    }}
  ],
  "learning_tasks": [
    {{
      "dimension": "对应能力维度",
      "task": "具体任务",
      "timebox": "建议周期",
      "output": "验收产物"
    }}
  ]
}}
"""

    raw, used_llm, warning = _invoke_json_agent("成长规划智能体", prompt)
    plan = _normalize_growth_plan(
        raw,
        fallback,
        tool_tasks=task_result.get("tasks", [])
    )
    tool_calls = _append_tool_call(
        state,
        called_by="成长规划智能体",
        tool_name="TaskDecompositionTool",
        purpose="把发展焦点拆成带验收标准的学习任务",
        input_summary="development_focus + ability_report",
        output=task_result
    )

    return {
        **plan,
        "tool_calls": tool_calls,
        "shared_workspace": _merge_workspace(
            state,
            task_decomposition=task_result,
            growth_plan=plan
        ),
        "collaboration_log": _append_collaboration_log(
            state,
            sender="成长规划智能体",
            receiver="质量复核智能体",
            message="成长路径和任务拆解已生成，请复核是否都能追溯到画像证据和工具结果。",
            artifact="growth_plan + task_decomposition"
        ),
        "used_llm": bool(state.get("used_llm", False)) or used_llm,
        "llm_agents": _append_llm_agent(state, "成长规划智能体", used_llm),
        "agent_warning": _append_warning(state, warning),
        "workflow_steps": _append_workflow_step(
            state,
            step="05",
            agent="成长规划智能体",
            llm_role="LLM 专家" if used_llm else "规则兜底智能体",
            task="调用任务拆解工具，把画像结论转成阶段路径和验收任务",
            status="llm_completed" if used_llm else "fallback_completed",
            output=(
                f"工具拆解 {len(task_result['tasks'])} 个任务；生成 {len(plan['growth_path'])} 个成长阶段、"
                f"{len(plan['learning_tasks'])} 个可验收任务。"
            )
        )
    }


# =========================================================
# 节点六：质量复核智能体
# =========================================================

def _fallback_quality_review(state: DiagnosisState) -> list[str]:
    review = [
        "画像结论只使用学生已提交信息和系统识别证据，未补写未出现经历。",
        "四维分数由规则评分智能体生成，LLM 专家只负责证据解释、归因和任务细化。",
        "学生能力画像已与岗位匹配分析拆分，当前结果不包含岗位排名或岗位匹配度。"
    ]

    if state.get("tool_calls"):
        review.append(f"本次画像共记录 {len(state.get('tool_calls', []))} 次工具调用，可追溯分析依据。")

    return review


def review_profile_node(state: DiagnosisState) -> dict[str, Any]:
    """
    由质量复核智能体检查画像是否可信、是否混入岗位匹配内容。
    """

    fallback = _fallback_quality_review(state)
    consistency_result = consistency_audit_tool(state)
    final_audit_result = {
        "consistency": consistency_result,
        "tool_call_count": len(state.get("tool_calls", [])),
        "handoff_count": len(state.get("collaboration_log", [])),
        "task_count": len(state.get("learning_tasks", [])),
        "has_job_ranking": False,
        "decision": (
            "可发布"
            if state.get("tool_calls") and state.get("collaboration_log")
            else "可发布，但旧记录缺少完整工具链"
        )
    }

    prompt = f"""
你是学生能力画像工作流中的“质量复核智能体”。

任务：像审稿人一样检查能力画像报告是否可靠。
你必须读取工具调用链、专家交接记录和 FinalAuditTool 输出，尤其确认没有编造经历、没有混入岗位匹配排名。
只输出 JSON，不要 Markdown。

学生信息：
{json.dumps(state["student"], ensure_ascii=False)}

工具调用链：
{json.dumps(state.get("tool_calls", []), ensure_ascii=False)}

专家协作交接：
{json.dumps(state.get("collaboration_log", []), ensure_ascii=False)}

FinalAuditTool 输出：
{json.dumps(final_audit_result, ensure_ascii=False)}

画像报告：
{json.dumps({
    "summary": state.get("summary", ""),
    "advantages": state.get("advantages", []),
    "weaknesses": state.get("weaknesses", []),
    "dimension_insights": state.get("dimension_insights", []),
    "growth_path": state.get("growth_path", [])
}, ensure_ascii=False)}

请输出：
{{
  "quality_review": ["复核结论1", "复核结论2", "复核结论3"]
}}
"""

    raw, used_llm, warning = _invoke_json_agent("质量复核智能体", prompt)
    quality_review = _safe_list(raw.get("quality_review")) if raw else []
    quality_review = quality_review or fallback
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
        "used_llm": bool(state.get("used_llm", False)) or used_llm,
        "llm_agents": _append_llm_agent(state, "质量复核智能体", used_llm),
        "agent_warning": _append_warning(state, warning),
        "workflow_steps": _append_workflow_step(
            state,
            step="06",
            agent="质量复核智能体",
            llm_role="LLM 专家" if used_llm else "规则兜底智能体",
            task="调用最终审计工具，复核工具链、专家交接和事实边界",
            status="llm_completed" if used_llm else "fallback_completed",
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
    builder.add_node("plan_growth", plan_growth_node)
    builder.add_node("review_profile", review_profile_node)

    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "score_ability")
    builder.add_edge("score_ability", "analyze_profile_evidence")
    builder.add_edge("analyze_profile_evidence", "diagnose_ability")
    builder.add_edge("diagnose_ability", "plan_growth")
    builder.add_edge("plan_growth", "review_profile")
    builder.add_edge("review_profile", END)

    return builder.compile()


diagnosis_graph = build_diagnosis_graph()


# =========================================================
# 对外调用函数
# =========================================================

def run_diagnosis_agent(student_data: dict[str, str]) -> dict[str, Any]:
    """
    main.py 只需要调用这个函数即可获得完整诊断结果。
    """

    result = diagnosis_graph.invoke({
        "student": student_data
    })

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
        "learning_tasks": result.get("learning_tasks", []),
        "quality_review": result.get("quality_review", []),
        "workflow_steps": result.get("workflow_steps", []),
        "tool_calls": result.get("tool_calls", []),
        "collaboration_log": result.get("collaboration_log", []),
        "review_findings": result.get("review_findings", []),
        "shared_workspace": result.get("shared_workspace", {}),
        "agent_roster": AGENT_ROSTER,
        "llm_agents": result.get("llm_agents", []),
        "growth_path": result["growth_path"],
        "used_llm": result.get("used_llm", False),
        "agent_warning": result.get("agent_warning", "")
    }
