from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.services.ability_match_service import (
    match_profile_to_job,
    score_four_dimensions,
)

try:
    from app.services.job_candidate_service import prefilter_job_records
except Exception:  # 避免单文件语法检查时因项目路径问题失败
    prefilter_job_records = None

try:
    from app.services.job_vector_service import retrieve_jobs_by_vector
except Exception:  # 避免单文件语法检查时因项目路径问题失败
    retrieve_jobs_by_vector = None

from app.services.llm_errors import LLMCallError
from app.services.model_config_service import create_configured_chat_model

load_dotenv()

logger = logging.getLogger(__name__)

DIMENSION_KEYS = ["professional", "practice", "tools", "career"]
DIMENSION_LABELS = {
    "professional": "专业基础能力",
    "practice": "技术实践能力",
    "tools": "工具技能能力",
    "career": "职业发展能力",
}

# 双向匹配权重：最终匹配分 = 学生适岗分 × 60% + 岗位适生分 × 40%
STUDENT_TO_JOB_WEIGHT = 0.6
JOB_TO_STUDENT_WEIGHT = 0.4

# 岗位适生分的五个解释性子维度
GROWTH_COMPONENT_KEYS = [
    "career_consistency",        # 职业方向一致性
    "gap_fillability",           # 缺口可补齐性
    "difficulty_suitability",    # 岗位难度适配性
    "learning_path_mappability", # 课程/项目/证书可映射性
    "growth_value",              # 岗位成长价值
]

JOB_FAMILY_RULES = [
    ("gis", ("gis", "地理信息", "空间信息", "测绘", "遥感", "地图")),
    ("ai_algorithm", ("算法", "机器学习", "深度学习", "大模型", "ai应用", "nlp", "视觉")),
    ("data", ("数据分析", "大数据", "数据开发", "数据工程", "数仓", "bi")),
    ("security", ("安全", "网络安全", "信息安全", "渗透")),
    ("cloud_ops", ("运维", "devops", "云计算", "云平台", "sre", "kubernetes")),
    ("testing", ("测试", "质量保障", "qa")),
    ("product", ("产品经理", "产品运营", "项目经理")),
    ("embedded", ("嵌入式", "物联网", "单片机", "驱动开发")),
    ("frontend", ("前端", "vue", "react", "小程序")),
    ("database", ("数据库", "dba")),
    ("network", ("网络工程", "网络运维")),
    (
        "application_development",
        ("后端", "服务端", "软件开发", "软件工程", "全栈", "java", "python", "开发工程师"),
    ),
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_safe_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_safe_text(v)}" for k, v in value.items())
    return str(value)


def _safe_json_loads(text: str) -> dict[str, Any]:
    """尽量从大模型返回中解析 JSON。"""
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _clamp_score(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0
    return max(0, min(100, round(number)))


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            return _listify(loaded)
        except Exception:
            parts = re.split(r"[,，、;；\n\r]+", text)
            return [p.strip() for p in parts if p.strip()]
    if isinstance(value, dict):
        result: list[str] = []
        for v in value.values():
            result.extend(_listify(v))
        return _dedupe(result)
    if isinstance(value, Iterable):
        result = []
        for item in value:
            result.extend(_listify(item))
        return _dedupe(result)
    return [str(value).strip()] if str(value).strip() else []


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _safe_text(value).strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def normalize_job_title(value: Any) -> str:
    """把大小写、空格、括号说明和常见岗位后缀统一为稳定的去重键。"""
    raw_text = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        _safe_text(value).lower(),
    )
    if "java" in raw_text and ("后端" in raw_text or "服务端" in raw_text):
        return "java后端工程师"
    if "python" in raw_text and ("后端" in raw_text or "服务端" in raw_text):
        return "python后端工程师"

    text = _safe_text(value).lower()
    text = re.sub(r"[\(（\[【].*?[\)）\]】]", "", text)
    text = re.sub(r"高级|中级|初级|资深|实习|应届", "", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    replacements = (
        ("java后端开发工程师", "java后端工程师"),
        ("python后端开发工程师", "python后端工程师"),
        ("全栈开发工程师", "全栈工程师"),
        ("前端开发工程师", "前端工程师"),
        ("软件开发工程师", "软件工程师"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def infer_job_family(value: Any) -> str:
    """将同义岗位归入较宽的岗位族，用于控制 TOP5 的方向多样性。"""
    title = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        _safe_text(value).lower(),
    )
    for family, keywords in JOB_FAMILY_RULES:
        if any(keyword in title for keyword in keywords):
            return family
    return title or "other"


def diversify_job_matches(
    matches: list[dict[str, Any]],
    top_n: int = 5,
    min_diversity_score: float = 60.0,
    max_score_gap: float = 18.0,
) -> list[dict[str, Any]]:
    """
    在保持分数优先的前提下去重并增加岗位族多样性。

    先在“至少 60 分且与最高分相差不超过 18 分”的岗位中做多样化：
    第一轮每个岗位族只取 1 个，第二轮放宽到每族 2 个；
    最后按分数补齐，但始终不返回归一化标题相同的岗位。
    """
    ranked = sorted(
        matches,
        key=lambda item: item.get("match_score", 0),
        reverse=True,
    )
    if not ranked:
        return []

    best_score = float(ranked[0].get("match_score", 0) or 0)
    diversity_floor = max(min_diversity_score, best_score - max_score_gap)
    diversity_ranked = [
        item
        for item in ranked
        if float(item.get("match_score", 0) or 0) >= diversity_floor
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    selected_titles: set[str] = set()
    family_counts: dict[str, int] = {}

    for family_limit in (1, 2):
        for item in diversity_ranked:
            if len(selected) >= top_n:
                break
            if id(item) in selected_ids:
                continue

            title_key = normalize_job_title(item.get("job_name"))
            if not title_key or title_key in selected_titles:
                continue

            family = infer_job_family(item.get("job_name"))
            if family_counts.get(family, 0) >= family_limit:
                continue

            enriched = dict(item)
            enriched["job_family"] = family
            selected.append(enriched)
            selected_ids.add(id(item))
            selected_titles.add(title_key)
            family_counts[family] = family_counts.get(family, 0) + 1

        if len(selected) >= top_n:
            break

    # 高相关岗位族不足时按原始分数补齐，绝不为了多样化强塞低分岗位。
    for item in ranked:
        if len(selected) >= top_n:
            break
        if id(item) in selected_ids:
            continue

        title_key = normalize_job_title(item.get("job_name"))
        if not title_key or title_key in selected_titles:
            continue

        enriched = dict(item)
        enriched["job_family"] = infer_job_family(item.get("job_name"))
        selected.append(enriched)
        selected_ids.add(id(item))
        selected_titles.add(title_key)

    primary_count = min(5, len(selected))
    primary = sorted(
        selected[:primary_count],
        key=lambda item: item.get("match_score", 0),
        reverse=True,
    )
    remaining = sorted(
        selected[primary_count:],
        key=lambda item: item.get("match_score", 0),
        reverse=True,
    )
    return primary + remaining


def diversify_candidate_records(
    records: list[Any],
    top_k: int = 60,
    target_job: Any = "",
) -> list[Any]:
    """
    在向量召回结果中控制单一岗位族占比，避免候选池被同类岗位淹没。

    目标岗位族最多保留 24 条，其他岗位族最多保留 8 条；如果岗位族
    数量不足，再按原始相似度顺序补齐。全程去除归一化标题相同的岗位。
    """
    if top_k <= 0:
        return []

    target_family = infer_job_family(target_job)
    selected: list[Any] = []
    deferred: list[Any] = []
    selected_titles: set[str] = set()
    family_counts: dict[str, int] = {}

    def job_title(record: Any) -> Any:
        if isinstance(record, dict):
            return record.get("job_name") or record.get("name") or record.get("title")
        return (
            getattr(record, "job_name", None)
            or getattr(record, "name", None)
            or getattr(record, "title", None)
        )

    for record in records:
        title = job_title(record)
        title_key = normalize_job_title(title)
        if not title_key or title_key in selected_titles:
            continue

        family = infer_job_family(title)
        family_limit = 24 if family == target_family else 8
        if family_counts.get(family, 0) >= family_limit:
            deferred.append(record)
            continue

        selected.append(record)
        selected_titles.add(title_key)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= top_k:
            return selected

    # 岗位库方向不足时，继续按原始相似度顺序补齐，避免候选数量过少。
    for record in deferred:
        title_key = normalize_job_title(job_title(record))
        if not title_key or title_key in selected_titles:
            continue
        selected.append(record)
        selected_titles.add(title_key)
        if len(selected) >= top_k:
            break

    return selected


def _create_llm() -> ChatOpenAI:
    return create_configured_chat_model(
        temperature=0,
        timeout=int(os.getenv("JOB_MATCH_LLM_TIMEOUT_SECONDS", "120")),
        max_retries=0,
        task_name="ABILITY_MATCH",
        legacy_task_model_envs=("ABILITY_MATCH_MODEL",),
    )


def _ensure_assessment_payload(data: dict[str, Any]) -> None:
    scores = data.get("ability_scores")
    evidence = data.get("score_evidence")
    if not isinstance(scores, dict) or not isinstance(evidence, dict):
        raise LLMCallError()
    if any(key not in scores for key in DIMENSION_KEYS):
        raise LLMCallError()
    if any(key not in evidence for key in DIMENSION_KEYS):
        raise LLMCallError()


def build_resume_context(student_data: dict[str, Any], resume_text: str | None = None) -> str:
    fields = {
        "姓名": student_data.get("name"),
        "专业": student_data.get("major"),
        "年级/学历": student_data.get("grade") or student_data.get("education"),
        "目标岗位": student_data.get("target_job"),
        "技能": student_data.get("skills"),
        "项目经历": student_data.get("projects"),
        "实习经历": student_data.get("internships"),
        "竞赛经历": student_data.get("competitions"),
        "证书": student_data.get("certificates"),
        "自我介绍": student_data.get("self_intro"),
        "简历全文": resume_text or student_data.get("resume_text") or student_data.get("normalized_text"),
    }
    return "\n".join(f"{k}: {_safe_text(v)}" for k, v in fields.items() if _safe_text(v).strip())


def _validate_assessment(data: dict[str, Any]) -> dict[str, Any]:
    scores_raw = data.get("ability_scores") or {}
    scores = {key: _clamp_score(scores_raw.get(key)) for key in DIMENSION_KEYS}
    evidence_raw = data.get("score_evidence") or {}
    evidence = {key: _listify(evidence_raw.get(key)) for key in DIMENSION_KEYS}
    return {
        "ability_scores": scores,
        "score_evidence": evidence,
        "recognized_skills": _dedupe(_listify(data.get("recognized_skills"))),
        "target_roles": _dedupe(_listify(data.get("target_roles"))),
        "assessment_summary": _safe_text(data.get("assessment_summary")),
        "used_llm": bool(data.get("used_llm", True)),
        "agent_warning": _safe_text(data.get("agent_warning")),
    }


def score_four_dimensions_llm(student_data: dict[str, Any], resume_text: str | None = None) -> dict[str, Any]:
    """
    AI 大模型版四维能力评分。
    大模型依据简历语义、项目难度、技术深度、岗位目标输出结构化评分。
    """
    llm = _create_llm()

    resume_context = build_resume_context(student_data, resume_text)
    prompt = f"""
你是“学生成长诊断与岗位匹配系统”的 AI 能力评估专家。请基于简历语义进行能力评估，不要用关键词数量机械计分。

请评估四维能力：
1. professional：专业基础能力。看专业课程、计算机基础、后端/前端/算法等基础知识深度。
2. practice：技术实践能力。看项目复杂度、职责真实性、工程落地、业务理解、问题解决证据。
3. tools：工具技能能力。看开发工具、框架、数据库、中间件、部署、协作工具的工程使用熟练度。
4. career：职业发展能力。看目标岗位清晰度、简历完整性、表达能力、证书/英语/职业准备、软技能证据。

评分要求：
- 每个分数必须是 0-100 的整数。
- 必须根据简历证据评分，不能编造没有出现的经历。
- 如果只“了解”某技术，不要按“熟练掌握”给高分。
- 项目中体现架构设计、部署、性能优化、认证鉴权、缓存、消息队列等，可以提高 practice/tools。
- 输出必须是严格 JSON，不要 Markdown。

学生简历/表单信息：
{resume_context}

返回格式：
{{
  "ability_scores": {{
    "professional": 0,
    "practice": 0,
    "tools": 0,
    "career": 0
  }},
  "score_evidence": {{
    "professional": ["证据1", "证据2"],
    "practice": ["证据1", "证据2"],
    "tools": ["证据1", "证据2"],
    "career": ["证据1", "证据2"]
  }},
  "recognized_skills": ["从简历语义识别出的技能"],
  "target_roles": ["适合的岗位方向1", "适合的岗位方向2"],
  "assessment_summary": "100字以内总结"
}}
"""
    try:
        response = llm.invoke(prompt)
        parsed = _safe_json_loads(_safe_text(response.content))
        _ensure_assessment_payload(parsed)
        parsed["used_llm"] = True
        return _validate_assessment(parsed)
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


def record_to_job_dict(record: Any) -> dict[str, Any]:
    def get(*keys: str) -> Any:
        for key in keys:
            if isinstance(record, dict) and key in record:
                return record.get(key)
            if hasattr(record, key):
                return getattr(record, key)
        return None

    return {
        "job_name": get("job_name", "name"),
        "company_name": get("company_name"),
        "hiring_city": get("hiring_city"),
        "educational_requirements": get("educational_requirements", "education_requirement", "education"),
        "required_skills": _listify(get("required_skills", "required_skills_json", "skills", "skill_tags")),
        "related_projects": _listify(get("related_projects", "related_projects_json", "projects")),
        "recommended_courses": _listify(get("recommended_courses", "recommended_courses_json", "courses")),
        "recommended_certificates": _listify(get("recommended_certificates", "recommended_certificates_json", "certificates")),
        "salary_range": get("salary_range"),
        "description": get("description", "job_description"),
        "local_match_score": get("match_score"),
        "local_matched_skills": _listify(get("matched_skills")),
        "local_missing_skills": _listify(get("missing_skills", "skill_gaps")),
        "local_recommend_reason": get("recommend_reason", "reason"),
    }


def _normalize_component_scores(raw: Any, expected_keys: list[str]) -> dict[str, int]:
    if not isinstance(raw, dict):
        raw = {}
    return {key: _clamp_score(raw.get(key)) for key in expected_keys}


def _build_bidirectional_summary(
    student_to_job_score: int,
    job_to_student_score: int,
    score_components: dict[str, int],
    growth_components: dict[str, int],
) -> str:
    return (
        f"学生适岗分{student_to_job_score}，岗位适生分{job_to_student_score}。"
        f"其中职业方向一致性{growth_components.get('career_consistency', 0)}，"
        f"缺口可补齐性{growth_components.get('gap_fillability', 0)}，"
        f"岗位难度适配性{growth_components.get('difficulty_suitability', 0)}。"
    )


def _validate_job_match(item: dict[str, Any], job_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    job_id = _safe_text(item.get("job_id"))
    base = job_map.get(job_id, {})
    if not base:
        item_job_name = normalize_job_title(item.get("job_name"))
        base = next(
            (
                job
                for job in job_map.values()
                if normalize_job_title(job.get("job_name")) == item_job_name
            ),
            {},
        )
    if not base:
        raise LLMCallError()
    job_name = _safe_text(item.get("job_name") or base.get("job_name") or "未知岗位")

    score_components = _normalize_component_scores(
        item.get("score_components"),
        [
            "resume_job_fit",
            "project_relevance",
            "skill_depth",
            "education_and_experience_fit",
            "career_direction_fit",
        ],
    )
    growth_components = _normalize_component_scores(item.get("growth_components"), GROWTH_COMPONENT_KEYS)

    local_score = _clamp_score(base.get("local_match_score"))
    student_to_job_score = _clamp_score(
        item.get("student_to_job_score")
        or item.get("student_fit_score")
        or item.get("resume_to_job_score")
        or score_components.get("resume_job_fit")
        or local_score
    )
    job_to_student_score = _clamp_score(
        item.get("job_to_student_score")
        or item.get("growth_path_score")
        or item.get("job_growth_score")
        or growth_components.get("career_consistency")
        or local_score
    )

    # 如果大模型没有给最终分，就按双向权重重新计算。
    calculated_final = round(student_to_job_score * STUDENT_TO_JOB_WEIGHT + job_to_student_score * JOB_TO_STUDENT_WEIGHT)
    match_score = _clamp_score(item.get("match_score") if item.get("match_score") is not None else calculated_final)

    bidirectional_summary = _safe_text(item.get("bidirectional_summary")) or _build_bidirectional_summary(
        student_to_job_score,
        job_to_student_score,
        score_components,
        growth_components,
    )

    recommend_reason = _safe_text(item.get("recommend_reason") or item.get("reason"))
    if not recommend_reason:
        recommend_reason = _safe_text(base.get("local_recommend_reason")) or bidirectional_summary

    growth_path_reason = _safe_text(item.get("growth_path_reason"))
    if not growth_path_reason:
        growth_path_reason = _safe_text(item.get("growth_value_reason") or item.get("job_to_student_reason"))

    return {
        "job_name": job_name,
        "company_name": _safe_text(base.get("company_name")),
        "hiring_city": _safe_text(base.get("hiring_city")),
        "salary_range": _safe_text(base.get("salary_range")),
        "education_requirement": _safe_text(base.get("educational_requirements")),

        # 兼容原前端字段：原页面继续使用 match_score、matched_skills、missing_skills 等字段即可。
        "match_score": match_score,
        "matched_skills": _dedupe(
            _listify(item.get("matched_skills"))
            or _listify(base.get("local_matched_skills"))
        ),
        "missing_skills": _dedupe(
            _listify(item.get("missing_skills"))
            or _listify(base.get("local_missing_skills"))
        ),
        "skill_gaps": _dedupe(
            _listify(item.get("skill_gaps"))
            or _listify(item.get("missing_skills"))
            or _listify(base.get("local_missing_skills"))
        )[:8],
        "matched_projects": _dedupe(_listify(item.get("matched_projects"))),
        "matched_certificates": _dedupe(_listify(item.get("matched_certificates"))),
        "recommend_reason": recommend_reason,
        "reason": recommend_reason,

        # 新增：双向匹配算法核心输出。
        "student_to_job_score": student_to_job_score,
        "job_to_student_score": job_to_student_score,
        "student_fit_score": student_to_job_score,  # 兼容别名
        "growth_path_score": job_to_student_score,  # 兼容别名
        "bidirectional_summary": bidirectional_summary,
        "growth_path_reason": growth_path_reason,
        "score_components": score_components,
        "growth_components": growth_components,
        "match_formula": "match_score = student_to_job_score * 0.6 + job_to_student_score * 0.4",
        "used_llm": True,
    }




def _build_job_match_prompt(
    student_data: dict[str, Any],
    assessment: dict[str, Any],
    resume_context: str,
    compact_jobs: list[dict[str, Any]],
) -> str:
    """构造单个 batch 的双向岗位匹配 prompt。同步和异步函数共用。"""
    return f"""
你是“学生成长诊断与路径规划智能体”的岗位精排专家。请基于简历语义、四维画像和本地初排证据，对候选岗位重新排序。

请对每个岗位同时计算两个方向的分数：

一、学生适岗分 student_to_job_score，0-100：
回答“学生当前能力是否满足岗位要求”。重点看：
1. resume_job_fit：简历技能与岗位技能要求的匹配程度；
2. project_relevance：项目经历与岗位业务/技术场景的相关性；
3. skill_depth：技能掌握深度，区分“了解、熟悉、熟练、实际使用”；
4. education_and_experience_fit：学历、专业、年级、经验与岗位要求是否匹配；
5. career_direction_fit：目标岗位与简历主线是否一致。

二、岗位适生分 job_to_student_score，0-100：
回答“这个岗位是否适合作为学生下一阶段成长目标”，重点看职业方向一致性、缺口可补齐性、难度和成长价值。

最终匹配分必须按下面公式计算：
match_score = student_to_job_score * 0.6 + job_to_student_score * 0.4

评分要求：
- 分数必须是 0-100 整数。
- 80以上表示高度匹配，60-79表示基本匹配，40-59表示存在明显短板，40以下表示不建议优先投递。
- 不能编造简历没有体现的技能或项目。
- 必须返回本批次所有岗位的评分结果。
- 每个岗位只返回 5 个字段，不要输出技能列表、评分子项或成长路径。
- recommend_reason 控制在 35 个汉字以内。
- 输出严格 JSON，不要 Markdown。

候选人信息：
{resume_context}

四维能力画像：
{json.dumps(assessment, ensure_ascii=False)}

岗位列表：
{json.dumps(compact_jobs, ensure_ascii=False)}

返回格式：
{{
  "matches": [
    {{
      "job_id": "job_1",
      "job_name": "岗位名称",
      "student_to_job_score": 0,
      "job_to_student_score": 0,
      "match_score": 0,
      "recommend_reason": "精排理由，35字以内"
    }}
  ]
}}
"""


def _prepare_job_batches(jobs: list[dict[str, Any]], batch_size: int) -> list[tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]]:
    """把岗位列表切分为 batch，同时构造 job_id 到岗位原始信息的映射。"""
    batches: list[tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        job_map: dict[str, dict[str, Any]] = {}
        compact_jobs = []
        for offset, job in enumerate(batch):
            job_id = f"job_{start + offset + 1}"
            job_map[job_id] = job
            compact_jobs.append({"job_id": job_id, **job})
        batches.append((job_map, compact_jobs))
    return batches


async def _invoke_match_batch_async(
    llm: ChatOpenAI,
    prompt: str,
    job_map: dict[str, dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """异步调用单个 batch，自动重试并容忍单条数据格式异常。"""
    async with semaphore:
        max_attempts = max(1, int(os.getenv("JOB_MATCH_LLM_MAX_ATTEMPTS", "3")))
        best_results: list[dict[str, Any]] = []
        expected_count = len(job_map)

        for attempt in range(1, max_attempts + 1):
            try:
                response = await llm.ainvoke(prompt)
                parsed = _safe_json_loads(_safe_text(response.content))
                matches = parsed.get("matches", [])
                results: list[dict[str, Any]] = []
                if isinstance(matches, list):
                    for item in matches:
                        if not isinstance(item, dict):
                            continue
                        try:
                            results.append(_validate_job_match(item, job_map))
                        except LLMCallError:
                            continue

                if len(results) > len(best_results):
                    best_results = results
                if len(results) >= expected_count:
                    return results
            except Exception as exc:
                logger.warning(
                    "岗位 AI 精排第 %s/%s 次调用失败: %s",
                    attempt,
                    max_attempts,
                    exc,
                )

            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 4))

        return best_results


async def calculate_ai_job_match_async(
    student_data: dict[str, Any],
    job_records: list[Any],
    assessment: dict[str, Any] | None = None,
    resume_text: str | None = None,
    top_n: int = 5,
    batch_size: int = 12,
    max_concurrency: int = 3,
) -> list[dict[str, Any]]:
    """
    异步版 AI 双向岗位匹配。

    核心优化：
    - 原来多个 batch 串行执行：第 1 批完成后才开始第 2 批；
    - 现在使用 asyncio.gather 并发执行多个 batch；
    - max_concurrency 控制并发上限，避免接口限流。
    """
    llm = _create_llm()
    if llm is None or not job_records:
        return []

    assessment = assessment or score_four_dimensions_llm(student_data, resume_text)
    resume_context = build_resume_context(student_data, resume_text)
    jobs = [record_to_job_dict(record) for record in job_records]
    batches = _prepare_job_batches(jobs, batch_size)

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    tasks = []
    for job_map, compact_jobs in batches:
        prompt = _build_job_match_prompt(student_data, assessment, resume_context, compact_jobs)
        tasks.append(_invoke_match_batch_async(llm, prompt, job_map, semaphore))

    nested_results = await asyncio.gather(*tasks)
    all_results = [item for batch_result in nested_results for item in batch_result]
    return diversify_job_matches(all_results, top_n=top_n)


def _run_async_safely(coro: Any) -> Any:
    """
    在同步函数中安全运行异步任务。

    当前 main.py 是同步调用 calculate_job_match()，所以这里保留同步入口。
    如果以后把 FastAPI 路由改成 async，可以直接 await calculate_ai_job_match_async()。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # 极少数情况下，如果当前线程已经存在事件循环，则退回新线程执行，避免 RuntimeError。
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        return future.result()

def calculate_ai_job_match(
    student_data: dict[str, Any],
    job_records: list[Any],
    assessment: dict[str, Any] | None = None,
    resume_text: str | None = None,
    top_n: int = 5,
    batch_size: int = 12,
) -> list[dict[str, Any]]:
    """
    同步兼容入口：内部调用异步版 calculate_ai_job_match_async()。

    这样 main.py 不需要改动，原来的调用方式仍然有效；
    但大模型批处理已经从串行调用优化为异步并发调用。
    """
    max_concurrency = int(os.getenv("JOB_MATCH_MAX_CONCURRENCY", "3"))
    return _run_async_safely(calculate_ai_job_match_async(
        student_data=student_data,
        job_records=job_records,
        assessment=assessment,
        resume_text=resume_text,
        top_n=top_n,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    ))


def calculate_local_job_match(
    student_data: dict[str, Any],
    job_records: list[Any],
    assessment: dict[str, Any] | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    本地即时岗位匹配。

    从较大的向量召回池中选出 30 个唯一且方向分散的岗位，再使用
    技能覆盖、项目证据、画像分数、目标方向和学历要求完成本地排序。
    此流程不会创建 LLM 客户端。
    """
    if not job_records:
        return []

    assessment = assessment or score_four_dimensions(student_data)
    candidate_records = job_records

    if retrieve_jobs_by_vector is not None:
        try:
            recalled_records = retrieve_jobs_by_vector(
                student_data=student_data,
                job_records=job_records,
                top_k=300,
                min_keep=12,
            )
            candidate_records = diversify_candidate_records(
                recalled_records,
                top_k=30,
                target_job=student_data.get("target_job"),
            )
        except Exception:
            candidate_records = job_records

    if (not candidate_records or candidate_records == job_records) and prefilter_job_records is not None:
        try:
            recalled_records = prefilter_job_records(
                student_data=student_data,
                job_records=job_records,
                top_k=60,
            )
            candidate_records = diversify_candidate_records(
                recalled_records,
                top_k=30,
                target_job=student_data.get("target_job"),
            )
        except Exception:
            candidate_records = job_records

    local_results: list[dict[str, Any]] = []
    for record in candidate_records:
        result = match_profile_to_job(student_data, record, assessment=assessment)
        result["used_llm"] = False
        result["match_source"] = "local"
        result["student_to_job_score"] = result["match_score"]
        result["job_to_student_score"] = result["match_score"]
        local_results.append(result)

    return diversify_job_matches(
        local_results,
        top_n=top_n,
        min_diversity_score=40,
        max_score_gap=25,
    )


def refine_job_matches_with_llm(
    student_data: dict[str, Any],
    local_matches: list[dict[str, Any]],
    assessment: dict[str, Any],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    仅把本地 TOP10 一次性发送给 LLM 精排。

    调用失败、超时或返回为空时直接返回本地结果，页面不会被长期阻塞。
    """
    candidates = local_matches[:10]
    if not candidates:
        return []

    refined = calculate_ai_job_match(
        student_data,
        candidates,
        assessment=assessment,
        top_n=min(top_n, len(candidates)),
        batch_size=max(1, len(candidates)),
    )
    if not refined:
        raise LLMCallError()

    refined_titles = {
        normalize_job_title(item.get("job_name"))
        for item in refined
    }
    for candidate in candidates:
        if normalize_job_title(candidate.get("job_name")) not in refined_titles:
            fallback = dict(candidate)
            fallback["match_source"] = "local_fallback"
            refined.append(fallback)

    for item in refined:
        if item.get("used_llm"):
            item["match_source"] = "llm"
    return diversify_job_matches(refined, top_n=min(top_n, len(candidates)))


# 保留旧调用名，但默认改为本地即时匹配。
def calculate_job_match(
    student_data: dict[str, Any],
    job_records: list[Any],
    assessment: dict[str, Any] | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    return calculate_local_job_match(
        student_data=student_data,
        job_records=job_records,
        assessment=assessment,
        top_n=top_n,
    )
