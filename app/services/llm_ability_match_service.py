from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

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


def _create_llm() -> ChatOpenAI | None:
    """创建兼容 OpenAI 接口的大模型客户端。

    兼容两套 .env 命名：
    - 通用：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
    - DeepSeek：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL / ABILITY_MATCH_MODEL
    """
    if os.getenv("USE_LLM", "true").lower() != "true":
        return None

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        return None

    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )
    model = (
        os.getenv("ABILITY_MATCH_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )

    kwargs: dict[str, Any] = {"model": model, "temperature": 0.1, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


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
    if llm is None:
        return _validate_assessment({
            "ability_scores": {k: 0 for k in DIMENSION_KEYS},
            "score_evidence": {k: [] for k in DIMENSION_KEYS},
            "recognized_skills": [],
            "target_roles": [],
            "assessment_summary": "未配置大模型 API，无法进行 AI 语义评分。",
            "used_llm": False,
            "agent_warning": "请在 .env 中配置 LLM_API_KEY/DEEPSEEK_API_KEY，并设置 USE_LLM=true。",
        })

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
        parsed["used_llm"] = True
        return _validate_assessment(parsed)
    except Exception as exc:
        return _validate_assessment({
            "ability_scores": {k: 0 for k in DIMENSION_KEYS},
            "score_evidence": {k: [] for k in DIMENSION_KEYS},
            "assessment_summary": "大模型调用失败，未生成 AI 评分。",
            "used_llm": False,
            "agent_warning": f"{type(exc).__name__}: {exc}",
        })


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

    student_to_job_score = _clamp_score(
        item.get("student_to_job_score")
        or item.get("student_fit_score")
        or item.get("resume_to_job_score")
        or score_components.get("resume_job_fit")
    )
    job_to_student_score = _clamp_score(
        item.get("job_to_student_score")
        or item.get("growth_path_score")
        or item.get("job_growth_score")
        or growth_components.get("career_consistency")
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
        recommend_reason = bidirectional_summary

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
        "matched_skills": _dedupe(_listify(item.get("matched_skills"))),
        "missing_skills": _dedupe(_listify(item.get("missing_skills"))),
        "skill_gaps": _dedupe(_listify(item.get("skill_gaps") or item.get("missing_skills")))[:8],
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


def calculate_ai_job_match(
    student_data: dict[str, Any],
    job_records: list[Any],
    assessment: dict[str, Any] | None = None,
    resume_text: str | None = None,
    top_n: int = 5,
    batch_size: int = 12,
) -> list[dict[str, Any]]:
    """
    AI 大模型版双向岗位匹配。

    双向含义：
    1. 学生适岗分 student_to_job_score：学生当前能力是否满足岗位要求。
    2. 岗位适生分 job_to_student_score：岗位是否适合学生下一阶段成长路径。

    最终匹配分：
    match_score = student_to_job_score * 0.6 + job_to_student_score * 0.4
    """
    llm = _create_llm()
    if llm is None or not job_records:
        return []

    assessment = assessment or score_four_dimensions_llm(student_data, resume_text)
    resume_context = build_resume_context(student_data, resume_text)
    all_results: list[dict[str, Any]] = []

    jobs = [record_to_job_dict(record) for record in job_records]
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        job_map: dict[str, dict[str, Any]] = {}
        compact_jobs = []
        for offset, job in enumerate(batch):
            job_id = f"job_{start + offset + 1}"
            job_map[job_id] = job
            compact_jobs.append({"job_id": job_id, **job})

        prompt = f"""
你是“学生成长诊断与路径规划智能体”的双向岗位匹配专家。请不要按关键词覆盖率机械计算，而要基于简历语义、四维能力画像和岗位数据进行判断。

请对每个岗位同时计算两个方向的分数：

一、学生适岗分 student_to_job_score，0-100：
回答“学生当前能力是否满足岗位要求”。重点看：
1. resume_job_fit：简历技能与岗位技能要求的匹配程度；
2. project_relevance：项目经历与岗位业务/技术场景的相关性；
3. skill_depth：技能掌握深度，区分“了解、熟悉、熟练、实际使用”；
4. education_and_experience_fit：学历、专业、年级、经验与岗位要求是否匹配；
5. career_direction_fit：目标岗位与简历主线是否一致。

二、岗位适生分 job_to_student_score，0-100：
回答“这个岗位是否适合作为学生下一阶段成长目标”。重点看：
1. career_consistency：职业方向一致性，岗位方向是否与学生目标和能力主线一致；
2. gap_fillability：缺口可补齐性，缺失技能能否通过课程、项目、证书、训练任务补齐；
3. difficulty_suitability：岗位难度适配性，是否既不过难也不过易，适合当前阶段；
4. learning_path_mappability：课程/项目/证书可映射性，是否能拆成清晰成长路径；
5. growth_value：岗位成长价值，是否能推动学生形成持续能力提升。

最终匹配分必须按下面公式计算：
match_score = student_to_job_score * 0.6 + job_to_student_score * 0.4

评分要求：
- 分数必须是 0-100 整数。
- 80以上表示高度匹配，60-79表示基本匹配，40-59表示存在明显短板，40以下表示不建议优先投递。
- 不能编造简历没有体现的技能或项目。
- 必须返回本批次所有岗位的评分结果。
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
      "matched_skills": ["简历能支撑的匹配技能"],
      "missing_skills": ["岗位要求但简历证据不足的技能"],
      "skill_gaps": ["最建议补齐的短板"],
      "matched_projects": ["与岗位相关的项目证据"],
      "matched_certificates": ["匹配证书"],
      "score_components": {{
        "resume_job_fit": 0,
        "project_relevance": 0,
        "skill_depth": 0,
        "education_and_experience_fit": 0,
        "career_direction_fit": 0
      }},
      "growth_components": {{
        "career_consistency": 0,
        "gap_fillability": 0,
        "difficulty_suitability": 0,
        "learning_path_mappability": 0,
        "growth_value": 0
      }},
      "growth_path_reason": "说明岗位为什么适合或不适合学生成长路径，60字以内",
      "bidirectional_summary": "说明学生适岗分和岗位适生分的综合判断，60字以内",
      "recommend_reason": "推荐或不推荐的原因，60字以内"
    }}
  ]
}}
"""
        try:
            response = llm.invoke(prompt)
            parsed = _safe_json_loads(_safe_text(response.content))
            matches = parsed.get("matches", [])
            if isinstance(matches, list):
                for item in matches:
                    if isinstance(item, dict):
                        all_results.append(_validate_job_match(item, job_map))
        except Exception:
            continue

    all_results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return all_results[:top_n]


# 为了少改原项目调用点，提供与旧服务相同的函数名。
def calculate_job_match(student_data: dict[str, Any], job_records: list[Any]) -> list[dict[str, Any]]:
    assessment = score_four_dimensions_llm(student_data)
    return calculate_ai_job_match(student_data, job_records, assessment=assessment, top_n=5)
