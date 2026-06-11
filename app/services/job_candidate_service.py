from __future__ import annotations

import json
import re
from typing import Any


def _safe_text(value: Any) -> str:
    """把任意字段转成可检索文本。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_safe_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(f"{k} {_safe_text(v)}" for k, v in value.items())
    return str(value)


def _get_field(record: Any, *names: str) -> Any:
    """兼容 SQLAlchemy 对象和 dict 两种岗位记录。"""
    for name in names:
        if isinstance(record, dict) and name in record:
            return record.get(name)
        if hasattr(record, name):
            return getattr(record, name)
    return ""


def _json_or_text_list(value: Any) -> list[str]:
    """把 JSON 字符串、列表或普通文本拆成关键词列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(v).strip() for v in data if str(v).strip()]
        if isinstance(data, dict):
            return [str(v).strip() for v in data.values() if str(v).strip()]
    except Exception:
        pass
    parts = re.split(r"[,，、/|;；\n\r\t]+", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_token(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


# 常见 IT 方向关键词，用于把“Java 开发”“后端开发”“前端”等目标转成更稳定的召回词。
ROLE_KEYWORDS: dict[str, list[str]] = {
    "java": ["java", "spring", "springboot", "spring cloud", "mybatis", "后端", "软件开发"],
    "后端": ["后端", "java", "python", "go", "spring", "接口", "服务端", "数据库", "redis"],
    "python": ["python", "django", "flask", "fastapi", "后端", "数据分析"],
    "前端": ["前端", "vue", "react", "javascript", "typescript", "html", "css", "小程序"],
    "算法": ["算法", "机器学习", "深度学习", "大模型", "nlp", "cv", "pytorch", "tensorflow"],
    "测试": ["测试", "自动化测试", "软件测试", "接口测试", "性能测试", "selenium"],
    "运维": ["运维", "linux", "docker", "kubernetes", "k8s", "nginx", "云计算", "devops"],
    "数据": ["数据", "sql", "mysql", "数据分析", "数仓", "hadoop", "spark", "bi"],
}


def _extract_student_text(student_data: dict[str, Any]) -> str:
    fields = [
        "target_job", "major", "grade", "skills", "projects", "competitions",
        "certificates", "self_intro", "resume_text", "education", "experience",
    ]
    return " ".join(_safe_text(student_data.get(f)) for f in fields)


def _extract_student_terms(student_data: dict[str, Any]) -> set[str]:
    """从学生数据中提取目标岗位、技能、项目里的轻量召回关键词。"""
    text = _extract_student_text(student_data)
    raw_terms = re.split(r"[,，、/|;；\n\r\t\s]+", text)
    terms = {t.strip() for t in raw_terms if len(t.strip()) >= 2}

    target = _safe_text(student_data.get("target_job"))
    target_norm = _normalize_token(target)
    for role, keywords in ROLE_KEYWORDS.items():
        if role in target_norm or any(_normalize_token(k) in target_norm for k in keywords):
            terms.update(keywords)

    # 加入一些简历文本中常见的技术词，避免中文分词不充分。
    common_skills = [
        "Java", "Spring", "Spring Boot", "SpringCloud", "MyBatis", "MySQL", "Redis",
        "Docker", "Linux", "Git", "Maven", "Nginx", "RabbitMQ", "Vue", "React",
        "Python", "FastAPI", "Django", "Flask", "JavaScript", "TypeScript", "Kubernetes",
        "微服务", "分布式", "高并发", "接口", "数据库", "缓存", "后端", "前端", "算法",
    ]
    lower_text = text.lower()
    for skill in common_skills:
        if skill.lower() in lower_text:
            terms.add(skill)

    return {_normalize_token(t) for t in terms if _normalize_token(t)}


def _build_job_text(record: Any) -> str:
    """拼接岗位核心字段，用于预筛选文本匹配。"""
    values = [
        _get_field(record, "job_name"),
        _get_field(record, "company_name"),
        _get_field(record, "hiring_city"),
        _get_field(record, "educational_requirements", "education_requirement"),
        _get_field(record, "required_skills", "required_skills_json", "skills", "skill_tags"),
        _get_field(record, "related_projects", "related_projects_json"),
        _get_field(record, "recommended_courses", "recommended_courses_json"),
        _get_field(record, "recommended_certificates", "recommended_certificates_json"),
        _get_field(record, "salary_range"),
        _get_field(record, "description", "job_description"),
    ]
    return " ".join(_safe_text(v) for v in values)


def _job_required_terms(record: Any) -> set[str]:
    skills = _json_or_text_list(_get_field(record, "required_skills", "required_skills_json", "skills", "skill_tags"))
    projects = _json_or_text_list(_get_field(record, "related_projects", "related_projects_json"))
    words = skills + projects
    return {_normalize_token(w) for w in words if _normalize_token(w)}


def _score_job(student_terms: set[str], student_text_norm: str, record: Any) -> float:
    """轻量候选岗位预筛分。分数只用于召回候选，不作为最终匹配分。"""
    job_text = _build_job_text(record)
    job_text_norm = _normalize_token(job_text)
    job_name_norm = _normalize_token(_safe_text(_get_field(record, "job_name")))
    required_terms = _job_required_terms(record)

    score = 0.0

    # 1. 目标岗位/岗位名称匹配权重较高。
    target_terms = [t for t in student_terms if t in job_name_norm]
    score += min(len(target_terms), 5) * 8.0

    # 2. 学生关键词与岗位全文命中。
    text_hits = [t for t in student_terms if len(t) >= 2 and t in job_text_norm]
    score += min(len(text_hits), 12) * 3.0

    # 3. 岗位 required_skills 与学生文本重合。
    skill_hits = [t for t in required_terms if len(t) >= 2 and t in student_text_norm]
    if required_terms:
        score += (len(skill_hits) / max(len(required_terms), 1)) * 45.0
        score += min(len(skill_hits), 10) * 4.0

    # 4. 岗位方向加分：避免 Java 简历被前端、算法岗位抢占。
    for role, keywords in ROLE_KEYWORDS.items():
        role_norm = _normalize_token(role)
        if role_norm in student_text_norm:
            for kw in keywords:
                kw_norm = _normalize_token(kw)
                if kw_norm and kw_norm in job_text_norm:
                    score += 5.0

    # 5. 基础保底：岗位名称与学生文本有直接包含。
    if job_name_norm and job_name_norm in student_text_norm:
        score += 20.0

    return score


def prefilter_job_records(
    student_data: dict[str, Any],
    job_records: list[Any],
    top_k: int = 60,
    min_keep: int = 12,
) -> list[Any]:
    """
    候选岗位预筛选。先用轻量文本/技能匹配把全部岗位缩小到 top_k，再交给大模型双向精排。

    设计目的：
    - 原来 674 条岗位全部进入大模型，batch_size=12 时约 57 批；
    - 预筛选 top_k=60 后约 5 批，响应时间明显降低；
    - 预筛选只做“召回”，最终分数仍由大模型双向匹配计算。
    """
    if not job_records:
        return []
    if len(job_records) <= top_k:
        return list(job_records)

    student_terms = _extract_student_terms(student_data)
    student_text_norm = _normalize_token(_extract_student_text(student_data))

    scored: list[tuple[float, int, Any]] = []
    for idx, record in enumerate(job_records):
        score = _score_job(student_terms, student_text_norm, record)
        scored.append((score, idx, record))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected = [record for score, _, record in scored[:top_k] if score > 0]

    # 保底策略：如果命中的岗位太少，补充前 top_k 条，避免无推荐结果。
    if len(selected) < min_keep:
        existing_ids = {id(r) for r in selected}
        for _, _, record in scored:
            if id(record) not in existing_ids:
                selected.append(record)
                existing_ids.add(id(record))
            if len(selected) >= min(top_k, len(job_records)):
                break

    return selected[:top_k]
