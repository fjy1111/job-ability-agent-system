from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

"""
岗位向量检索服务。

放置位置：
    app/services/job_vector_service.py

作用：
    在大模型双向精排前，先把学生简历画像和岗位文本转换为向量，
    用余弦相似度召回 top_k 个候选岗位，减少进入大模型的岗位数量。

说明：
    当前版本采用“本地轻量词袋向量 + 余弦相似度”，不依赖 FAISS、Chroma、Redis、数据库扩展。
    优点是可以直接放进你当前项目运行；后续如果要升级为真正的 embedding 模型，
    只需要替换 build_text_vector() 和 cosine_similarity() 前后的向量生成逻辑即可。
"""

# 常见 IT 技术词，用于弥补中文没有分词库时的召回问题。
TECH_KEYWORDS = [
    "java", "spring", "spring boot", "springcloud", "mybatis", "mysql", "redis",
    "python", "fastapi", "django", "flask", "爬虫", "数据分析", "机器学习", "深度学习",
    "vue", "react", "javascript", "typescript", "html", "css", "前端", "后端",
    "docker", "kubernetes", "k8s", "linux", "nginx", "git", "maven", "rabbitmq",
    "微服务", "分布式", "高并发", "缓存", "接口", "数据库", "算法", "大模型",
    "nlp", "cv", "pytorch", "tensorflow", "测试", "自动化测试", "运维", "云计算",
]

ROLE_EXPANSION: dict[str, list[str]] = {
    "java": ["java", "spring", "spring boot", "mybatis", "mysql", "redis", "后端", "接口", "微服务"],
    "后端": ["后端", "java", "python", "go", "spring", "fastapi", "数据库", "接口", "redis"],
    "前端": ["前端", "vue", "react", "javascript", "typescript", "html", "css", "小程序"],
    "算法": ["算法", "机器学习", "深度学习", "大模型", "nlp", "cv", "pytorch", "tensorflow"],
    "测试": ["测试", "自动化测试", "接口测试", "性能测试", "selenium", "postman"],
    "运维": ["运维", "linux", "docker", "kubernetes", "k8s", "nginx", "devops", "云计算"],
    "数据": ["数据", "sql", "mysql", "数据分析", "数仓", "hadoop", "spark", "bi"],
}

# 简单内存索引：同一批岗位重复检索时不必重复构建岗位向量。
_VECTOR_INDEX_CACHE: dict[str, list[dict[str, Any]]] = {}


def _safe_text(value: Any) -> str:
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


def _record_id(record: Any, fallback: int) -> str:
    value = _get_field(record, "id", "job_id")
    return _safe_text(value).strip() or f"idx_{fallback}"


def build_student_retrieval_text(student_data: dict[str, Any]) -> str:
    """拼接学生检索文本：目标岗位、专业、技能、项目、证书、简历全文。"""
    fields = [
        "target_job", "major", "grade", "education", "skills", "projects", "internships",
        "competitions", "certificates", "self_intro", "resume_text", "normalized_text", "experience",
    ]
    text = " ".join(_safe_text(student_data.get(field)) for field in fields)

    # 根据目标岗位扩展同义/相关技术词，提升召回稳定性。
    target = _safe_text(student_data.get("target_job")).lower()
    expanded: list[str] = []
    for role, words in ROLE_EXPANSION.items():
        if role in target:
            expanded.extend(words)
    if expanded:
        text = text + " " + " ".join(expanded)
    return text


def build_job_retrieval_text(record: Any) -> str:
    """拼接岗位检索文本：岗位名、公司、技能要求、项目、课程、证书、描述等。"""
    values = [
        _get_field(record, "job_name", "name"),
        _get_field(record, "job_category", "category"),
        _get_field(record, "company_name"),
        _get_field(record, "hiring_city"),
        _get_field(record, "educational_requirements", "education_requirement", "education"),
        _get_field(record, "experience_requirements", "experience_requirement"),
        _get_field(record, "required_skills", "required_skills_json", "skills", "skill_tags"),
        _get_field(record, "required_abilities", "abilities"),
        _get_field(record, "related_projects", "related_projects_json", "projects"),
        _get_field(record, "related_courses", "recommended_courses", "recommended_courses_json", "courses"),
        _get_field(record, "recommended_certificates", "recommended_certificates_json", "certificates"),
        _get_field(record, "career_path"),
        _get_field(record, "salary_range"),
        _get_field(record, "description", "job_description"),
    ]
    return " ".join(_safe_text(v) for v in values)


def tokenize(text: str) -> list[str]:
    """
    轻量分词：
    - 英文、数字按单词切分；
    - 中文连续片段按 2-4 字 ngram 切分；
    - 额外识别常见 IT 技术词。
    """
    text = _safe_text(text).lower()
    tokens: list[str] = []

    # 英文/数字技术词。
    tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9_#+.\-]{1,}|\d+", text))

    # 中文片段 ngram。
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for chunk in chinese_chunks:
        max_n = min(4, len(chunk))
        for n in range(2, max_n + 1):
            for i in range(0, len(chunk) - n + 1):
                tokens.append(chunk[i:i + n])

    compact_text = re.sub(r"\s+", " ", text)
    for keyword in TECH_KEYWORDS:
        if keyword.lower() in compact_text:
            # 技术词提高权重：加入两次。
            tokens.append(keyword.lower())
            tokens.append(keyword.lower())

    return [token.strip() for token in tokens if len(token.strip()) >= 2]


def build_text_vector(text: str) -> dict[str, float]:
    """把文本转换为稀疏词频向量。"""
    counts = Counter(tokenize(text))
    if not counts:
        return {}

    # 使用 1 + log(tf) 缓解超长文本词频过高问题。
    return {token: 1.0 + math.log(freq) for token, freq in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """计算两个稀疏向量的余弦相似度。"""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot = sum(weight * vec_b.get(token, 0.0) for token, weight in vec_a.items())
    norm_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jobs_signature(job_records: list[Any]) -> str:
    """为岗位列表生成签名，用于复用内存向量索引。"""
    payload = []
    for idx, record in enumerate(job_records or []):
        payload.append({
            "id": _record_id(record, idx),
            "job_name": _safe_text(_get_field(record, "job_name", "name")),
            "skills": _safe_text(_get_field(record, "required_skills", "required_skills_json", "skills", "skill_tags")),
            "description": _safe_text(_get_field(record, "description", "job_description")),
        })
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_job_vector_index(job_records: list[Any]) -> list[dict[str, Any]]:
    """
    构建岗位向量索引。

    返回结构：
    [
        {"record": 原岗位对象, "text": 岗位文本, "vector": 稀疏向量},
        ...
    ]
    """
    if not job_records:
        return []

    signature = _jobs_signature(job_records)
    cached = _VECTOR_INDEX_CACHE.get(signature)
    if cached is not None:
        return cached

    index: list[dict[str, Any]] = []
    for record in job_records:
        text = build_job_retrieval_text(record)
        index.append({
            "record": record,
            "text": text,
            "vector": build_text_vector(text),
        })

    # 简单控制缓存数量，避免长期运行时内存增长。
    if len(_VECTOR_INDEX_CACHE) >= 8:
        _VECTOR_INDEX_CACHE.clear()
    _VECTOR_INDEX_CACHE[signature] = index
    return index


def retrieve_jobs_by_vector(
    student_data: dict[str, Any],
    job_records: list[Any],
    top_k: int = 60,
    min_keep: int = 12,
) -> list[Any]:
    """
    基于向量相似度召回候选岗位。

    参数：
        student_data: 学生表单/简历信息。
        job_records: 数据库中读取的全部岗位。
        top_k: 最多召回多少个岗位进入大模型精排。
        min_keep: 最少保留数量，避免相似度过低导致无候选。

    返回：
        候选岗位对象列表，顺序按向量相似度从高到低。
    """
    if not job_records:
        return []
    if len(job_records) <= top_k:
        return list(job_records)

    student_text = build_student_retrieval_text(student_data)
    student_vector = build_text_vector(student_text)
    index = build_job_vector_index(job_records)

    scored: list[tuple[float, int, Any]] = []
    for idx, item in enumerate(index):
        score = cosine_similarity(student_vector, item.get("vector", {}))
        scored.append((score, idx, item["record"]))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected = [record for score, _, record in scored[:top_k] if score > 0]

    # 保底策略：如果文本太少导致相似度为 0，则仍返回 top_k 条，保证页面有结果。
    if len(selected) < min_keep:
        existing_ids = {id(record) for record in selected}
        for _, _, record in scored:
            if id(record) not in existing_ids:
                selected.append(record)
                existing_ids.add(id(record))
            if len(selected) >= min(top_k, len(job_records)):
                break

    return selected[:top_k]


def clear_job_vector_index_cache() -> None:
    """岗位库更新后可手动清空向量索引缓存。"""
    _VECTOR_INDEX_CACHE.clear()


def get_job_vector_index_stats() -> dict[str, Any]:
    """查看向量索引缓存状态，调试时可用。"""
    return {
        "cached_index_count": len(_VECTOR_INDEX_CACHE),
        "retrieval_type": "local_sparse_vector_cosine",
        "tech_keywords_count": len(TECH_KEYWORDS),
    }
