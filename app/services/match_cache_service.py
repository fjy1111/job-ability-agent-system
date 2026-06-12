from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

"""
岗位匹配缓存服务。

放置位置：
    app/services/match_cache_service.py

作用：
    1. 为“学生数据 + 候选岗位 + 模型版本 + 算法版本”生成稳定缓存键；
    2. 缓存岗位匹配 TOP5 结果；
    3. 用户重复刷新岗位匹配页面时，命中缓存后不再重复调用大模型。

说明：
    当前版本使用内存字典缓存，不需要安装 Redis，也不需要修改数据库。
    项目重启后缓存会清空，适合开发、调试、比赛演示阶段使用。
"""

# 缓存版本号：当 prompt、评分公式、候选岗位召回策略发生变化时，手动改这个值即可让旧缓存失效。
MATCH_CACHE_ALGORITHM_VERSION = "match_cache_v7_reliable_ai_refine"

# 默认缓存有效期：30 分钟。比赛演示时，重复刷新页面基本都会命中缓存。
DEFAULT_TTL_SECONDS = 30 * 60

_MATCH_CACHE: dict[str, dict[str, Any]] = {}


def _safe_text(value: Any) -> str:
    """把任意值安全转换成字符串，避免 None、list、dict 影响哈希生成。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_safe_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _normalize_student_data(student_data: dict[str, Any]) -> dict[str, str]:
    """只选取会影响岗位匹配结果的学生字段参与缓存键生成。"""
    important_fields = [
        "name",
        "major",
        "grade",
        "target_job",
        "skills",
        "projects",
        "competitions",
        "certificates",
        "self_intro",
    ]
    return {
        field: _safe_text(student_data.get(field)).strip()
        for field in important_fields
    }


def _record_to_cache_dict(record: Any) -> dict[str, str]:
    """把岗位记录转换成稳定的简化字典，用于生成岗位库版本哈希。"""
    fields = [
        "id",
        "job_name",
        "job_category",
        "required_skills",
        "required_skills_json",
        "required_abilities",
        "related_courses",
        "recommended_courses",
        "recommended_courses_json",
        "related_projects",
        "related_projects_json",
        "recommended_certificates",
        "recommended_certificates_json",
        "educational_requirements",
        "experience_requirements",
        "career_path",
        "salary_range",
    ]

    result: dict[str, str] = {}
    for field in fields:
        result[field] = _safe_text(getattr(record, field, "")).strip()
    return result


def build_job_version(job_records: list[Any]) -> str:
    """
    根据候选岗位列表生成岗位版本号。

    候选岗位数量变化、岗位技能要求变化、课程/证书/项目字段变化时，
    这个版本号都会变化，因此不会误用旧缓存。
    """
    normalized_jobs = [_record_to_cache_dict(record) for record in job_records]
    normalized_jobs.sort(key=lambda item: (item.get("id", ""), item.get("job_name", "")))

    raw_text = json.dumps(normalized_jobs, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def build_match_cache_key(
    student_data: dict[str, Any],
    job_records: list[Any],
    model_name: str = "qwen-plus",
    algorithm_version: str = MATCH_CACHE_ALGORITHM_VERSION,
) -> str:
    """
    生成岗位匹配缓存键。

    缓存键组成：
        学生关键信息哈希 + 候选岗位列表哈希 + 模型名 + 算法版本
    """
    payload = {
        "student": _normalize_student_data(student_data),
        "job_version": build_job_version(job_records),
        "model_name": model_name,
        "algorithm_version": algorithm_version,
    }
    raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def get_match_cache(cache_key: str) -> list[dict[str, Any]] | None:
    """读取缓存。命中且未过期返回结果，否则返回 None。"""
    item = _MATCH_CACHE.get(cache_key)
    if not item:
        return None

    now = time.time()
    expires_at = float(item.get("expires_at", 0))
    if expires_at <= now:
        _MATCH_CACHE.pop(cache_key, None)
        return None

    value = item.get("value")
    if not isinstance(value, list):
        return None

    return copy.deepcopy(value)


def set_match_cache(
    cache_key: str,
    value: list[dict[str, Any]],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """写入缓存。"""
    if not cache_key or not isinstance(value, list):
        return

    now = time.time()
    _MATCH_CACHE[cache_key] = {
        "value": copy.deepcopy(value),
        "created_at": now,
        "expires_at": now + ttl_seconds,
    }


def clear_match_cache() -> None:
    """清空全部岗位匹配缓存。"""
    _MATCH_CACHE.clear()


def get_match_cache_stats() -> dict[str, Any]:
    """查看缓存状态，方便调试。"""
    now = time.time()
    valid_count = 0
    expired_count = 0

    for item in _MATCH_CACHE.values():
        if float(item.get("expires_at", 0)) > now:
            valid_count += 1
        else:
            expired_count += 1

    return {
        "total": len(_MATCH_CACHE),
        "valid": valid_count,
        "expired": expired_count,
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "algorithm_version": MATCH_CACHE_ALGORITHM_VERSION,
    }
