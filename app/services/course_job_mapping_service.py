from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any


COURSE_ALIASES: dict[str, list[str]] = {
    "数据库应用": ["数据库应用", "数据库原理", "数据库系统概论", "数据库系统", "SQL数据库", "MySQL数据库应用", "数据库技术"],
    "Java程序设计": ["Java程序设计", "Java开发基础", "Java语言程序设计", "Java编程", "面向对象程序设计"],
    "Python程序设计": ["Python程序设计", "Python编程", "Python基础", "Python语言程序设计"],
    "数据结构": ["数据结构", "数据结构与算法", "算法与数据结构"],
    "计算机网络": ["计算机网络", "网络协议", "TCP/IP", "网络基础"],
    "操作系统": ["操作系统", "计算机操作系统", "Linux操作系统", "Linux系统管理"],
    "软件工程": ["软件工程", "软件项目管理", "软件开发生命周期", "系统分析与设计"],
    "Web前端开发": ["Web前端开发", "前端开发", "HTML5与CSS3", "JavaScript程序设计", "Vue开发", "React开发"],
    "人工智能导论": ["人工智能导论", "人工智能基础", "AI基础"],
    "机器学习": ["机器学习", "机器学习导论", "模式识别", "数据挖掘"],
    "深度学习": ["深度学习", "神经网络", "PyTorch", "TensorFlow"],
    "大数据技术": ["大数据技术", "大数据技术基础", "Hadoop生态", "Spark数据处理", "Flink实时计算"],
    "云计算": ["云计算", "云计算基础", "容器技术", "Docker与Kubernetes"],
    "网络安全": ["网络安全", "信息安全", "Web安全", "安全攻防", "漏洞扫描"],
    "软件测试": ["软件测试", "软件测试技术", "自动化测试", "接口测试"],
    "数据分析": ["数据分析", "Excel数据分析", "统计学基础", "概率论与数理统计", "数据可视化"],
    "C#程序设计": ["C#程序设计", "C#开发", "C#编程", "C＃程序设计", "CSharp", "C Sharp", ".NET程序设计"],
    "计算机图形学": ["计算机图形学", "图形学", "图形渲染", "OpenGL", "WebGL", "三维图形"],
    "地理信息系统开发": ["地理信息系统设计与开发", "地理信息系统", "GIS开发", "GIS", "地信开发", "地图开发"],
    "ArcGIS应用与开发": ["ArcGIS应用与开发", "ArcGIS开发", "ArcGIS", "GIS应用开发"],
}


COURSE_ABILITY_MAP: dict[str, list[str]] = {
    "数据库应用": ["SQL", "MySQL", "数据库设计", "索引优化", "事务", "数据建模"],
    "Java程序设计": ["Java", "面向对象", "Spring Boot", "后端开发", "接口开发"],
    "Python程序设计": ["Python", "脚本开发", "数据处理", "后端开发", "自动化"],
    "数据结构": ["数据结构", "算法", "复杂度分析", "编程基础"],
    "计算机网络": ["HTTP", "TCP/IP", "网络协议", "接口通信", "网络安全基础"],
    "操作系统": ["Linux", "进程线程", "系统部署", "Shell", "服务器运维"],
    "软件工程": ["需求分析", "系统设计", "项目管理", "测试流程", "文档能力"],
    "Web前端开发": ["HTML", "CSS", "JavaScript", "Vue", "React", "前端工程化"],
    "人工智能导论": ["人工智能", "机器学习基础", "模型应用", "AI应用场景"],
    "机器学习": ["机器学习", "特征工程", "模型训练", "数据挖掘", "算法建模"],
    "深度学习": ["深度学习", "PyTorch", "TensorFlow", "神经网络", "模型优化"],
    "大数据技术": ["Hadoop", "Hive", "Spark", "Flink", "Kafka", "数据仓库"],
    "云计算": ["云服务器", "Docker", "Kubernetes", "DevOps", "CI/CD", "云平台"],
    "网络安全": ["信息安全", "Web安全", "漏洞扫描", "安全加固", "应急响应"],
    "软件测试": ["软件测试", "接口测试", "自动化测试", "测试用例", "缺陷管理"],
    "数据分析": ["Excel", "Python", "Pandas", "SQL", "统计分析", "数据可视化"],
    "C#程序设计": ["C#", ".NET", "面向对象", "后端开发", "接口开发"],
    "计算机图形学": ["计算机图形学", "图形渲染", "OpenGL", "WebGL", "算法"],
    "地理信息系统开发": ["GIS", "ArcGIS", "空间数据", "地图开发", "数据可视化"],
    "ArcGIS应用与开发": ["ArcGIS", "GIS", "空间分析", "地图开发", "空间数据"],
}


ABILITY_ALIASES: dict[str, list[str]] = {
    "SQL": ["SQL", "数据库查询"],
    "MySQL": ["MySQL", "mysql"],
    "HTML": ["HTML", "HTML5"],
    "CSS": ["CSS", "CSS3"],
    "数据库设计": ["数据库设计", "表结构", "数据建模", "ER图"],
    "索引优化": ["索引", "索引优化", "SQL优化", "查询优化"],
    "事务": ["事务", "隔离级别", "锁机制"],
    "Java": ["Java", "java"],
    "面向对象": ["面向对象", "OOP"],
    "C#": ["C#", "C＃", "C Sharp", "CSharp", ".NET", "ASP.NET"],
    ".NET": [".NET", "dotnet", "ASP.NET", "C#"],
    "Spring Boot": ["Spring Boot", "SpringBoot", "Spring"],
    "Python": ["Python", "python"],
    "后端开发": ["后端", "后端开发", "服务端", "接口开发", "API"],
    "接口开发": ["接口", "API", "RESTful", "接口设计", "接口联调"],
    "Linux": ["Linux", "Unix", "Shell"],
    "Docker": ["Docker", "容器"],
    "Kubernetes": ["Kubernetes", "K8s", "k8s"],
    "JavaScript": ["JavaScript", "JS", "ECMAScript"],
    "Vue": ["Vue", "Vue.js", "Vue2", "Vue3"],
    "React": ["React", "React.js"],
    "前端工程化": ["前端工程化", "Webpack", "Vite", "组件化", "工程化"],
    "机器学习": ["机器学习", "ML", "模型训练"],
    "深度学习": ["深度学习", "神经网络", "DL"],
    "PyTorch": ["PyTorch", "torch"],
    "TensorFlow": ["TensorFlow", "tensorflow"],
    "Hadoop": ["Hadoop", "HDFS"],
    "Spark": ["Spark", "PySpark"],
    "Flink": ["Flink", "实时计算"],
    "Kafka": ["Kafka", "消息队列"],
    "数据仓库": ["数据仓库", "数仓", "Hive", "ETL"],
    "软件测试": ["软件测试", "功能测试", "测试用例"],
    "接口测试": ["接口测试", "Postman", "Apifox"],
    "自动化测试": ["自动化测试", "Selenium", "pytest"],
    "网络安全基础": ["网络安全", "信息安全", "安全基础"],
    "Web安全": ["Web安全", "渗透测试", "漏洞"],
    "数据可视化": ["数据可视化", "ECharts", "Tableau", "Power BI"],
    "计算机图形学": ["计算机图形学", "图形学", "三维图形"],
    "图形渲染": ["图形渲染", "渲染", "图形开发"],
    "OpenGL": ["OpenGL"],
    "WebGL": ["WebGL"],
    "GIS": ["GIS", "地理信息", "地理信息系统"],
    "ArcGIS": ["ArcGIS"],
    "空间数据": ["空间数据", "地理空间数据", "遥感数据"],
    "空间分析": ["空间分析", "地理分析"],
    "地图开发": ["地图开发", "地图服务", "WebGIS"],
}


COURSE_ROLE_HINTS: dict[str, list[str]] = {
    "Web前端开发": ["前端", "HTML", "CSS", "JavaScript", "TypeScript", "Vue", "React"],
    "C#程序设计": ["C#", "C＃", ".NET", "ASP.NET", "软件开发", "后端"],
    "计算机图形学": ["图形", "OpenGL", "WebGL", "渲染", "三维", "算法"],
    "地理信息系统开发": ["GIS", "ArcGIS", "地理信息", "地图", "遥感", "空间数据"],
    "ArcGIS应用与开发": ["ArcGIS", "GIS", "地理信息", "地图", "遥感", "空间分析"],
    "数据库应用": ["数据库", "SQL", "MySQL", "PostgreSQL"],
    "操作系统": ["Linux", "Shell", "运维", "服务器", "系统"],
    "计算机网络": ["网络", "TCP/IP", "HTTP", "安全", "协议"],
}


COURSE_SECTION_PATTERN = re.compile(
    r"(主要课程|相关课程|核心课程|专业课程|课程)[：:\s]*(?P<content>[^。；;\n]{2,260})",
    flags=re.IGNORECASE,
)

_JOB_INDEX_CACHE: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_safe_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_safe_text(item) for item in value.values())
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
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


def _split_items(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_split_items(item))
        return _dedupe(items)

    text = _safe_text(value).strip()
    if not text:
        return []

    try:
        loaded = json.loads(text)
        if loaded != text:
            return _split_items(loaded)
    except Exception:
        pass

    parts = re.split(r"[,，、;/；\n\r]+", text)
    return _dedupe([part.strip(" []【】()（）\"'") for part in parts if part.strip()])


def _normalize_inferred_ability_map(
    inferred_ability_map: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    if not inferred_ability_map:
        return normalized

    for course_name, payload in inferred_ability_map.items():
        standard_name = normalize_course_name(course_name)
        if isinstance(payload, dict):
            abilities = _split_items(payload.get("abilities", []))
            normalized[standard_name] = {
                "abilities": abilities,
                "confidence": payload.get("confidence", payload.get("confidence_score", 0.0)),
                "reason": _safe_text(payload.get("reason", "")),
                "source_label": _safe_text(payload.get("source_label", "AI推理")) or "AI推理",
            }
        else:
            normalized[standard_name] = {
                "abilities": _split_items(payload),
                "confidence": 0.0,
                "reason": "",
                "source_label": "AI推理",
            }

    return normalized


def _is_ascii_token(keyword: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9#+./-]{1,30}", keyword) is not None


@lru_cache(maxsize=2048)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    pattern = rf"(?<![A-Za-z0-9#+./-]){re.escape(keyword)}(?![A-Za-z0-9#+./-])"
    return re.compile(pattern, flags=re.IGNORECASE)


def _contains_prepared(text: str, lower_text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False

    if keyword.lower() not in lower_text:
        return False

    if _is_ascii_token(keyword):
        return _keyword_pattern(keyword).search(text) is not None

    return True


def _contains(text: str, keyword: str) -> bool:
    return _contains_prepared(text, text.lower(), keyword)


@lru_cache(maxsize=2048)
def normalize_course_name(course_name: str) -> str:
    text = _safe_text(course_name).strip()
    if not text:
        return ""

    for standard_name, aliases in COURSE_ALIASES.items():
        if any(_contains(text, alias) or _contains(alias, text) for alias in aliases):
            return standard_name

    return text


def extract_courses_from_resume(
    resume_text: str,
    inferred_ability_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    从简历文本中提取课程，并归一到标准课程名。
    """
    text = _safe_text(resume_text)
    raw_courses: list[str] = []

    for match in COURSE_SECTION_PATTERN.finditer(text):
        raw_courses.extend(_split_items(match.group("content")))

    # 简历格式不规范时，直接扫描常见课程别名。
    for standard_name, aliases in COURSE_ALIASES.items():
        if any(_contains(text, alias) for alias in aliases):
            raw_courses.append(standard_name)

    inferred_abilities = _normalize_inferred_ability_map(inferred_ability_map)
    courses = []
    for raw_course in _dedupe(raw_courses):
        standard_name = normalize_course_name(raw_course)
        if not standard_name:
            continue
        aliases = COURSE_ALIASES.get(standard_name, [standard_name])
        abilities = COURSE_ABILITY_MAP.get(standard_name, [])
        ability_source = "local_rule"
        ability_confidence = 1.0 if abilities else 0.0
        ability_reason = ""
        source_label = "本地知识库"
        if not abilities and standard_name in inferred_abilities:
            inferred = inferred_abilities[standard_name]
            abilities = inferred.get("abilities", [])
            ability_source = "ai_inference"
            ability_confidence = inferred.get("confidence", 0.0)
            ability_reason = inferred.get("reason", "")
            source_label = inferred.get("source_label", "AI推理")
        courses.append({
            "raw_name": raw_course,
            "course_name": standard_name,
            "aliases": aliases,
            "abilities": abilities,
            "ability_source": ability_source,
            "ability_confidence": ability_confidence,
            "ability_reason": ability_reason,
            "source_label": source_label,
            "evidence": f"简历中识别到课程：{raw_course}",
        })

    # 按标准课程去重，保留第一个证据。
    deduped: dict[str, dict[str, Any]] = {}
    for course in courses:
        deduped.setdefault(course["course_name"], course)
    return list(deduped.values())


def _get_record_attr(record: Any, key: str, default: Any = "") -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _extract_job_items(record: Any, *keys: str) -> list[str]:
    for key in keys:
        value = _get_record_attr(record, key, None)
        if value:
            return _split_items(value)
    return []


def _ability_match_score(ability: str, job_text: str, job_text_lower: str) -> tuple[bool, str]:
    candidates = [ability, *ABILITY_ALIASES.get(ability, [])]
    for candidate in candidates:
        if _contains_prepared(job_text, job_text_lower, candidate):
            return True, candidate
    return False, ""


def _prepare_job_record(record: Any) -> dict[str, Any]:
    if isinstance(record, dict) and record.get("_prepared"):
        return record

    required_skills = _extract_job_items(
        record,
        "required_skills",
        "required_skills_json",
        "skills",
    )
    related_projects = _extract_job_items(
        record,
        "related_projects",
        "related_projects_json",
        "projects",
    )
    recommended_courses = _extract_job_items(
        record,
        "recommended_courses",
        "recommended_courses_json",
        "courses",
    )
    job_name = _safe_text(_get_record_attr(record, "job_name", "未知岗位"))
    job_text = " ".join([
        job_name,
        " ".join(required_skills),
        " ".join(related_projects),
        " ".join(recommended_courses),
        _safe_text(_get_record_attr(record, "description", "")),
    ])

    return {
        "_prepared": True,
        "job_name": job_name,
        "required_skills": required_skills,
        "related_projects": related_projects,
        "recommended_courses": recommended_courses,
        "normalized_courses": [normalize_course_name(item) for item in recommended_courses],
        "job_text": job_text,
        "job_text_lower": job_text.lower(),
    }


def _build_job_cache_key(job_records: list[Any]) -> tuple[int, int, int, int] | None:
    ids: list[int] = []
    for record in job_records:
        record_id = _get_record_attr(record, "id", None)
        if record_id is None:
            return None
        try:
            ids.append(int(record_id))
        except (TypeError, ValueError):
            return None

    if not ids:
        return (0, 0, 0, 0)

    return (len(ids), min(ids), max(ids), sum(ids))


def _prepare_job_records(job_records: list[Any]) -> list[dict[str, Any]]:
    cache_key = _build_job_cache_key(job_records)
    if cache_key is None:
        return [_prepare_job_record(record) for record in job_records]

    cached_jobs = _JOB_INDEX_CACHE.get(cache_key)
    if cached_jobs is not None:
        return cached_jobs

    prepared_jobs = [_prepare_job_record(record) for record in job_records]
    if len(_JOB_INDEX_CACHE) >= 4:
        _JOB_INDEX_CACHE.clear()
    _JOB_INDEX_CACHE[cache_key] = prepared_jobs
    return prepared_jobs


def _top_unique_job_edges(edges: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected_edges: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    for edge in edges:
        job_key = edge["job_name"].strip().lower()
        if job_key in seen_jobs:
            continue
        seen_jobs.add(job_key)
        selected_edges.append(edge)
        if len(selected_edges) >= limit:
            break
    return selected_edges


def score_course_to_job(course: dict[str, Any], job_record: Any) -> dict[str, Any] | None:
    job_record = _prepare_job_record(job_record)
    job_name = job_record["job_name"]
    related_projects = job_record["related_projects"]
    recommended_courses = job_record["recommended_courses"]
    normalized_job_courses = job_record["normalized_courses"]
    job_text = job_record["job_text"]
    job_text_lower = job_record["job_text_lower"]

    matched_abilities = []
    matched_job_terms = []
    for ability in course.get("abilities", []):
        matched, term = _ability_match_score(ability, job_text, job_text_lower)
        if matched:
            matched_abilities.append(ability)
            matched_job_terms.append(term)

    matched_courses = [
        item for item in recommended_courses
        if normalize_course_name(item) == course["course_name"]
    ]
    same_course_hit = course["course_name"] in normalized_job_courses
    role_hits = [
        hint for hint in COURSE_ROLE_HINTS.get(course["course_name"], [])
        if _contains_prepared(job_text, job_text_lower, hint)
    ]

    ability_total = max(len(course.get("abilities", [])), 1)
    ability_coverage = len(matched_abilities) / ability_total
    course_hit_score = 1.0 if same_course_hit else 0.0
    project_hit = 1.0 if matched_abilities and related_projects else 0.0
    role_hit_score = min(len(_dedupe(role_hits)), 3) / 3

    score = round(
        ability_coverage * 50
        + course_hit_score * 25
        + project_hit * 10
        + min(len(matched_job_terms), 3) / 3 * 5
        + role_hit_score * 10
    )

    if score < 20:
        return None

    reason_parts = []
    if matched_abilities:
        reason_parts.append(f"课程能力覆盖：{'、'.join(matched_abilities[:5])}")
    if matched_courses:
        reason_parts.append(f"岗位推荐课程包含：{'、'.join(matched_courses[:3])}")
    if role_hits:
        reason_parts.append(f"岗位关键词命中：{'、'.join(_dedupe(role_hits)[:4])}")
    if not reason_parts:
        reason_parts.append("课程能力与岗位要求存在基础相关性")

    return {
        "course_name": course["course_name"],
        "job_name": job_name,
        "score": max(0, min(100, score)),
        "matched_abilities": _dedupe(matched_abilities),
        "matched_job_terms": _dedupe(matched_job_terms),
        "matched_courses": _dedupe(matched_courses),
        "ability_source": course.get("ability_source", "local_rule"),
        "ability_confidence": course.get("ability_confidence", 1.0),
        "source_label": course.get("source_label", "本地知识库"),
        "reason": "；".join(reason_parts),
    }


def build_course_job_mapping_graph(
    resume_text: str,
    job_records: list[Any],
    top_jobs_per_course: int = 5,
    inferred_ability_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    courses = extract_courses_from_resume(
        resume_text,
        inferred_ability_map=inferred_ability_map,
    )
    prepared_jobs = _prepare_job_records(job_records)
    edges: list[dict[str, Any]] = []

    for course in courses:
        course_edges = []
        for job_record in prepared_jobs:
            edge = score_course_to_job(course, job_record)
            if edge:
                course_edges.append(edge)
        course_edges.sort(key=lambda item: item["score"], reverse=True)
        edges.extend(_top_unique_job_edges(course_edges, top_jobs_per_course))

    jobs: dict[str, dict[str, Any]] = {}
    for edge in edges:
        jobs.setdefault(edge["job_name"], {
            "job_name": edge["job_name"],
            "max_score": edge["score"],
        })
        jobs[edge["job_name"]]["max_score"] = max(jobs[edge["job_name"]]["max_score"], edge["score"])

    courses_with_score = []
    for course in courses:
        related_edges = [edge for edge in edges if edge["course_name"] == course["course_name"]]
        courses_with_score.append({
            **course,
            "top_score": max([edge["score"] for edge in related_edges], default=0),
            "job_count": len(related_edges),
        })

    courses_with_score.sort(key=lambda item: (item["top_score"], item["job_count"]), reverse=True)
    sorted_jobs = sorted(jobs.values(), key=lambda item: item["max_score"], reverse=True)

    return {
        "courses": courses_with_score,
        "jobs": sorted_jobs,
        "edges": edges,
        "summary": {
            "course_count": len(courses_with_score),
            "job_count": len(sorted_jobs),
            "edge_count": len(edges),
        },
    }
