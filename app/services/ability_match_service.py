from __future__ import annotations

import json
import re
from typing import Any


KEYWORD_ALIASES: dict[str, list[str]] = {
    "Java": ["Java", "java"],
    "Spring Boot": ["Spring Boot", "SpringBoot", "springboot", "Spring"],
    "Spring Cloud": ["Spring Cloud", "SpringCloud", "springcloud"],
    "MyBatis": ["MyBatis", "mybatis", "Mybatis"],
    "Python": ["Python", "python"],
    "Django": ["Django", "django"],
    "Flask": ["Flask", "flask"],
    "FastAPI": ["FastAPI", "Fast API", "fastapi"],
    "JavaScript": ["JavaScript", "Javascript", "JS", "js"],
    "TypeScript": ["TypeScript", "Typescript", "TS", "ts"],
    "React": ["React", "react", "React.js", "ReactJS"],
    "Vue": ["Vue", "vue", "Vue.js", "Vue3", "Vue2"],
    "HTML": ["HTML", "HTML5", "html", "h5", "H5"],
    "CSS": ["CSS", "CSS3", "css"],
    "Node.js": ["Node.js", "Nodejs", "nodejs", "Node"],
    "MySQL": ["MySQL", "mysql"],
    "PostgreSQL": ["PostgreSQL", "postgresql", "Postgres", "PG"],
    "Redis": ["Redis", "redis"],
    "MongoDB": ["MongoDB", "mongodb", "Mongo"],
    "Oracle": ["Oracle", "oracle"],
    "SQL": ["SQL", "sql"],
    "Linux": ["Linux", "linux", "Unix", "Shell"],
    "Git": ["Git", "git", "GitHub", "GitLab"],
    "Docker": ["Docker", "docker", "Docker Compose", "docker-compose"],
    "Kubernetes": ["Kubernetes", "kubernetes", "K8s", "k8s"],
    "Nginx": ["Nginx", "nginx"],
    "Jenkins": ["Jenkins", "jenkins"],
    "CI/CD": ["CI/CD", "CICD", "持续集成", "持续交付"],
    "Maven": ["Maven", "maven"],
    "RabbitMQ": ["RabbitMQ", "rabbitmq"],
    "Kafka": ["Kafka", "kafka"],
    "Celery": ["Celery", "celery"],
    "RESTful API": ["RESTful", "REST API", "RESTful API", "REST接口", "接口设计"],
    "微服务": ["微服务", "Microservice", "Microservices"],
    "分布式": ["分布式", "Distributed"],
    "高并发": ["高并发", "并发", "高可用", "高性能"],
    "性能优化": ["性能优化", "调优", "性能调优", "优化"],
    "缓存": ["缓存", "cache", "Cache"],
    "消息队列": ["消息队列", "MQ", "队列"],
    "JVM": ["JVM", "jvm"],
    "数据结构": ["数据结构"],
    "算法": ["算法", "算法基础"],
    "操作系统": ["操作系统"],
    "计算机网络": ["计算机网络", "网络协议", "TCP/IP", "HTTP"],
    "数据库": ["数据库", "DB"],
    "设计模式": ["设计模式"],
    "单元测试": ["单元测试", "JUnit", "pytest", "测试用例"],
    "接口测试": ["接口测试", "Postman", "Apifox"],
}


ROLE_PROFILES: dict[str, dict[str, list[str]]] = {
    "java_backend": {
        "core": [
            "Java",
            "Spring Boot",
            "Spring Cloud",
            "MyBatis",
            "MySQL",
            "Redis",
            "RESTful API",
            "Git",
            "Linux",
            "Maven",
        ],
        "foundations": ["数据结构", "算法", "操作系统", "计算机网络", "数据库", "设计模式"],
        "tools": ["Git", "Linux", "Maven", "Docker", "Nginx", "Jenkins", "Kubernetes", "RabbitMQ", "Kafka"],
        "advanced": ["微服务", "分布式", "高并发", "性能优化", "消息队列", "缓存", "JVM"],
    },
    "python_backend": {
        "core": [
            "Python",
            "Django",
            "Flask",
            "FastAPI",
            "MySQL",
            "PostgreSQL",
            "Redis",
            "RESTful API",
            "Git",
            "Linux",
        ],
        "foundations": ["数据结构", "算法", "操作系统", "计算机网络", "数据库", "设计模式"],
        "tools": ["Git", "Linux", "Docker", "Kubernetes", "Nginx", "Celery", "RabbitMQ", "Kafka"],
        "advanced": ["微服务", "分布式", "高并发", "性能优化", "消息队列", "缓存"],
    },
    "frontend": {
        "core": ["JavaScript", "TypeScript", "HTML", "CSS", "Vue", "React", "Node.js", "Git"],
        "foundations": ["数据结构", "算法", "计算机网络", "设计模式"],
        "tools": ["Git", "Nginx", "CI/CD"],
        "advanced": ["性能优化", "工程化", "组件化", "可视化", "跨端"],
    },
    "algorithm": {
        "core": ["Python", "算法", "数据结构", "SQL"],
        "foundations": ["算法", "数据结构", "操作系统", "计算机网络"],
        "tools": ["Git", "Linux", "Docker"],
        "advanced": ["机器学习", "深度学习", "大模型", "NLP", "CV"],
    },
    "general": {
        "core": ["Python", "Java", "JavaScript", "MySQL", "Git", "Linux"],
        "foundations": ["数据结构", "算法", "操作系统", "计算机网络", "数据库"],
        "tools": ["Git", "Linux", "Docker"],
        "advanced": ["性能优化", "分布式", "高并发"],
    },
}


PROJECT_KEYWORDS = [
    "项目",
    "系统",
    "平台",
    "接口",
    "后台",
    "管理系统",
    "电商",
    "博客",
    "秒杀",
    "权限",
    "登录",
    "支付",
    "部署",
    "上线",
]
SOFT_KEYWORDS = ["沟通", "协作", "团队", "学习", "责任", "主动", "抗压", "文档", "复盘"]
EDUCATION_KEYWORDS = ["本科", "硕士", "研究生", "计算机", "软件工程", "信息", "通信"]
CERTIFICATE_KEYWORDS = ["软考", "系统设计师", "软件设计师", "英语四级", "英语六级", "CET-4", "CET-6"]
QUANTITATIVE_KEYWORDS = ["高并发", "高可用", "性能", "QPS", "TPS", "万", "百万", "千万", "优化", "压测"]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_safe_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_safe_text(item) for item in value.values())
    return str(value)


def _json_loads(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    if not value:
        return []
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _flatten_keywords(value: Any) -> list[str]:
    parsed = _json_loads(value)
    if isinstance(parsed, dict):
        items: list[str] = []
        for item in parsed.values():
            items.extend(_flatten_keywords(item))
        return _dedupe(items)
    if isinstance(parsed, (list, tuple, set)):
        items = []
        for item in parsed:
            items.extend(_flatten_keywords(item))
        return _dedupe(items)
    text = _safe_text(parsed)
    return _dedupe(part.strip() for part in re.split(r"[,，、;/；\n\r]+", text) if part.strip())


def _dedupe(values: Any) -> list[str]:
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


def _contains(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False

    candidates = [keyword, *KEYWORD_ALIASES.get(keyword, [])]
    for token in candidates:
        token = _safe_text(token).strip()
        if not token:
            continue
        if re.fullmatch(r"[A-Za-z0-9#+.]{1,24}", token):
            pattern = rf"(?<![A-Za-z0-9#+.]){re.escape(token)}(?![A-Za-z0-9#+.])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        elif token.lower() in text.lower():
            return True
    return False


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    return _dedupe(keyword for keyword in keywords if _contains(text, keyword))


def _coverage(hits: list[str], expected: list[str]) -> float:
    return len(set(hit.lower() for hit in hits)) / max(len(expected), 1)


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def build_profile_text(student_data: dict[str, Any]) -> str:
    fields = [
        "name",
        "major",
        "education",
        "target_job",
        "skills",
        "projects",
        "internships",
        "competitions",
        "certificates",
        "self_intro",
        "resume_text",
        "normalized_text",
    ]
    return " ".join(_safe_text(student_data.get(field)) for field in fields)


def infer_target_role(student_data: dict[str, Any] | None = None, job_name: str | None = None) -> str:
    student_data = student_data or {}
    role_text = f"{_safe_text(job_name)} {_safe_text(student_data.get('target_job'))}"
    full_text = f"{role_text} {build_profile_text(student_data)}"

    if any(_contains(role_text, item) for item in ["Java后端", "Java 后端", "Java开发", "Java 开发", "Spring Boot", "Spring"]):
        return "java_backend"
    if any(_contains(role_text, item) for item in ["Python", "Django", "Flask", "FastAPI"]):
        return "python_backend"
    if any(_contains(role_text, item) for item in ["前端", "React", "Vue", "JavaScript", "TypeScript", "HTML", "CSS"]):
        return "frontend"
    if any(_contains(role_text, item) for item in ["算法", "机器学习", "深度学习", "大模型", "LLM", "Agent"]):
        return "algorithm"

    role_hits = {
        "java_backend": len(keyword_hits(full_text, ROLE_PROFILES["java_backend"]["core"])),
        "python_backend": len(keyword_hits(full_text, ROLE_PROFILES["python_backend"]["core"])),
        "frontend": len(keyword_hits(full_text, ROLE_PROFILES["frontend"]["core"])),
        "algorithm": len(keyword_hits(full_text, ROLE_PROFILES["algorithm"]["core"])),
    }
    best_role, best_count = max(role_hits.items(), key=lambda item: item[1])
    return best_role if best_count >= 2 else "general"


def score_four_dimensions(student_data: dict[str, Any], target_job: str | None = None) -> dict[str, Any]:
    role = infer_target_role(student_data, target_job)
    profile = ROLE_PROFILES[role]
    text = f"{build_profile_text(student_data)} {_safe_text(target_job)}"

    core_hits = keyword_hits(text, profile["core"])
    foundation_hits = keyword_hits(text, profile["foundations"])
    tool_hits = keyword_hits(text, profile["tools"])
    advanced_hits = keyword_hits(text, profile["advanced"])
    project_hits = keyword_hits(text, PROJECT_KEYWORDS)
    soft_hits = keyword_hits(text, SOFT_KEYWORDS)
    education_hits = keyword_hits(text, EDUCATION_KEYWORDS)
    certificate_hits = keyword_hits(text, CERTIFICATE_KEYWORDS)

    has_project = bool(_safe_text(student_data.get("projects"))) or bool(project_hits)
    has_internship = bool(_safe_text(student_data.get("internships"))) or any(
        _contains(text, item) for item in ["实习", "实训", "校招", "应届"]
    )
    has_competition = bool(_safe_text(student_data.get("competitions"))) or any(
        _contains(text, item) for item in ["竞赛", "比赛", "蓝桥杯", "ACM", "CCPC", "ICPC"]
    )
    has_target_job = bool(_safe_text(target_job or student_data.get("target_job")))
    has_intro = len(_safe_text(student_data.get("self_intro"))) >= 30
    quantitative_hits = keyword_hits(text, QUANTITATIVE_KEYWORDS)

    resume_fields = sum(
        bool(_safe_text(student_data.get(field)))
        for field in ["major", "education", "skills", "projects", "internships", "competitions", "certificates", "self_intro"]
    )

    professional = 35 + _coverage(core_hits, profile["core"]) * 35 + _coverage(foundation_hits, profile["foundations"]) * 20
    professional += 8 if education_hits else 0
    professional += min(len(certificate_hits) * 3, 7)

    practice = 25
    practice += 18 if has_project else 0
    practice += 12 if has_internship else 0
    practice += 10 if has_competition else 0
    practice += _coverage(project_hits, PROJECT_KEYWORDS) * 20
    practice += _coverage(advanced_hits, profile["advanced"]) * 15
    practice += 5 if quantitative_hits else 0

    tools = 30 + _coverage(tool_hits, profile["tools"]) * 45
    tools += _coverage(core_hits, profile["core"]) * 15
    tools += _coverage(advanced_hits, profile["advanced"]) * 10

    career = 35
    career += 20 if has_target_job else 0
    career += min(resume_fields * 5, 20)
    career += 10 if has_intro else 0
    career += min(len(soft_hits) * 3, 10)
    career += 5 if education_hits else 0

    scores = {
        "professional": _clamp_score(professional),
        "practice": _clamp_score(practice),
        "tools": _clamp_score(tools),
        "career": _clamp_score(career),
    }

    evidence = {
        "professional": _dedupe(core_hits + foundation_hits + education_hits + certificate_hits),
        "practice": _dedupe(project_hits + advanced_hits + quantitative_hits),
        "tools": _dedupe(tool_hits + core_hits),
        "career": _dedupe(soft_hits + (["目标岗位明确"] if has_target_job else []) + education_hits),
    }

    breakdown = {
        "target_role": role,
        "core_skill_coverage": round(_coverage(core_hits, profile["core"]), 3),
        "foundation_coverage": round(_coverage(foundation_hits, profile["foundations"]), 3),
        "tool_coverage": round(_coverage(tool_hits, profile["tools"]), 3),
        "advanced_coverage": round(_coverage(advanced_hits, profile["advanced"]), 3),
        "resume_field_count": resume_fields,
        "has_project": has_project,
        "has_internship": has_internship,
        "has_competition": has_competition,
        "has_target_job": has_target_job,
    }

    return {
        "ability_scores": scores,
        "score_evidence": evidence,
        "score_breakdown": breakdown,
        "recognized_skills": _dedupe(core_hits + foundation_hits + tool_hits + advanced_hits),
    }


def _get_attr(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _extract_job_keywords(record: Any) -> tuple[list[str], list[str], list[str]]:
    required_skills = _flatten_keywords(
        _get_attr(record, "required_skills")
        or _get_attr(record, "required_skills_json")
        or _get_attr(record, "skills")
        or _get_attr(record, "skill_tags")
    )
    related_projects = _flatten_keywords(
        _get_attr(record, "related_projects")
        or _get_attr(record, "related_projects_json")
        or _get_attr(record, "projects")
        or _get_attr(record, "description")
    )
    recommended_certificates = _flatten_keywords(
        _get_attr(record, "recommended_certificates")
        or _get_attr(record, "recommended_certificates_json")
        or _get_attr(record, "certificates")
    )
    return required_skills, related_projects, recommended_certificates


def _education_fits(student_text: str, requirement: str) -> float:
    if not requirement:
        return 1.0
    if any(_contains(requirement, item) for item in ["不限", "学历不限"]):
        return 1.0
    if any(_contains(requirement, item) for item in ["硕士", "研究生"]):
        return 1.0 if any(_contains(student_text, item) for item in ["硕士", "研究生", "博士"]) else 0.35
    if _contains(requirement, "本科"):
        return 1.0 if any(_contains(student_text, item) for item in ["本科", "硕士", "研究生", "博士"]) else 0.45
    if any(_contains(requirement, item) for item in ["大专", "专科"]):
        return 1.0 if any(_contains(student_text, item) for item in ["大专", "专科", "本科", "硕士", "研究生"]) else 0.55
    return 0.8


def match_profile_to_job(
    student_data: dict[str, Any],
    record: Any,
    assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assessment = assessment or score_four_dimensions(student_data)
    scores = assessment["ability_scores"]
    student_text = build_profile_text(student_data)

    job_name = _safe_text(_get_attr(record, "job_name") or _get_attr(record, "name") or "未知岗位")
    category = _safe_text(_get_attr(record, "category") or infer_target_role(student_data, job_name))
    description = _safe_text(_get_attr(record, "description") or _get_attr(record, "job_description"))
    education_requirement = _safe_text(_get_attr(record, "education_requirement") or _get_attr(record, "education"))

    required_skills, related_projects, recommended_certificates = _extract_job_keywords(record)
    if not required_skills:
        required_skills = ROLE_PROFILES[infer_target_role(student_data, job_name)]["core"]

    matched_skills = keyword_hits(student_text, required_skills)
    matched_projects = keyword_hits(student_text, related_projects)
    matched_certificates = keyword_hits(student_text, recommended_certificates)
    missing_skills = [skill for skill in required_skills if skill.lower() not in {item.lower() for item in matched_skills}]

    skill_ratio = _coverage(matched_skills, required_skills)
    project_ratio = _coverage(matched_projects, related_projects) if related_projects else min(scores["practice"] / 100, 0.8)
    role = infer_target_role(student_data, job_name)
    student_role = assessment.get("score_breakdown", {}).get("target_role", infer_target_role(student_data))
    role_fit = 1.0 if role == student_role else 0.65
    education_fit = _education_fits(student_text, education_requirement)
    ability_avg = (
        scores["professional"] * 0.30
        + scores["practice"] * 0.25
        + scores["tools"] * 0.30
        + scores["career"] * 0.15
    ) / 100
    certificate_component = _coverage(matched_certificates, recommended_certificates) * 2 if recommended_certificates else 0

    match_score = _clamp_score(
        skill_ratio * 55
        + project_ratio * 15
        + ability_avg * 15
        + role_fit * 10
        + education_fit * 3
        + certificate_component
    )

    return {
        "job_name": job_name,
        "category": category,
        "match_score": match_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_gaps": missing_skills[:8],
        "matched_projects": matched_projects,
        "matched_certificates": matched_certificates,
        "description": description,
        "score_components": {
            "skill_coverage": round(skill_ratio, 3),
            "project_evidence": round(project_ratio, 3),
            "ability_average": round(ability_avg, 3),
            "role_fit": round(role_fit, 3),
            "education_fit": round(education_fit, 3),
        },
        "recommend_reason": generate_match_reason(job_name, matched_skills, matched_projects, missing_skills),
        "reason": generate_match_reason(job_name, matched_skills, matched_projects, missing_skills),
    }


def generate_match_reason(
    job_name: str,
    matched_skills: list[str],
    matched_projects: list[str],
    missing_skills: list[str],
) -> str:
    reason = f"系统基于简历能力画像和岗位知识图谱分析，认为该候选人与{job_name}存在匹配基础。"
    if matched_skills:
        reason += f" 已匹配技能：{'、'.join(matched_skills[:8])}。"
    if matched_projects:
        reason += f" 简历中出现相关实践线索：{'、'.join(matched_projects[:5])}。"
    if missing_skills:
        reason += f" 后续建议补强：{'、'.join(missing_skills[:6])}。"
    return reason
