from __future__ import annotations

import re
from typing import Any


MISSING_VALUES = {"", "未填写", "未提取", "未知", "无"}
SECTION_TITLES = {
    "个人信息",
    "基本信息",
    "教育经历",
    "教育背景",
    "相关技能",
    "专业技能",
    "技能",
    "项目经历",
    "项目经验",
    "实习经历",
    "竞赛经历",
    "获奖经历",
    "证书",
    "自我评价",
    "自我介绍",
    "求职意向",
}
DEGREE_PATTERN = r"本科|硕士|研究生|博士|大专|专科"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_value(value: str) -> str:
    return re.sub(r"^[|｜\s]+|[|｜\s]+$", "", _safe_text(value))


def _lines(text: str) -> list[str]:
    return [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in _safe_text(text).splitlines()
        if line.strip()
    ]


def _is_missing(value: str) -> bool:
    return _clean_value(value) in MISSING_VALUES


def _pick_line_value(text: str, labels: list[str], default: str = "") -> str:
    for label in labels:
        pattern = rf"(?:^|[\n\r])\s*{re.escape(label)}[：: \t]*([^\n\r，,；;]{{1,80}})"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_value(match.group(1))
    return default


def _extract_section(text: str, labels: list[str], limit: int = 500) -> str:
    for label in labels:
        pattern = rf"{re.escape(label)}[：:\s]*(.+?)(?=\n\s*(?:教育经历|项目经历|实习经历|专业技能|技能|证书|竞赛|自我评价|求职意向|$))"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.S)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()[:limit]
    return ""


def _extract_name(text: str) -> str:
    labeled = _pick_line_value(text, ["姓名", "名字"])
    if not _is_missing(labeled):
        return labeled

    contact_markers = ("性别", "年龄", "电话", "手机", "邮箱", "户籍", "政治面貌")
    for index, line in enumerate(_lines(text)[:12]):
        candidate = re.sub(r"\s+", "", line)
        if candidate in SECTION_TITLES:
            continue
        if not re.fullmatch(r"[\u4e00-\u9fff·]{2,4}", candidate):
            continue
        nearby = " ".join(_lines(text)[index + 1:index + 4])
        if any(marker in nearby for marker in contact_markers):
            return candidate
    return "未填写"


def _extract_education(text: str) -> tuple[str, str]:
    labeled_major = _pick_line_value(text, ["所学专业", "专业"])
    labeled_grade = _pick_line_value(text, ["年级", "学历", "最高学历"])

    major = "" if _is_missing(labeled_major) else labeled_major
    grade = "" if _is_missing(labeled_grade) else labeled_grade

    for line in _lines(text):
        degree_match = re.search(rf"(?P<degree>{DEGREE_PATTERN})", line)
        if not degree_match:
            continue

        if not grade:
            grade = degree_match.group("degree")

        if major:
            break

        before_degree = line[:degree_match.start()]
        before_degree = re.sub(
            r"(?:19|20)\s*\d{2}\s*[./年-]\s*\d{1,2}.*?(?:19|20)\s*\d{2}\s*[./年-]\s*\d{1,2}",
            "",
            before_degree,
        )
        before_degree = re.sub(r"^.*?(?:大学|学院)\s*", "", before_degree)
        candidate = _clean_value(before_degree)
        if 2 <= len(candidate) <= 40 and not re.search(r"课程|项目|论文|经历", candidate):
            major = candidate
            break

    return major or "未填写", grade or "未填写"


def _extract_target_job(text: str) -> str:
    labeled = _pick_line_value(text, ["目标岗位", "求职意向", "意向岗位", "应聘岗位"])
    if not _is_missing(labeled):
        return labeled

    role_match = re.search(
        r"([A-Za-z0-9\u4e00-\u9fff+#.]{1,24}"
        r"(?:后端开发工程师|前端开发工程师|软件开发工程师|开发工程师|"
        r"测试工程师|运维工程师|算法工程师|数据分析师|产品经理|工程师))",
        text,
        flags=re.IGNORECASE,
    )
    if role_match:
        return _clean_value(role_match.group(1))

    compact = re.sub(r"\s+", "", text).lower()
    if "java" in compact and any(keyword in compact for keyword in ["spring", "mybatis", "redis"]):
        return "Java 后端工程师"
    if "python" in compact and any(keyword in compact for keyword in ["django", "flask", "fastapi"]):
        return "Python 后端工程师"
    if any(keyword in compact for keyword in ["vue", "react", "javascript", "typescript"]):
        return "前端开发工程师"
    return "未填写"


def extract_student_profile_from_resume(message: str, resume_text: str) -> dict[str, str]:
    """
    从简历文本中抽取画像分支所需的基础字段。
    这是首页智能体的轻量兜底抽取器；深度评分仍交给诊断智能体完成。
    """
    combined = f"{_safe_text(message)}\n{_safe_text(resume_text)}"
    name = _extract_name(resume_text)
    major, grade = _extract_education(resume_text)
    target_job = _extract_target_job(combined)

    skills = _extract_section(combined, ["专业技能", "相关技能", "技能", "技能证书"], 800)
    projects = _extract_section(combined, ["项目经历", "项目经验"], 1000)
    competitions = _extract_section(combined, ["竞赛经历", "比赛经历", "获奖经历"], 500)
    certificates = _extract_section(combined, ["证书", "技能证书", "资格证书"], 500)
    self_intro = _extract_section(combined, ["自我评价", "自我介绍", "个人总结"], 500)

    return {
        "name": name,
        "major": major,
        "grade": grade,
        "target_job": target_job,
        "skills": skills or "未提取到明确技能",
        "projects": projects,
        "competitions": competitions,
        "certificates": certificates,
        "self_intro": self_intro or "由首页智能体根据简历自动生成画像。",
    }
