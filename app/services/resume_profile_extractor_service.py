from __future__ import annotations

import re
from typing import Any


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _pick_line_value(text: str, labels: list[str], default: str = "") -> str:
    for label in labels:
        pattern = rf"{re.escape(label)}[：:\s]*([^\n\r，,；;]{{1,80}})"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return default


def _extract_section(text: str, labels: list[str], limit: int = 500) -> str:
    for label in labels:
        pattern = rf"{re.escape(label)}[：:\s]*(.+?)(?=\n\s*(?:教育经历|项目经历|实习经历|专业技能|技能|证书|竞赛|自我评价|求职意向|$))"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.S)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()[:limit]
    return ""


def extract_student_profile_from_resume(message: str, resume_text: str) -> dict[str, str]:
    """
    从简历文本中抽取画像分支所需的基础字段。
    这是首页智能体的轻量兜底抽取器；深度评分仍交给诊断智能体完成。
    """
    combined = f"{_safe_text(message)}\n{_safe_text(resume_text)}"
    name = _pick_line_value(combined, ["姓名", "名字"], "未填写")
    major = _pick_line_value(combined, ["专业", "所学专业"], "未填写")
    grade = _pick_line_value(combined, ["年级", "学历", "教育背景"], "未填写")
    target_job = _pick_line_value(combined, ["目标岗位", "求职意向", "意向岗位"], "未填写")

    if target_job == "未填写":
        role_match = re.search(r"([A-Za-z0-9\u4e00-\u9fff+#.]{2,30}(?:工程师|开发|分析师|产品经理|测试|运维))", combined)
        if role_match:
            target_job = role_match.group(1)

    skills = _extract_section(combined, ["专业技能", "技能", "技能证书"], 800)
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
