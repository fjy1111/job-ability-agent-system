import json


def text_contains_keywords(text: str, keywords: list[str]) -> list[str]:
    """
    判断文本中命中了哪些关键词
    """
    matched = []

    if not text:
        return matched

    text_lower = text.lower()

    for keyword in keywords:
        if keyword.lower() in text_lower:
            matched.append(keyword)

    return matched


def calculate_job_match(student_data: dict, job_records: list) -> list[dict]:
    """
    根据学生信息和数据库中的岗位能力知识图谱，计算岗位匹配度
    """

    skills_text = student_data.get("skills", "")
    projects_text = student_data.get("projects", "")
    certificates_text = student_data.get("certificates", "")

    results = []

    for record in job_records:
        required_skills = json.loads(record.required_skills_json)
        related_projects = json.loads(record.related_projects_json)
        recommended_certificates = json.loads(record.recommended_certificates_json)

        matched_skills = text_contains_keywords(skills_text, required_skills)
        matched_projects = text_contains_keywords(projects_text, related_projects)
        matched_certificates = text_contains_keywords(certificates_text, recommended_certificates)

        skill_score = len(matched_skills) / max(len(required_skills), 1) * 50
        project_score = len(matched_projects) / max(len(related_projects), 1) * 30
        certificate_score = len(matched_certificates) / max(len(recommended_certificates), 1) * 20

        total_score = round(skill_score + project_score + certificate_score, 2)

        missing_skills = [
            skill for skill in required_skills
            if skill not in matched_skills
        ]

        results.append({
            "job_name": record.job_name,
            "match_score": total_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_projects": matched_projects,
            "matched_certificates": matched_certificates,
            "recommend_reason": generate_reason(
                record.job_name,
                matched_skills,
                matched_projects,
                missing_skills
            )
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)

    return results[:5]


def generate_reason(job_name, matched_skills, matched_projects, missing_skills):
    """
    生成推荐理由
    """

    reason = f"系统基于岗位能力知识图谱分析，认为该学生与{job_name}具有一定匹配度。"

    if matched_skills:
        reason += f" 已匹配技能包括：{'、'.join(matched_skills)}。"

    if matched_projects:
        reason += f" 项目经历中包含相关实践：{'、'.join(matched_projects)}。"

    if missing_skills:
        reason += f" 后续建议重点补充：{'、'.join(missing_skills)}。"

    return reason