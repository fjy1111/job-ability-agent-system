from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - the fallback path keeps the page usable.
    ChatOpenAI = None


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


DEFAULT_INTERVIEWER = "林老师"

QUESTION_COUNT = 6

ROLE_KEYWORDS = [
    "Python",
    "Java",
    "Spring Boot",
    "FastAPI",
    "Django",
    "MySQL",
    "Redis",
    "Linux",
    "Docker",
    "Git",
    "Vue",
    "React",
    "TypeScript",
    "JavaScript",
    "数据分析",
    "机器学习",
    "深度学习",
    "算法",
    "大模型",
    "LangChain",
    "LangGraph",
    "项目管理",
    "需求分析",
    "沟通表达",
    "自动化测试",
    "接口测试",
]


def _safe_text(value: str | None) -> str:
    return (value or "").strip()


def _clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def _safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        return {}

    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _create_llm() -> Any | None:
    if ChatOpenAI is None:
        return None

    use_llm = os.getenv("USE_LLM", "true").lower() == "true"
    if not use_llm:
        return None

    api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not api_key or not model:
        return None

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.35,
        timeout=15,
        max_retries=0,
    )


def _pick_focus_keywords(text: str, limit: int = 8) -> list[str]:
    focus = [
        keyword
        for keyword in ROLE_KEYWORDS
        if keyword.lower() in text.lower()
    ]

    raw_items = re.split(r"[\s,，、;；。:：/|]+", text)
    for item in raw_items:
        word = item.strip("()（）[]【】<>《》")
        if len(word) < 2 or len(word) > 20:
            continue
        if word not in focus and re.search(r"[\u4e00-\u9fffA-Za-z]", word):
            focus.append(word)
        if len(focus) >= limit:
            break

    return focus[:limit]


def _infer_role(target_role: str, job_description: str, resume_text: str) -> str:
    target_role = _safe_text(target_role)
    if target_role:
        return target_role

    combined = f"{job_description} {resume_text}"
    role_patterns = [
        r"([A-Za-z0-9\u4e00-\u9fff]{2,24}(?:工程师|开发|分析师|产品经理|测试|运营|设计师))",
        r"(?:岗位|职位|目标)[:：\s]*([A-Za-z0-9\u4e00-\u9fff]{2,24})",
    ]

    for pattern in role_patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group(1)

    return "目标岗位"


def _build_fallback_questions(role: str, focus_keywords: list[str]) -> list[str]:
    focus_text = "、".join(focus_keywords[:3]) if focus_keywords else "岗位核心技能"

    return [
        f"请先做一个面向“{role}”的自我介绍，重点说明你的能力和这个岗位的关系。",
        "请选择一个最能代表你能力的项目，说明背景、你的职责、关键行动和最终结果。",
        f"这个岗位会关注{focus_text}。你目前有哪些相关经验？还有哪些地方需要补强？",
        "如果入职后遇到一个陌生但紧急的任务，你会如何拆解、推进和同步进展？",
        f"你认为自己和“{role}”之间最大的差距是什么？接下来三个月你会怎么提升？",
        "最后，请用一段简洁有说服力的话总结为什么我们应该选择你。",
    ]


def _generate_llm_questions(
    role: str,
    job_description: str,
    resume_text: str,
    fallback_questions: list[str],
) -> list[str]:
    llm = _create_llm()
    if llm is None:
        return fallback_questions

    prompt = f"""
你是严谨但友好的 AI 模拟面试官。请根据目标岗位、JD 和候选人简历，生成 {QUESTION_COUNT} 个循序渐进的中文面试问题。

要求：
1. 问题要像真人面试官一样自然、具体。
2. 覆盖自我介绍、项目经历、岗位技能、问题拆解、短板反思、录用理由。
3. 不要编造候选人没有提供的信息。
4. 只输出 JSON，不要 Markdown。

目标岗位：{role}
岗位信息：{job_description[:1500]}
简历信息：{resume_text[:1800]}

JSON 格式：
{{
  "questions": ["问题1", "问题2", "问题3", "问题4", "问题5", "问题6"]
}}
"""

    try:
        response = llm.invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        questions = parsed.get("questions", [])
        if isinstance(questions, list) and len(questions) >= 4:
            cleaned = [
                _safe_text(str(item))
                for item in questions
                if _safe_text(str(item))
            ]
            if len(cleaned) >= 4:
                return (cleaned + fallback_questions)[:QUESTION_COUNT]
    except Exception:
        return fallback_questions

    return fallback_questions


def build_interview_session(
    target_role: str,
    job_description: str,
    resume_text: str,
) -> dict[str, Any]:
    role = _infer_role(target_role, job_description, resume_text)
    combined_context = f"{role} {job_description} {resume_text}"
    focus_keywords = _pick_focus_keywords(combined_context)
    fallback_questions = _build_fallback_questions(role, focus_keywords)
    questions = _generate_llm_questions(
        role=role,
        job_description=job_description,
        resume_text=resume_text,
        fallback_questions=fallback_questions,
    )

    first_question = questions[0]

    return {
        "session_id": str(uuid.uuid4()),
        "interviewer_name": DEFAULT_INTERVIEWER,
        "target_role": role,
        "job_description": job_description[:2000],
        "resume_text": resume_text[:2500],
        "focus_keywords": focus_keywords,
        "questions": questions,
        "total_rounds": len(questions),
        "current_round": 1,
        "current_question": first_question,
        "opening_message": (
            f"你好，我是{DEFAULT_INTERVIEWER}。今天我们围绕“{role}”做一次模拟面试。"
            "我会像正式面试一样提问，也会在每次回答后给你具体建议。"
        ),
    }


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _score_answer(answer: str, focus_keywords: list[str]) -> dict[str, int]:
    length = len(answer)
    has_context = _has_any(answer, ["背景", "需求", "目标", "任务", "问题", "场景"])
    has_action = _has_any(answer, ["负责", "完成", "设计", "实现", "推进", "协作", "使用", "优化", "分析"])
    has_result = bool(re.search(r"\d|%|提升|降低|上线|交付|通过|获奖|用户|准确率|效率|排名|完成", answer))
    matched_focus = [
        keyword
        for keyword in focus_keywords
        if keyword and keyword.lower() in answer.lower()
    ]

    structure_score = 45
    structure_score += 15 if has_context else 0
    structure_score += 20 if has_action else 0
    structure_score += 20 if has_result else 0

    relevance_score = 58 + min(len(matched_focus), 4) * 8
    if length >= 80:
        relevance_score += 8
    if length < 35:
        relevance_score -= 18

    evidence_score = 42
    evidence_score += 25 if has_action else 0
    evidence_score += 25 if has_result else 0
    if re.search(r"\d", answer):
        evidence_score += 8

    expression_score = 55
    if 80 <= length <= 450:
        expression_score += 22
    elif length > 450:
        expression_score += 10
    if "。" in answer or "，" in answer:
        expression_score += 8

    scores = {
        "structure_score": _clamp(structure_score),
        "relevance_score": _clamp(relevance_score),
        "evidence_score": _clamp(evidence_score),
        "expression_score": _clamp(expression_score),
    }
    scores["overall_score"] = round(sum(scores.values()) / len(scores))
    return scores


def _fallback_feedback(
    session: dict[str, Any],
    question: str,
    answer: str,
) -> dict[str, Any]:
    role = _safe_text(session.get("target_role")) or "目标岗位"
    focus_keywords = session.get("focus_keywords", [])
    if not isinstance(focus_keywords, list):
        focus_keywords = []

    scores = _score_answer(answer, focus_keywords)
    has_context = _has_any(answer, ["背景", "需求", "目标", "任务", "问题", "场景"])
    has_action = _has_any(answer, ["负责", "完成", "设计", "实现", "推进", "协作", "使用", "优化", "分析"])
    has_result = bool(re.search(r"\d|%|提升|降低|上线|交付|通过|获奖|用户|准确率|效率|排名|完成", answer))
    matched_focus = [
        keyword
        for keyword in focus_keywords
        if keyword and keyword.lower() in answer.lower()
    ]

    suggestions: list[str] = []

    if len(answer) < 60:
        suggestions.append("回答偏短，可以补充一个具体项目或学习经历，让面试官看到证据。")
    if not has_context:
        suggestions.append("建议先交代背景或目标，再讲行动，回答会更完整。")
    if not has_action:
        suggestions.append("需要突出你本人做了什么，例如负责的模块、使用的方法和推进过程。")
    if not has_result:
        suggestions.append("建议补上结果或量化指标，例如完成产出、效率提升、准确率、用户反馈等。")
    if focus_keywords and not matched_focus:
        suggestions.append(f"可以主动关联岗位关键词：{'、'.join(focus_keywords[:3])}。")

    while len(suggestions) < 3:
        suggestions.append("结尾可以用一句话回扣目标岗位，说明这段经历能迁移到实际工作中。")

    strengths: list[str] = []
    if has_action:
        strengths.append("能说明自己的行动和参与过程。")
    if has_result:
        strengths.append("回答中出现了结果意识。")
    if matched_focus:
        strengths.append(f"和岗位关键词有一定关联：{'、'.join(matched_focus[:3])}。")
    if not strengths:
        strengths.append("已经完成了正面回答，可以继续补充细节来增强说服力。")

    feedback_text = (
        "这轮回答的方向是对的。"
        if scores["overall_score"] >= 75
        else "这轮回答已经给出了基础信息，但还需要更像面试现场的完整表达。"
    )

    polished_answer = (
        f"可以这样补强：先说明你面对的任务背景，再讲你围绕“{role}”相关能力采取的具体行动，"
        "最后用结果收尾。如果有数字、交付物或复盘结论，要优先说出来。"
    )

    return {
        **scores,
        "feedback_text": feedback_text,
        "strengths": strengths[:3],
        "suggestions": suggestions[:3],
        "polished_answer": polished_answer,
        "matched_focus": matched_focus[:5],
        "used_llm": False,
        "question": question,
    }


def _generate_llm_feedback(
    session: dict[str, Any],
    question: str,
    answer: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    llm = _create_llm()
    if llm is None:
        return fallback

    prompt = f"""
你是 AI 模拟面试官。请根据当前问题和候选人回答，给出简洁、具体、可执行的中文面试反馈。

要求：
1. 像真人面试官一样自然。
2. 不要虚构候选人没有说过的经历。
3. 必须指出回答优点和下一步改进。
4. 只输出 JSON，不要 Markdown。

目标岗位：{session.get("target_role", "目标岗位")}
岗位关键词：{json.dumps(session.get("focus_keywords", []), ensure_ascii=False)}
当前问题：{question}
候选人回答：{answer}
系统评分：{json.dumps({k: fallback[k] for k in ["overall_score", "structure_score", "relevance_score", "evidence_score", "expression_score"]}, ensure_ascii=False)}

JSON 格式：
{{
  "feedback_text": "一句总体点评",
  "strengths": ["优点1", "优点2"],
  "suggestions": ["建议1", "建议2", "建议3"],
  "polished_answer": "一段可参考的补强表达"
}}
"""

    try:
        response = llm.invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        if not parsed:
            return fallback

        return {
            **fallback,
            "feedback_text": _safe_text(parsed.get("feedback_text")) or fallback["feedback_text"],
            "strengths": parsed.get("strengths") or fallback["strengths"],
            "suggestions": parsed.get("suggestions") or fallback["suggestions"],
            "polished_answer": _safe_text(parsed.get("polished_answer")) or fallback["polished_answer"],
            "used_llm": True,
        }
    except Exception as exc:
        result = dict(fallback)
        result["agent_warning"] = f"大模型点评调用失败，当前使用规则版建议：{type(exc).__name__}"
        return result


def _build_final_report(completed_items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        item.get("feedback", {}).get("overall_score")
        for item in completed_items
        if isinstance(item.get("feedback"), dict)
    ]
    valid_scores = [int(score) for score in scores if isinstance(score, int)]
    average_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0

    all_suggestions: list[str] = []
    for item in completed_items:
        feedback = item.get("feedback", {})
        if isinstance(feedback, dict):
            all_suggestions.extend(feedback.get("suggestions", []))

    unique_suggestions = list(dict.fromkeys(all_suggestions))[:4]

    if average_score >= 80:
        summary = "整体表现较稳定，已经具备较好的岗位表达意识。"
    elif average_score >= 65:
        summary = "整体表现达到基础面试要求，继续补充证据和结果会更有竞争力。"
    else:
        summary = "当前回答还偏基础，建议先用项目经历和量化结果搭建更清晰的表达框架。"

    return {
        "average_score": average_score,
        "summary": summary,
        "improvements": unique_suggestions or [
            "每道题都尽量包含背景、行动和结果。",
            "把项目经历和目标岗位能力明确关联起来。",
        ],
        "next_actions": [
            "整理 2 个可反复讲的项目故事，每个故事写清背景、行动、结果。",
            "为目标岗位准备 8-10 个高频问题，并按 STAR 结构复盘。",
            "把回答中的结果补成可验证指标，如数据、交付物、排名或反馈。",
        ],
    }


def respond_to_interview_answer(
    session: dict[str, Any],
    question: str,
    answer: str,
    round_index: int,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    answer = _safe_text(answer)
    history = history or []
    questions = session.get("questions") or []

    fallback = _fallback_feedback(session, question, answer)
    feedback = _generate_llm_feedback(
        session=session,
        question=question,
        answer=answer,
        fallback=fallback,
    )

    completed_item = {
        "round_index": round_index,
        "question": question,
        "answer": answer,
        "feedback": feedback,
    }
    completed_items = [*history, completed_item]

    next_round = round_index + 1
    total_rounds = len(questions)
    finished = next_round > total_rounds
    next_question = "" if finished else questions[next_round - 1]

    session_update = dict(session)
    session_update["current_round"] = next_round if not finished else total_rounds
    session_update["current_question"] = next_question

    return {
        "feedback": feedback,
        "next_question": next_question,
        "next_round": next_round,
        "total_rounds": total_rounds,
        "finished": finished,
        "final_report": _build_final_report(completed_items) if finished else None,
        "session": session_update,
        "history_item": completed_item,
    }
