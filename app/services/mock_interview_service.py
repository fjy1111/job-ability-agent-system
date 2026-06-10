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
except Exception:  # pragma: no cover
    ChatOpenAI = None

from app.services.llm_errors import LLMCallError


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


DEFAULT_INTERVIEWER = "林老师"
QUESTION_COUNT = 6

ROLE_KEYWORDS = [
    "Python", "Java", "Spring Boot", "FastAPI", "Django", "MySQL", "Redis",
    "Linux", "Docker", "Git", "Vue", "React", "TypeScript", "JavaScript",
    "数据分析", "机器学习", "深度学习", "算法", "大模型", "LangChain",
    "LangGraph", "项目管理", "需求分析", "沟通表达", "自动化测试", "接口测试",
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def _safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        raise LLMCallError()

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise LLMCallError()
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise LLMCallError()

    if not isinstance(data, dict):
        raise LLMCallError()
    return data


def _create_llm() -> Any:
    if ChatOpenAI is None:
        raise LLMCallError()
    if os.getenv("USE_LLM", "true").lower() != "true":
        raise LLMCallError()

    api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("LLM_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or os.getenv("DASHSCOPE_BASE_URL", "").strip()
    )
    model = (
        os.getenv("INTERVIEW_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or os.getenv("DASHSCOPE_MODEL", "").strip()
    )

    if not api_key or not model:
        raise LLMCallError()

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.35,
        timeout=60,
        max_retries=0,
    )


def _pick_focus_keywords(text: str, limit: int = 8) -> list[str]:
    focus = [keyword for keyword in ROLE_KEYWORDS if keyword.lower() in text.lower()]
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


def _generate_llm_questions(role: str, job_description: str, resume_text: str) -> list[str]:
    prompt = f"""
你是严谨但友好的 AI 模拟面试官。请根据目标岗位、JD 和候选人简历，生成 {QUESTION_COUNT} 个循序渐进的中文面试问题。

要求：
1. 问题要像真人面试官一样自然、具体。
2. 覆盖自我介绍、项目经历、岗位技能、问题拆解、短板反思、录用理由。
3. 不要编造候选人没有提供的信息。
4. 只输出 JSON，不要 Markdown。

目标岗位：{role}
岗位信息：{job_description[:2000]}
简历信息：{resume_text[:3000]}

JSON 格式：
{{
  "questions": ["问题1", "问题2", "问题3", "问题4", "问题5", "问题6"]
}}
"""
    try:
        response = _create_llm().invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        questions = [
            _safe_text(item)
            for item in parsed.get("questions", [])
            if _safe_text(item)
        ]
        if len(questions) < QUESTION_COUNT:
            raise LLMCallError()
        return questions[:QUESTION_COUNT]
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


def build_interview_session(
    target_role: str,
    job_description: str,
    resume_text: str,
) -> dict[str, Any]:
    role = _infer_role(target_role, job_description, resume_text)
    combined_context = f"{role} {job_description} {resume_text}"
    focus_keywords = _pick_focus_keywords(combined_context)
    questions = _generate_llm_questions(
        role=role,
        job_description=job_description,
        resume_text=resume_text,
    )

    return {
        "session_id": str(uuid.uuid4()),
        "interviewer_name": DEFAULT_INTERVIEWER,
        "target_role": role,
        "job_description": job_description[:2000],
        "resume_text": resume_text[:3000],
        "focus_keywords": focus_keywords,
        "questions": questions,
        "total_rounds": len(questions),
        "current_round": 1,
        "current_question": questions[0],
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
    matched_focus = [keyword for keyword in focus_keywords if keyword and keyword.lower() in answer.lower()]

    structure_score = 45 + (15 if has_context else 0) + (20 if has_action else 0) + (20 if has_result else 0)
    relevance_score = 58 + min(len(matched_focus), 4) * 8
    relevance_score += 8 if length >= 80 else 0
    relevance_score -= 18 if length < 35 else 0
    evidence_score = 42 + (25 if has_action else 0) + (25 if has_result else 0) + (8 if re.search(r"\d", answer) else 0)
    expression_score = 55 + (22 if 80 <= length <= 450 else 10 if length > 450 else 0)
    expression_score += 8 if "。" in answer or "，" in answer else 0

    scores = {
        "structure_score": _clamp(structure_score),
        "relevance_score": _clamp(relevance_score),
        "evidence_score": _clamp(evidence_score),
        "expression_score": _clamp(expression_score),
    }
    scores["overall_score"] = round(sum(scores.values()) / len(scores))
    return scores


def _generate_llm_feedback(
    session: dict[str, Any],
    question: str,
    answer: str,
    scores: dict[str, int],
) -> dict[str, Any]:
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
结构化评分：{json.dumps(scores, ensure_ascii=False)}

JSON 格式：
{{
  "feedback_text": "一句总体点评",
  "strengths": ["优点1", "优点2"],
  "suggestions": ["建议1", "建议2", "建议3"],
  "polished_answer": "一段可参考的补强表达"
}}
"""
    try:
        response = _create_llm().invoke(prompt)
        parsed = _safe_json_loads(str(response.content))
        required = ["feedback_text", "strengths", "suggestions", "polished_answer"]
        if any(key not in parsed for key in required):
            raise LLMCallError()
        return {
            **scores,
            "feedback_text": _safe_text(parsed.get("feedback_text")),
            "strengths": parsed.get("strengths") if isinstance(parsed.get("strengths"), list) else [],
            "suggestions": parsed.get("suggestions") if isinstance(parsed.get("suggestions"), list) else [],
            "polished_answer": _safe_text(parsed.get("polished_answer")),
            "used_llm": True,
            "question": question,
        }
    except LLMCallError:
        raise
    except Exception:
        raise LLMCallError()


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

    return {
        "average_score": average_score,
        "summary": "本次模拟面试已完成，建议把高频问题、项目证据和量化结果继续沉淀成固定表达。",
        "improvements": list(dict.fromkeys(all_suggestions))[:4],
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
    focus_keywords = session.get("focus_keywords", [])
    if not isinstance(focus_keywords, list):
        focus_keywords = []

    feedback = _generate_llm_feedback(
        session=session,
        question=question,
        answer=answer,
        scores=_score_answer(answer, focus_keywords),
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
