from __future__ import annotations

from typing import Any


DEFAULT_QUESTIONS = [
    "请先做一个简短的自我介绍，并说明你为什么想投递这个岗位。",
    "请结合一个项目经历，说明你在其中承担的职责、使用的技术和最终成果。",
    "如果入职后遇到一个你暂时不会的技术问题，你会如何拆解和解决？",
]


def build_interview_session(
    target_role: str = "",
    job_description: str = "",
    resume_text: str = "",
) -> dict[str, Any]:
    role = target_role.strip() or "目标岗位"
    return {
        "target_role": role,
        "job_description": job_description.strip(),
        "resume_text": resume_text.strip(),
        "questions": DEFAULT_QUESTIONS,
        "current_round": 1,
        "total_rounds": len(DEFAULT_QUESTIONS),
        "current_question": DEFAULT_QUESTIONS[0],
        "opening_message": f"你好，我将围绕{role}进行模拟面试。"
    }


def respond_to_interview_answer(
    session: dict[str, Any],
    question: str,
    answer: str,
    round_index: int,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    questions = session.get("questions") or DEFAULT_QUESTIONS
    next_index = round_index + 1
    finished = next_index > len(questions)

    feedback = {
        "score": 72 if len(answer.strip()) >= 80 else 62,
        "highlights": [
            "回答已经覆盖了问题核心",
            "可以继续结合具体项目和量化结果增强说服力"
        ],
        "improvements": [
            "建议按背景、行动、结果组织回答",
            "尽量补充技术细节、个人贡献和最终效果"
        ],
        "summary": "当前为本地兜底面试点评；接入大模型后可生成更细致的追问和评分。"
    }

    return {
        "feedback": feedback,
        "finished": finished,
        "next_question": "" if finished else questions[next_index - 1],
        "next_round": next_index,
        "history": [
            *(history or []),
            {
                "round": round_index,
                "question": question,
                "answer": answer,
                "feedback": feedback,
            }
        ]
    }
