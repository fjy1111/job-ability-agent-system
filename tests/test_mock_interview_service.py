import json
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.mock_interview_service import (
    build_interview_session,
    build_written_exam,
    grade_written_exam,
    respond_to_interview_answer,
)


class FakeLlm:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, _prompt):
        return SimpleNamespace(content=json.dumps(next(self.responses), ensure_ascii=False))


class MockInterviewServiceTests(unittest.TestCase):
    def test_builds_randomized_six_question_written_exam(self):
        question_pool = [
            {
                "question": f"Java 岗位笔试题 {index}",
                "options": [f"选项 {index}-{letter}" for letter in "ABCD"],
                "correct_index": 0,
                "explanation": f"第 {index} 题解析",
                "category": "Java 基础",
            }
            for index in range(1, 11)
        ]
        with patch.dict(os.environ, {"INTERVIEW_WRITTEN_USE_LLM": "true"}), patch(
            "app.services.mock_interview_service._create_llm",
            return_value=FakeLlm([{"questions": question_pool}]),
        ):
            exam = build_written_exam(
                target_role="Java 后端工程师",
                job_description="熟悉 Spring Boot、MySQL 和 Redis",
                resume_text="使用 Spring Boot 完成校园招聘系统",
                rng=random.Random(7),
            )

        self.assertEqual(exam["target_role"], "Java 后端工程师")
        self.assertEqual(exam["total_questions"], 6)
        self.assertEqual(len({item["question"] for item in exam["questions"]}), 6)
        self.assertTrue(all(len(item["options"]) == 4 for item in exam["questions"]))
        self.assertTrue(all(0 <= item["correct_index"] <= 3 for item in exam["questions"]))

    def test_grades_written_exam_and_marks_unanswered_question(self):
        exam = {
            "questions": [
                {
                    "id": "q1",
                    "question": "正确选项是什么？",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 1,
                    "explanation": "B 是正确答案。",
                    "category": "基础",
                },
                {
                    "id": "q2",
                    "question": "第二题？",
                    "options": ["A2", "B2", "C2", "D2"],
                    "correct_index": 2,
                    "explanation": "C2 是正确答案。",
                    "category": "实践",
                },
            ]
        }

        result = grade_written_exam(exam, {"q1": 1})

        self.assertEqual(result["correct_count"], 1)
        self.assertEqual(result["score"], 50)
        self.assertEqual(result["details"][1]["selected_answer"], "未作答")
        self.assertFalse(result["details"][1]["is_correct"])

    def test_reuses_written_question_pool_for_same_resume_and_role(self):
        question_pool = [
            {
                "question": f"缓存题 {index}",
                "options": [f"选项 {index}-{letter}" for letter in "ABCD"],
                "correct_index": 0,
                "explanation": "缓存解析",
                "category": "缓存测试",
            }
            for index in range(1, 7)
        ]
        with patch.dict(os.environ, {"INTERVIEW_WRITTEN_USE_LLM": "true"}), patch(
            "app.services.mock_interview_service._generate_llm_written_question_pool",
            return_value=question_pool,
        ) as generator:
            build_written_exam(
                target_role="缓存测试工程师",
                job_description="缓存岗位描述",
                resume_text="唯一缓存简历文本",
                rng=random.Random(1),
            )
            build_written_exam(
                target_role="缓存测试工程师",
                job_description="缓存岗位描述",
                resume_text="唯一缓存简历文本",
                rng=random.Random(2),
            )

        generator.assert_called_once()

    def test_builds_interview_session_with_first_question(self):
        questions = [f"面试问题 {index}" for index in range(1, 7)]
        with patch(
            "app.services.mock_interview_service._create_llm",
            return_value=FakeLlm([{"questions": questions}]),
        ):
            session = build_interview_session(
                target_role="Java 后端工程师",
                job_description="熟悉 Spring Boot、MySQL 和 Redis",
                resume_text="使用 Spring Boot 完成校园招聘系统",
            )

        self.assertEqual(session["target_role"], "Java 后端工程师")
        self.assertEqual(session["total_rounds"], 6)
        self.assertEqual(session["current_question"], questions[0])

    def test_scores_answer_and_returns_next_question(self):
        questions = [f"面试问题 {index}" for index in range(1, 7)]
        session = {
            "target_role": "Java 后端工程师",
            "focus_keywords": ["Java", "Spring Boot"],
            "questions": questions,
            "current_round": 1,
            "current_question": questions[0],
        }
        feedback = {
            "feedback_text": "回答结构清晰。",
            "strengths": ["说明了负责内容"],
            "suggestions": ["补充量化结果"],
            "polished_answer": "我负责接口开发，并将查询耗时降低 30%。",
        }

        with patch(
            "app.services.mock_interview_service._create_llm",
            return_value=FakeLlm([feedback]),
        ):
            result = respond_to_interview_answer(
                session=session,
                question=questions[0],
                answer="我负责使用 Java 和 Spring Boot 完成接口开发，最终将查询耗时降低 30%。",
                round_index=1,
                history=[],
            )

        self.assertTrue(result["feedback"]["used_llm"])
        self.assertEqual(result["next_question"], questions[1])
        self.assertFalse(result["finished"])


if __name__ == "__main__":
    unittest.main()
