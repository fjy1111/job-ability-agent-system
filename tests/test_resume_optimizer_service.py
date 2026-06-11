import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_errors import LLMCallError
from app.services.resume_optimizer_service import optimize_resume


VALID_RESULT = {
    "overall_score": 80,
    "keyword_score": 82,
    "structure_score": 78,
    "summary": "简历结构清晰，建议继续强化量化成果。",
    "optimized_resume": "Java 开发工程师\n熟悉 Spring Boot 与 MySQL。",
    "strengths": ["技术栈匹配"],
    "weaknesses": ["项目量化不足"],
    "rewrite_suggestions": [],
    "action_items": ["补充性能数据"],
}


class FakeLLM:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(content=next(self.contents))


class ResumeOptimizerServiceTests(unittest.TestCase):
    def test_requests_json_output_and_accepts_valid_result(self):
        llm = FakeLLM([json.dumps(VALID_RESULT, ensure_ascii=False)])

        with patch(
            "app.services.resume_optimizer_service._create_llm",
            return_value=llm,
        ):
            result = optimize_resume("Java 简历内容", "优化 Java 后端简历")

        self.assertEqual(result["overall_score"], 80)
        self.assertEqual(
            llm.calls[0][1]["response_format"],
            {"type": "json_object"},
        )

    def test_retries_once_when_first_response_is_invalid(self):
        llm = FakeLLM([
            "不是 JSON",
            json.dumps(VALID_RESULT, ensure_ascii=False),
        ])

        with patch(
            "app.services.resume_optimizer_service._create_llm",
            return_value=llm,
        ):
            result = optimize_resume("Java 简历内容", "优化 Java 后端简历")

        self.assertEqual(result["summary"], VALID_RESULT["summary"])
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("上一次的输出未通过", llm.calls[1][0])

    def test_raises_after_two_invalid_responses(self):
        llm = FakeLLM(["不是 JSON", "{}"])

        with patch(
            "app.services.resume_optimizer_service._create_llm",
            return_value=llm,
        ):
            with self.assertRaises(LLMCallError):
                optimize_resume("Java 简历内容", "优化 Java 后端简历")


if __name__ == "__main__":
    unittest.main()
