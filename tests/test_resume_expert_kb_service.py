import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import resume_expert_kb_service as service


class ResumeExpertKbServiceTests(unittest.TestCase):
    def test_retrieves_rules_by_resume_keywords(self):
        rules = service.retrieve_resume_expert_rules(
            "项目经历：Linux C++ 文件服务器，使用 Redis 缓存，支持搜索推荐。",
            max_rules=8,
        )
        titles = {rule["title"] for rule in rules}

        self.assertLessEqual(len(rules), 8)
        self.assertIn("Linux/C++ 项目要突出系统编程能力", titles)
        self.assertIn("Redis 缓存需要说明用途", titles)
        self.assertIn("搜索推荐类项目需要突出索引与排序", titles)

    def test_returns_default_rules_when_no_keyword_matches(self):
        rules = service.retrieve_resume_expert_rules("我喜欢阅读与运动，沟通主动。")

        self.assertEqual(
            [rule["title"] for rule in rules],
            [
                "项目经历需要补充个人角色",
                "尽量补充量化结果",
                "技术亮点要写成“技术方案 + 解决问题 + 效果”",
            ],
        )

    def test_missing_rule_file_degrades_to_empty_list(self):
        with patch.object(
            service,
            "RULE_FILE_CANDIDATES",
            (Path("missing_resume_expert_rules.json"),),
        ):
            self.assertEqual(
                service.retrieve_resume_expert_rules("Redis 项目经历"),
                [],
            )


if __name__ == "__main__":
    unittest.main()
