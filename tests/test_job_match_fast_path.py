import os
import unittest
from unittest.mock import patch

from app.services.ability_match_service import keyword_hits
from app.services.llm_ability_match_service import calculate_job_match
from app.services.llm_gap_path_agent import generate_top5_gap_paths
from app.services.job_vector_service import (
    build_job_vector_index,
    clear_job_vector_index_cache,
)


STUDENT = {
    "name": "测试学生",
    "major": "计算机科学",
    "grade": "本科",
    "target_job": "Java 后端开发",
    "skills": "Java、Spring Boot、MySQL、Redis、Git、Linux",
    "projects": "校园招聘系统，使用 Spring Boot 和 Redis 优化岗位查询",
    "competitions": "",
    "certificates": "",
    "self_intro": "希望从事 Java 后端开发。",
}

JOBS = [
    {
        "job_name": "Java 后端开发工程师",
        "company_name": "测试公司",
        "hiring_city": "上海",
        "educational_requirements": "本科",
        "required_skills_json": '["Java", "Spring Boot", "MySQL", "Redis"]',
        "related_projects_json": '["招聘系统", "接口开发"]',
        "recommended_courses_json": '["数据库"]',
        "recommended_certificates_json": "[]",
        "salary_range": "面议",
    },
    {
        "job_name": "视觉算法工程师",
        "required_skills_json": '["Python", "PyTorch", "计算机视觉"]',
        "related_projects_json": '["图像分类"]',
        "recommended_courses_json": '["机器学习"]',
        "recommended_certificates_json": "[]",
    },
]


class JobMatchFastPathTests(unittest.TestCase):
    def tearDown(self):
        clear_job_vector_index_cache()

    def test_default_job_match_does_not_create_llm_client(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JOB_MATCH_USE_LLM", None)
            with patch(
                "app.services.llm_ability_match_service._create_llm"
            ) as create_llm:
                result = calculate_job_match(STUDENT, JOBS)

        create_llm.assert_not_called()
        self.assertEqual(result[0]["job_name"], "Java 后端开发工程师")
        self.assertFalse(result[0]["used_llm"])
        self.assertGreater(result[0]["match_score"], result[1]["match_score"])

    def test_default_gap_path_is_local_and_has_three_stages(self):
        matches = calculate_job_match(STUDENT, JOBS)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JOB_MATCH_GAP_PATH_USE_LLM", None)
            with patch(
                "app.services.llm_gap_path_agent._create_llm"
            ) as create_llm:
                result = generate_top5_gap_paths(STUDENT, matches)

        create_llm.assert_not_called()
        self.assertFalse(result["used_llm"])
        self.assertEqual(len(result["top5_gap_paths"]), 2)
        self.assertEqual(
            len(result["top5_gap_paths"][0]["learning_stages"]),
            3,
        )

    def test_vector_cache_rebinds_current_request_records(self):
        first_records = [{"id": 1, **JOBS[0]}]
        second_records = [{"id": 1, **JOBS[0]}]

        first_index = build_job_vector_index(first_records)
        second_index = build_job_vector_index(second_records)

        self.assertIs(first_index[0]["record"], first_records[0])
        self.assertIs(second_index[0]["record"], second_records[0])
        self.assertIsNot(second_index[0]["record"], first_records[0])

    def test_pdf_spaced_technical_terms_are_recognized(self):
        hits = keyword_hits(
            "熟悉 J a v a、M y S Q L、R e d i s 和 S p r i n g B o o t",
            ["Java", "MySQL", "Redis", "Spring Boot"],
        )

        self.assertEqual(
            hits,
            ["Java", "MySQL", "Redis", "Spring Boot"],
        )


if __name__ == "__main__":
    unittest.main()
