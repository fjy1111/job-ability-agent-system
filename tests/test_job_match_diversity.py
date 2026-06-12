import unittest

from app.services.llm_ability_match_service import (
    diversify_candidate_records,
    diversify_job_matches,
    infer_job_family,
    normalize_job_title,
)


class JobMatchDiversityTests(unittest.TestCase):
    def test_normalizes_java_backend_title_variants(self):
        variants = [
            "Java后端工程师",
            "JAVA 后端开发工程师",
            "Java后端开发工程师（SpringBoot机器人平台）",
        ]

        self.assertEqual(
            {normalize_job_title(item) for item in variants},
            {"java后端工程师"},
        )

    def test_diversifies_top_five_by_job_family(self):
        matches = [
            {"job_name": "Java后端工程师", "match_score": 96},
            {"job_name": "JAVA 后端开发工程师", "match_score": 95},
            {"job_name": "软件开发工程师", "match_score": 94},
            {"job_name": "Java后端开发工程师（GIS优先）", "match_score": 93},
            {"job_name": "数据分析师", "match_score": 90},
            {"job_name": "测试工程师", "match_score": 88},
            {"job_name": "云计算工程师", "match_score": 86},
            {"job_name": "前端开发工程师", "match_score": 84},
        ]

        result = diversify_job_matches(matches, top_n=5)

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["job_name"], "Java后端工程师")
        self.assertEqual(
            len({normalize_job_title(item["job_name"]) for item in result}),
            5,
        )
        self.assertEqual(
            len({infer_job_family(item["job_name"]) for item in result}),
            5,
        )
        self.assertNotIn("JAVA 后端开发工程师", {
            item["job_name"] for item in result
        })

    def test_candidate_pool_limits_dominant_job_family(self):
        records = [
            {"job_name": f"软件开发方向{i}"}
            for i in range(40)
        ]
        records.extend(
            {"job_name": f"数据分析师方向{i}"}
            for i in range(12)
        )
        records.extend(
            {"job_name": f"测试工程师方向{i}"}
            for i in range(12)
        )

        result = diversify_candidate_records(
            records,
            top_k=40,
            target_job="软件开发工程师",
        )
        families = [infer_job_family(item["job_name"]) for item in result]

        self.assertEqual(len(result), 40)
        self.assertEqual(families.count("application_development"), 24)
        self.assertEqual(families.count("data"), 8)
        self.assertEqual(families.count("testing"), 8)

    def test_does_not_force_irrelevant_family_for_diversity(self):
        matches = [
            {"job_name": "Java后端工程师", "match_score": 95},
            {"job_name": "数据库工程师", "match_score": 90},
            {"job_name": "软件开发工程师", "match_score": 88},
            {"job_name": "产品经理", "match_score": 30},
        ]

        result = diversify_job_matches(matches, top_n=3)

        self.assertEqual(
            [item["job_name"] for item in result],
            ["Java后端工程师", "数据库工程师", "软件开发工程师"],
        )


if __name__ == "__main__":
    unittest.main()
