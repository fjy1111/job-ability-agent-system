import unittest

from app.services.resume_profile_extractor_service import (
    extract_student_profile_from_resume,
)


UNLABELED_RESUME = """
黎盛
性别：男 年龄：2 4 电话：1 5 6 7 5 4 9 9 1 4 0
户籍：广西壮族自治区 邮箱：9 1 2 4 0 3 9 0 2 @ q q . c o m
教育经历
主要课程: 地理信息系统设计与开发, 计算机图形学
毕业设计（论文）：基于 SuperMap 的管网地理信息系统的设计与开发
2 0 2 0 . 0 9 - 2 0 2 4 . 0 7
辽宁工程技术大学
地理空间信息工程 本科|
相关技能
熟练掌握 j a v a，熟悉 S p r i n g B o o t、M y S Q L 和 R e d i s。
项目经历
物流运输管理平台（微服务）
"""


class ResumeProfileExtractorTests(unittest.TestCase):
    def test_extracts_unlabeled_pdf_profile(self):
        result = extract_student_profile_from_resume(
            "帮我生成画像",
            UNLABELED_RESUME,
        )

        self.assertEqual(result["name"], "黎盛")
        self.assertEqual(result["major"], "地理空间信息工程")
        self.assertEqual(result["grade"], "本科")
        self.assertEqual(result["target_job"], "Java 后端工程师")
        self.assertIn("熟练掌握", result["skills"])

    def test_explicit_labels_take_priority(self):
        result = extract_student_profile_from_resume(
            "帮我生成画像",
            "姓名：张三\n专业：计算机科学与技术\n学历：本科\n求职意向：Java开发工程师",
        )

        self.assertEqual(result["name"], "张三")
        self.assertEqual(result["major"], "计算机科学与技术")
        self.assertEqual(result["grade"], "本科")
        self.assertEqual(result["target_job"], "Java开发工程师")


if __name__ == "__main__":
    unittest.main()
