import unittest
from unittest.mock import patch

from app.agent.diagnosis_agent import run_diagnosis_agent


STUDENT = {
    "name": "测试学生",
    "major": "计算机科学与技术",
    "grade": "本科",
    "target_job": "Java 后端工程师",
    "skills": "Java、Spring Boot、MySQL、Redis、Git",
    "projects": "使用 Spring Boot 和 Redis 完成校园招聘系统。",
    "competitions": "",
    "certificates": "英语四级",
    "self_intro": "希望从事 Java 后端开发，并持续提升工程实践能力。",
}

DIMENSIONS = {
    "professional": "专业基础能力",
    "practice": "技术实践能力",
    "tools": "工具技能能力",
    "career": "职业发展能力",
}


def build_collaborative_draft():
    scores = {
        "professional": 75,
        "practice": 80,
        "tools": 82,
        "career": 72,
    }
    return {
        "ability_scores": scores,
        "score_evidence": {
            key: [f"{name}证据1", f"{name}证据2"]
            for key, name in DIMENSIONS.items()
        },
        "recognized_skills": ["Java", "Spring Boot", "MySQL", "Redis", "Git"],
        "assessment_summary": "具备较完整的 Java 后端基础和项目实践。",
        "advantages": ["后端技术栈较完整", "具备项目实践", "目标岗位清晰"],
        "weaknesses": ["量化成果不足", "复杂系统证据较少", "竞赛经历缺失"],
        "dimension_actions": {
            key: f"继续补充{name}的可验证成果。"
            for key, name in DIMENSIONS.items()
        },
        "quality_notes": [
            "四维分数均有对应证据。",
            "画像未引入简历之外的经历。",
        ],
    }


class DiagnosisSingleLlmWorkflowTests(unittest.TestCase):
    def test_one_llm_call_keeps_full_workflow_and_tools(self):
        draft = build_collaborative_draft()

        with patch(
            "app.agent.diagnosis_agent._invoke_json_agent",
            return_value=(draft, True, ""),
        ) as invoke_agent:
            result = run_diagnosis_agent(STUDENT)

        invoke_agent.assert_called_once()
        self.assertEqual(
            invoke_agent.call_args.args[0],
            "综合画像协作智能体",
        )
        self.assertEqual(result["llm_agents"], ["综合画像协作智能体"])
        self.assertEqual(len(result["workflow_steps"]), 5)
        self.assertEqual(len(result["tool_calls"]), 7)
        self.assertEqual(len(result["collaboration_log"]), 5)
        self.assertEqual(len(result["evidence_cards"]), 4)
        self.assertEqual(len(result["dimension_insights"]), 4)
        self.assertEqual(len(result["quality_review"]), 4)
        self.assertIn(draft["quality_notes"][0], result["quality_review"])


if __name__ == "__main__":
    unittest.main()
