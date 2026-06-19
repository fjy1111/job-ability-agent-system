import unittest

from app.services.ability_profile_cache_service import (
    attach_resume_cache_metadata,
    build_cached_agent_events,
    build_resume_cache_hash,
    is_matching_cached_result,
)


class AbilityProfileCacheTests(unittest.TestCase):
    def test_resume_hash_ignores_whitespace_only_changes(self):
        first = build_resume_cache_hash("姓名：张三\n技能：Java  MySQL")
        second = build_resume_cache_hash("  姓名：张三  技能：Java MySQL  ")
        self.assertEqual(first, second)

    def test_cached_result_is_split_back_into_five_agent_events(self):
        resume_hash = build_resume_cache_hash("测试简历")
        result = attach_resume_cache_metadata({
            "ability_scores": {"professional": 70},
            "dimension_insights": [{"name": "专业基础能力"}],
            "tool_calls": [{"tool_name": str(index)} for index in range(7)],
            "collaboration_log": [{"sender": str(index)} for index in range(5)],
            "workflow_steps": [
                {"step": f"0{index}", "agent": f"智能体{index}", "output": f"输出{index}"}
                for index in range(1, 6)
            ],
            "summary": "缓存总结",
        }, resume_hash)

        self.assertTrue(is_matching_cached_result(result, resume_hash))
        events = build_cached_agent_events(result)
        self.assertEqual(len(events), 5)
        self.assertEqual(
            [event["node"] for event in events],
            [
                "extract_profile",
                "score_ability",
                "analyze_profile_evidence",
                "diagnose_ability",
                "review_profile",
            ],
        )
        self.assertEqual(len(events[0]["data"]["tool_calls"]), 2)
        self.assertEqual(len(events[-1]["data"]["tool_calls"]), 7)
        self.assertTrue(all(event["cache_hit"] for event in events))


if __name__ == "__main__":
    unittest.main()
